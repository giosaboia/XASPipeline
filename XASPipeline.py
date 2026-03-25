
import argparse, h5py, inspect, logging, pathlib, struct, sys, time, yaml

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy as sp

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from matplotlib import colormaps
from multiprocessing.pool import ThreadPool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, BeforeValidator
from lmfit import Parameters, minimize, report_fit
from typing import Annotated, ClassVar, Literal, Optional, Any


def deltaE2k(deltaE):
    return np.sqrt(2 * sp.constants.m_e * sp.constants.eV * deltaE / np.square(sp.constants.hbar)) * 1E-10

def k2deltaE(k):
    return np.square(k* 1E10) * np.square(sp.constants.hbar) / (2 * sp.constants.m_e * sp.constants.eV)

def abs2AthenaRep(val: float) -> str:
    if val > 0.1:
        return " %.8f" %val
    else:
        return " %17.10E" %val
    
    bits = struct.unpack('<Q', struct.pack('<d', val))[0]

    sign = (bits >> 63) & 1
    exponent = (bits >> 52) & 0x7FF
    mantissa = bits & 0xFFFFFFFFFFFFF

    sign = "-" if sign else " "
    return f"  {sign}0.{str(mantissa)[:8]}E{(exponent-1022):03}"

def readDatCols(logger, file_path, cols) -> np.array:
    logger.info(f"Reading {file_path}")
    with open(file_path, 'r') as f:
        data, t = [], 0
        for line in f:
            if (line[0] == "#") | (line == "\n"):
                if "Scan started" not in line:
                    continue

                date_str = line.split(":", 1)[1].strip()
                t = datetime.strptime(date_str, r"%d-%m-%Y %H:%M:%S").timestamp()
                continue

            nums = line.split()
            data.append([float(nums[i]) for i in cols])
        return t, np.array(data)
    
def readNorm(file_path: pathlib.Path, useCol: int | str = "flat") -> tuple[np.ndarray, np.ndarray]:
    energies, absorption = [], []
    headerline = ""
    if isinstance(useCol, int):
        startReadout = True
    else:
        startReadout = False
    with open(file_path, 'r') as f:
        for l in f:
            if l[0] == "#":
                headerline = l
            else:
                if not startReadout:
                    try:
                        useCol = headerline.split()[1:].index(useCol)
                    except ValueError:
                        useCol = 1
                    startReadout = True

                words = l.split()
                energies.append(float(words[0]))
                absorption.append(float(words[useCol]))

    return np.array(energies), np.array(absorption)

#region dataclass
class XASPara:
    def __init__(self, edge:str, element: str, edge_pos: float, pre_edge_range: tuple[float], post_edge_range: tuple[float]):
        self.edge: str = edge
        self.element: str = element
        self.edge_pos: float = edge_pos
        self._pre_edge_range: tuple = pre_edge_range
        self._post_edge_range: tuple = post_edge_range
        # self.beamline: str = beamline
        # self.path: pathlib.Path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
        # self.name: str

    @property
    def pre_edge_range(self) -> tuple[float]:
        return tuple(self.edge_pos + e if not isinstance(e, type(None)) else None for e in self._pre_edge_range)

    @property
    def post_edge_range(self) -> tuple[float]:
        return tuple(self.edge_pos + e if not isinstance(e, type(None)) else None for e in self._post_edge_range)

@dataclass
class XASData:
    energies: npt.NDArray[np.floating]
    times: npt.NDArray[np.floating]
    absorption: npt.NDArray[np.floating]
    normalized: bool = False

    def __post_init__(self):
        self.removeNan()
        self.validate()

    @classmethod
    def extracter(cls, logger: logging.Logger, directory: pathlib.Path, name: str, mode: str):
        extract_func = None
        match mode:
            case "Balder":
                files = list(directory.glob(f"*{name}*.h5"))
                extract_func = lambda l, f: cls.extract_data_hdf5(l, f)
            case "P65-T":
                files = list(directory.glob(f"*{name}*.dat"))
                extract_func = lambda l, f: cls.extract_data_dat(l, f, [[1], [9] ,[10]], True)
            case "P65-F":
                files = list(directory.glob(f"*{name}*.dat"))
                extract_func = lambda l, f: cls.extract_data_dat(l, f, [[1], [12], [9]], False)
            case "P65-SDD":
                files = list(directory.glob(f"*{name}*.dat"))
                extract_func = lambda l, f: cls.extract_data_dat(l, f, [[1], [13, 14, 15, 16] ,[9]], False)
        if len(files) == 0:
            raise ValueError(f"No files with name '{name}' in '{directory}' found.")

        return extract_func(logger, files)

    @classmethod
    def extract_data_hdf5(cls, logger: logging.Logger, files: list[pathlib.Path]):
        def conv_time(date_str: str):
            try:
                return datetime.strptime(date_str, r"%Y-%m-%d %H:%M:%S.%f").timestamp()
            except ValueError:
                return np.nan

        for i, file in enumerate(files):
            with h5py.File(file, 'r') as run_datafile:
                e = np.array(run_datafile['energy'][0,:-1])
                t = np.array([conv_time(str(t)[2:-1]) for t in run_datafile['time']])
                m = np.array(run_datafile['mu'])[:,:-1]

            if i == 0:
                energy = e
                times = t
                mu = m
            else:
                if energy.shape != e.shape:
                    raise ValueError(f"Files cover different energy regimes: {energy.shape} != {e.shape}")
                elif any(energy - e > 0.1):
                    raise ValueError(f"Files cover different energy regimes: max delta: {max(abs(energy-e))}")
                
                times = np.concat([times, t], axis = 0)
                mu = np.concat([mu, m], axis = 0)

        return cls(energy, times - times[0], mu)
    
    @classmethod
    def extract_data_dat(cls, logger: logging.Logger, files: list[pathlib.Path], cols: list, log: bool):
        colSlicer = [num for sublist in cols for num in sublist]
        e_slice, num_slice, de_slice = slice(0, len(cols[0])+1), slice(len(cols[0]) + 1, len(cols[0]) + len(cols[1]) + 1), slice(len(cols[0]) + len(cols[1]) + 1, len(cols[0]) + len(cols[1]) + len(cols[2]) + 1)
        for i, file in enumerate(files):
            t, data = readDatCols(logger, file, colSlicer)
            e, m = np.sum(data[:, e_slice], axis=1), np.sum(data[:,num_slice], axis=1) / np.sum(data[:,de_slice], axis=1)

            if i == 0:
                energy, times, mu = e, [t], m
            else:
                if energy.shape != e.shape:
                    raise ValueError(f"Files cover different energy regimes: {energy.shape} != {e.shape}")
                elif any(energy - e > 0.1):
                    raise ValueError(f"Files cover different energy regimes: max delta: {max(abs(energy-e))}")
                
                times.append(t)
                mu = np.concat([mu, m], axis = 0)
        if log:
            return cls(times - times(0), energy, np.log(mu))
        else:
            return cls(times - times(0), energy, mu)
        
    def removeNan(self):
        has_nan = np.isnan(self.absorption).any(axis=1)
        self.absorption = self.absorption[~has_nan]
        self.times = self.times[~has_nan]
        
    def validate(self):
        """Validate shapes on creation."""
        self._validate_array(self.times, "times", expected_dim=1)
        self._validate_array(self.energies, "energies", expected_dim=1)
        self._validate_array(self.absorption, "absorption", expected_dim=2)

        M, N = self.absorption.shape
        if (len(self.times) != M) or (len(self.energies) != N):
            raise ValueError(f"Expected absorption data of shape {len(self.times)}, {len(self.energies)}, recieved {self.absorption.shape}")
        
    def _validate_array(self, arr: npt.NDArray, name: str, expected_dim: int):
        """Helper to check if an array is valid, of right dimension and numeric."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Field '{name}' must be a numpy ndarray, got {type(arr)}")
        
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Field '{name}' must contain numeric data, got {arr.dtype}")

        if arr.ndim != expected_dim:
            raise ValueError(f"Field '{name}' must be {expected_dim}D, but got {arr.ndim}D with shape {arr.shape}")
        
        if arr.size == 0:
            raise ValueError(f"Field '{name}' cannot be empty.")
    
    def toNORMind(self, path:pathlib.Path, name: str, para:XASPara, comment:str|None):
        for i in range(len(self.times)):
            with open(path / (f"{name}_{i:04d}.norm"), 'w') as f:
                f.write(f"# XAS Data processed by XASPipeline\n# Shape: (1, {len(self.energies)})\n")
                if comment is not None:
                    f.write("# " + comment + "\n")
                f.write(f"# Element.edge:                  {para.edge}\n")
                f.write(f"# Element.symbol:                {para.element}\n")
                f.write(f"# Column.1:                      energy eV\n")
                f.write(f"# Column.2:                      {int(self.times[i])}\n")
                f.write("# ///\n#------------------------\n")
                f.write("# "+"".join(["%-17s" %"e", "%-17s" %"norm"]) + "\n")
                for energy, abs_vals in zip(self.energies, self.absorption[i].T):
                    line_elements = [f" {energy:10.4f}    ", abs2AthenaRep(abs_vals)]
                    f.write("".join(line_elements) + "\n")
        
    def toNORM(self, path:pathlib.Path, name: str, para:XASPara, comment:str|None, mode:str):
        with open(path / (name + ".norm"), 'w') as f:
            f.write(f"# XAS Data processed by XASPipeline\n# Shape: ({len(self.times)}, {len(self.energies)})\n")
            if comment is not None:
                f.write("# " + comment + "\n")
            f.write(f"# Element.edge:                  {para.edge}\n")
            f.write(f"# Element.symbol:                {para.element}\n")
            f.write(f"# Column.1:                      energy eV\n")
            for i, t in enumerate(self.times):
                f.write(f"# Column.{i+2}:                      {int(t)}\n")
            f.write("# ///\n#------------------------\n")
            f.write("# "+"".join(["%-17s" %"e"] + ["%-17s" %f"norm{(int(t))}" for t in self.times]) + "\n")
            for energy, abs_vals in zip(self.energies, self.absorption.T):
                line_elements = [f" {energy:10.4f}    "] + [abs2AthenaRep(val) for val in abs_vals.tolist()]
                f.write("".join(line_elements) + "\n")
    
    def genKspace(self, e0):
        if not self.normalized:
            raise ValueError(f"absorption data has to be normalized before calling genKspace")
        start_idx = np.searchsorted(self.energies, e0)
        return deltaE2k(self.energies[start_idx:] - e0), self.absorption[:, start_idx:]    
        
    def energyRange2idx(self, low, up) -> slice:        
        if isinstance(low, type(None)) & isinstance(up, type(None)):
            return slice(0, -1)
        elif isinstance(low, type(None)):
            return slice(0, np.argmax(self.energies > up) -1)
        elif isinstance(up, type(None)):
            return slice(np.argmax(self.energies > low), -1)
        
        if low > up:
            raise ValueError("energyRange2idx requires the low value to be smaller than the up value")
        
        return slice(np.argmax(self.energies > low), np.argmax(self.energies > up) -1)
#endregion

#region XASReference
class XASRef(BaseModel):
    mu: Optional[np.ndarray] = None
    name: str
    source_e: Optional[np.ndarray] = None
    source_mu: Optional[np.ndarray] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_norm(cls, path: pathlib.Path) -> "XASRef":
        name = path.stem
        energy, mu = readNorm(path)
        return cls(name = name, source_e = energy, source_mu = mu)

    def resample(self, target_energy: np.ndarray):
        if self.mu is not None:
            if len(self.mu) == len(target_energy):
                return
        if self.source_e is not None and self.source_mu is not None:
            self.mu = self._rebin(target_energy)
        else:
            raise ValueError("XASRef has not data and no source to resample from!")
    
    def _rebin(self, target_energy: np.ndarray):
        midpoints = (target_energy[:-1] + target_energy[1:]) / 2
        edges = np.concatenate([
            [target_energy[0] - (target_energy[1] - target_energy[0]) / 2],
            midpoints,
            [target_energy[-1] + (target_energy[-1] - target_energy[-2]) / 2]
        ])

        cum_integral = sp.integrate.cumulative_trapezoid(self.source_mu, x = self.source_e, initial=0)
        resampled_integral = np.interp(edges, self.source_e, cum_integral)
        return np.diff(resampled_integral) / np.diff(edges)

def str2Ref(x: Any) -> XASRef:
    return XASRef.from_norm(pathlib.Path(x)) if isinstance(x, (str,pathlib.Path)) else x

#region PipelineContex
class PipelineContext(BaseModel):
    path: pathlib.Path
    exp_name: str
    beamline: Literal['Balder', 'P65_T', 'P65_F', 'P65_SDD']
    plot: Optional[bool]
#endregion

#region Processor
class Processor(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    logger: ClassVar[logging.Logger] = logging.getLogger("XAS-Pipeline")

    name: str = None
    plot: bool = False

    para: ClassVar[XASPara] = None
    _data: XASData = PrivateAttr(default=None)

    @classmethod
    def with_context(cls, data: dict, context: PipelineContext):
        for field_name in cls.model_fields:
            if field_name not in data and hasattr(context, field_name):
                val = getattr(context, field_name)
                if val is not None:
                    data[field_name] = val

        if data.get("name") is None:
            data["name"] = cls.__name__

        return cls.model_validate(data)
#endregion

#region Preprocessors
class Preprocessor(Processor):

    def transform(self, data: XASData) -> XASData:
        self._data = data
        self.logger.info(f"Preprocessor {self.name} started")
        start_time = time.time()
        
        self._transform()
        try:
            self._data.validate()
        except (ValueError, TypeError) as e:
            error_msg = f"Validation failed after '{self.name}': {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        duration = time.time() - start_time
        self.logger.info(f"Preprocessor {self.name} finished after {round(duration, 4)} s.")
        if self.plot:
            self._plot()
        return self._data

    @abstractmethod
    def _transform(self) -> None:
        pass

    def _plot(self) -> None:
        step = max(int(len(self._data.times)/20), 1)
        plt.subplots()
        for i, spectra in enumerate(self._data.absorption[::step, :]):
            plt.plot(self._data.energies, spectra, label=f"{self._data.times[step*i]:.0f}")
        plt.title(f"Preprocessor {self.name}")
        plt.legend(frameon=False, loc="lower right", ncols=2)
        plt.show()

class Normalizer(Preprocessor):
    """
    Normlizes the spectra.
    Args:
        post_order (int): Polynom-order of the post_edge_line fit (Default: 3)
    """
    post_order: int = 3
    def _transform(self) -> None:
        pre_edge_slice = self._data.energyRange2idx(*self.para.pre_edge_range)
        post_edge_slice = self._data.energyRange2idx(*self.para.post_edge_range)

        coeffs, residual, rank, s = np.linalg.lstsq(
            np.column_stack([self._data.energies[pre_edge_slice], np.ones_like(self._data.energies[pre_edge_slice])]),
            self._data.absorption[:,pre_edge_slice].T
        )
        pre_cor = np.outer(coeffs[0], self._data.energies) + coeffs[1][:,None]
        self._data.absorption -= pre_cor

        coeffs, residual, rank, s = np.linalg.lstsq(
            np.vander(self._data.energies[post_edge_slice], self.post_order + 1),
            self._data.absorption[:,post_edge_slice].T
        )
        post_cor = np.dot(np.vander(self._data.energies, self.post_order + 1), coeffs).T
        self._data.absorption /= post_cor
        rows_with_neg = (post_cor < 0).any(axis=1)

        self.logger.info(f"Preprocessor {self.name} removed {np.sum(rows_with_neg)} from {len(self._data.times)} due to negative values in post_line spline")
        self._data.absorption = self._data.absorption[~rows_with_neg]
        self._data.times = self._data.times[~rows_with_neg]
        self._data.normalized = True 

    def _plot(self) -> None:
        fig, ax = plt.subplots(figsize=(8,4))
        step = max(int(len(self._data.times)/20), 1)
        for i in range(0,len(self._data.times),step):
            ax.plot(self._data.energies, self._data.absorption[i])
        
        ax.axhline(1)
        for e in (val for val in self.para.pre_edge_range if val is not None):
            ax.axvline(e, ls="dashed", color="grey")
        for e in (val for val in self.para.post_edge_range if val is not None):
            ax.axvline(e, ls="dashed", color="grey")
        plt.show()


class NoiseFilter(Preprocessor):
    """
    Filters out Spectra with large noise-to-signal ratio. After normalization the spectra with low signal exhibit large noise.
    Args:
        gate (float): multipier of the median RMS that serves as the cutoff (Default: 3)
    """
    gate: float = 3

    def _transform(self) -> None:
        rms_noise = np.sqrt(np.mean(np.square(np.diff(self._data.absorption, n=2, axis=1)), axis=1))

        threshold = self.gate * np.median(rms_noise)
        mask = rms_noise < threshold
        if self.plot:
            plt.subplots()
            plt.title(f"Preprocessor {self.name}")
            # num_bins = int(np.ceil(rms_noise.max() / (np.median(rms_noise) / 8))) + 1
            # plt.hist(rms_noise, log=True, bins=[i * np.median(rms_noise) / 8 for i in range(num_bins)])
            plt.hist(rms_noise, log=True)
            plt.axvline(threshold, color="black", ls="dotted")
            plt.show()

        if np.sum(~mask):
            init_num = len(self._data.times)
            self._data.absorption = self._data.absorption[mask, :]
            self._data.times = self._data.times[mask]
            self.logger.info(f"Preprocessor {self.name} removed {np.sum(~mask)} from {init_num}")

class Savgol_filter(Preprocessor):
    """
    Applies the Sagvol_filter along the energy range. This smooths the spectrum by using a polynome in a window to approximate the base signal.
    Args:
        window_length (int): Length of the window used by the filter (Default: 15)
        polyorder (int): Order of the polynome used to approximate the signal (Default: 3)
    """
    window_length: int = 15
    polyorder: int = 3

    def _transform(self) -> None:
        self._data.absorption = sp.signal.savgol_filter(self._data.absorption, window_length=self.window_length, polyorder=self.polyorder, axis=1)

class Rebinner(Preprocessor):
    """
    Rebins the spectra along the energy axis in defined equienergetical steps in the preEdge and Edge region and equi-k steps in the postEdge
    Args:
        edge_range tuple[float]: Energy range used with edge_bin. If not provided value defined in global (XASPara) will be used.
        pre_edge_bin (float): Bin size of the pre edge region in eV (Default: 10)
        edge_bin (float): Bin size of the edge region in eV (Default: 0.5)
        post_edge_bin (float): Bin size of the post edge region in A-1 (Default: 0.05)
    """
    edge_range: tuple[float, float] = None
    _edge_range: tuple[float, float] = PrivateAttr(default=None)
    pre_edge_bin: float = 10
    edge_bin: float = 1
    post_edge_bin: float = 0.05

    def model_post_init(self, context):
        super().model_post_init(context)

        print("IDK ob ich hier redefinition der Edge zulassen sollte. IDK ob ich automatische erkennung aktivieren sollte")
        if self.edge_range is None:
            self.edge_range = (self.para._pre_edge_range[1], self.para._post_edge_range[0])

    def _compile_range(self):
        self._edge_range = tuple(e + self.para.edge_pos for e in self.edge_range)

    def _boxcar_average(self, e_start, e_end):
        i_start = np.searchsorted(self._data.energies, e_start)
        i_end = np.searchsorted(self._data.energies, e_end) - 1
        if i_start >= i_end:
            raise ValueError(f"no data in bin [{e_start}, {e_end}]") 
        
        t_start = (e_start - self._data.energies[i_start-1]) / (self._data.energies[i_start] - self._data.energies[i_start-1])
        t_end = (e_end - self._data.energies[i_end]) / (self._data.energies[i_end+1] - self._data.energies[i_end])
        abs_start = self._data.absorption[: ,i_start-1] * (1-t_start) + self._data.absorption[: ,i_start] * t_start
        abs_end = self._data.absorption[: ,i_end] * (1-t_end) + self._data.absorption[: ,i_end+1] * t_end


        areas = (abs_start + self._data.absorption[: ,i_start]) / 2 * (self._data.energies[i_start] - e_start)
        for i in range(i_start, i_end):
            areas += (self._data.absorption[: ,i] + self._data.absorption[: ,i+1]) / 2 * (self._data.energies[i+1] - self._data.energies[i])
        areas += (self._data.absorption[: ,i_end] + abs_end) / 2 * (e_end - self._data.energies[i_end])
        return areas / (e_end - e_start)

    def _transform(self):
        self._compile_range()

        pre_n  = int(np.ceil((self._edge_range[0] - self._data.energies[0]) / self.pre_edge_bin))
        edge_n = int(np.ceil((self._edge_range[1] - self._edge_range[0]) / self.edge_bin))
        post_e = self._edge_range[0] + edge_n * self.edge_bin
        post_k = deltaE2k(post_e - self.para.edge_pos)
              
        post_n = int(np.ceil((deltaE2k(self._data.energies[-1] - post_e) - post_k) / self.post_edge_bin))

        steps = np.concat([
            np.linspace(self._edge_range[0] - pre_n * self.pre_edge_bin, self._edge_range[0], pre_n, False),
            np.linspace(self._edge_range[0], self._edge_range[0] + edge_n * self.edge_bin, edge_n, False),
            k2deltaE(np.linspace(post_k, post_k + post_n * self.post_edge_bin, post_n, False)) + self.para.edge_pos
        ], axis=0)

        self._data.absorption = np.stack([self._boxcar_average(steps[i], steps[i+1]) for i in range(len(steps)-1)], axis = 1)
        self._data.energies = np.array([(steps[i] + steps[i+1]) / 2 for i in range(len(steps)-1)])

class Merger(Preprocessor):
    mode: Literal['all', 'auto', 'manuel'] = "all"
    threshold: float = 0.03
    _groups: np.ndarray[int] = PrivateAttr(default=None)
    _times: np.ndarray[float] = PrivateAttr(default=None)

    def _merge_all(self):
        self._data.absorption = self._data.absorption.mean(axis=0)[np.newaxis, :]
        self._data.times = np.arange(1)

    def _merge_manuel(self):
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(self._data.energies, np.arange(len(self._data.times)))
        heatmap = ax.pcolormesh(X, Y, self._data.absorption-np.mean(self._data.absorption, axis=0), cmap="bwr", shading="auto")
        fig.colorbar(heatmap)
        fig.show()
        
        ranges = []
        while True:
            inp = input(f"Type integer range of range {len(ranges)+1} (form: i_start i_stop):")
            if inp == "":
                break
            
            try:
                r = [int(s) for s in inp.split()]
            except ValueError as e:
                print(e)
                continue
            
            if len(r) != 2:
                print(ValueError(f"Could not interpret '{inp}' as two integers"))
                continue

            ranges.append(slice(*r))

        plt.close(fig)
        self._data.absorption = np.stack([np.mean(self._data.absorption[r], axis=0) for r in ranges], axis= 0)
        self._data.times = np.arange(len(ranges))

    def _merge_auto(self):
        dist_matrix = sp.spatial.distance.pdist(self._data.absorption, metric=lambda u,v: np.sqrt(np.mean(np.square(u-v))))

        if self.plot:
            plt.subplots()
            plt.title(f"Preprocessor {self.name}")
            plt.hist(dist_matrix, log=True)
            plt.show()

        Z = sp.cluster.hierarchy.linkage(dist_matrix, method='complete')
        self._groups = sp.cluster.hierarchy.fcluster(Z, t=self.threshold, criterion='distance') - 1

        g_total = max(self._groups) + 1
        group_list: list[list[int]] = [[] for x in range(g_total)]
        for i, g in enumerate(self._groups):
            group_list[g].append(i)

        self._data.absorption = np.stack([np.mean(self._data.absorption[g], axis=0) for g in group_list], axis=0)
        self._data.times = np.arange(g_total)
        self.logger.info(f"Preprocessor {self.name} has identified {g_total} groups")

    def _transform(self):
        self._times = self._data.times
        self.logger.info(f"Preprocessor {self.name} uses mode {self.mode}")
        match self.mode:
            case "all":
                return self._merge_all()
            case "manuel":
                return self._merge_manuel()
            case "auto":
                return self._merge_auto()
    
    def _plot(self):
        plt.subplots()
        plt.title(f"Preprocessor {self.name}")
        for t,g in zip(self._times, self._groups):
            plt.axvline(t, c=f"C{g}")
        # for i, g in enumerate(self._groups):
        #     plt.axvline(i, c=f"C{g}")
#endregion

#region Analyser
class Analyzer(Processor):

    def analyse(self, data: XASData) -> XASData:
        self._data = data
        self.logger.info(f"Analyzer {self.name} started")
        self._analyse()
        self.logger.info(f"Analyzer {self.name} finished without Problems")
    
    @abstractmethod
    def _analyse(self) -> None:
        pass

class SVDDecompositor(Analyzer):
    mode: Literal['threshold', 'n_comp'] = "threshold"
    threshold: float | None = 0
    n_comp: int | None = None
    def _analyse(self):
        U, S, Vh = np.linalg.svd(self._data.absorption, full_matrices=False)
        if self.mode == "threshold":
            n_keep = np.sum(S > self.threshold)
        else:
            n_keep = self.n_comp
        U, S, Vh = U[:, :n_keep], S[:n_keep], Vh[:n_keep, :]
        a_approx = U @ np.diag(S) @ Vh

        fig, ((axul, axur), (axll, axlr)) = plt.subplots(2, 2, figsize=(12,8), layout="tight", width_ratios=(1,1))
        fig.suptitle(f"Analyzer {self.name}")
        step = max(int(len(self._data.times)/20), 1)
        for i, spectra in enumerate((self._data.absorption - a_approx)[::step, :]):
            axul.plot(self._data.energies, spectra, label=f"{self._data.times[step*i]:.0f}")

        axur.bar(np.arange(len(S)), S)
        axur.set_yscale('log')

        for i, (contri, comp) in enumerate(zip((U * S).T, np.diag(S)@Vh)):
            axll.plot(self._data.times, contri)
            axlr.plot(self._data.energies, comp)
        
        axll.axhline(0, color="black")

class EdgeLC(Analyzer):
    pre: float = None
    post: float = None
    refs: list[Annotated[XASRef, BeforeValidator(str2Ref)]| int] = [0, -1]
    def _analyse(self):
        weight = 1000
        if isinstance(self.pre, type(None)):
            self.pre = self.para._pre_edge_range[1]
        if isinstance(self.post, type(None)):
            self.post = self.para._post_edge_range[0]
        e_slice = self._data.energyRange2idx(self.para.edge_pos + self.pre, self.para.edge_pos + self.post)

        for i, r in enumerate(self.refs):
            if isinstance(r, int):
                self.refs[i] = XASRef(mu = self._data.absorption[r,e_slice], name = f"Ref{i}")
            else:
                r.resample(self._data.energies[e_slice])
        
        mu = np.append(self._data.absorption[:,e_slice], np.full((len(self._data.times), 1), weight), axis=1)
        refs = np.column_stack([np.append(r.mu, [weight]) for r in self.refs])

        coeffs = np.ones((len(self._data.times), len(self.refs)))
        def fit_nnls(t_idx):
            coeffs[t_idx], _ = sp.optimize.nnls(refs, mu[t_idx])

        with ThreadPool(processes=8) as pool:
            pool.map(fit_nnls, range(len(self._data.times)))

        # paras = Parameters()
        # for r in self.refs[:-1]:
        #     paras.add(r.name.replace(" ", "_"), value = 1/len(self.refs), min = 0, max = 1)
        # paras.add(r.name, value = 1/len(self.refs), min = 0, max = 1, expr=f"1 - ({" + ".join(paras.keys())})")

        # results = [None for _ in self._data.times]
        # mu_refs = [r.mu for r in self.refs]

        # def fit_tp(t_idx):
        #     local_paras = paras.copy()
        #     def obj(p):
        #         return np.dot([p[r.name] for r in self.refs], mu_refs) - mu[t_idx]

        #     results[t_idx] = minimize(obj, paras, method="leastsq")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        fig.suptitle(f"Analyzer {self.name}")
        for i in range(len(self.refs)):
            ax1.plot(self._data.times, coeffs[:,i], label = self.refs[i].name)
        ax1.legend(loc="upper center", ncols=3)

        cum_coeffs = np.cumsum(coeffs,axis=1)
        width = np.min(np.diff(self._data.times))
        for i in reversed(range(len(self.refs))):
            ax2.bar(self._data.times, cum_coeffs[:,i], label = self.refs[i].name, width=width)
        ax2.legend(loc="upper center", ncols=3)
        ax2.set_ylim(0,1)
        ax2.set_xlim(0,self._data.times[-1])


class Plotter(Analyzer):
    diff: bool = True
    ref: int | np.ndarray = 0
    k_order: int = 2
    def _analyse(self):
        if self._data.normalized:
            fig, ((axul, axur), (axll, axlr)) = plt.subplots(2, 2, figsize=(12,8), layout="tight", width_ratios=(1,1))
        else:
            fig, (axul, axur) = plt.subplots(1, 2, figsize=(12,4), layout="tight", width_ratios=(1,1))
        fig.suptitle(f"Analyzer {self.name}")

        step = max(int(len(self._data.times)/20), 1)

        for i, spectra in enumerate(self._data.absorption[::step, :]):
            axul.plot(self._data.energies, spectra, label=f"{self._data.times[step*i]:.0f}")
        axul.legend(frameon=False, loc="lower right", ncols=2)
        axul.set_xlim(*self._data.energies[[0, -1]])

        X, Y = np.meshgrid(self._data.energies, self._data.times)
        if self.diff:
            mean = np.mean(self._data.absorption, axis=0)
            axur.pcolormesh(X, Y, self._data.absorption-mean, cmap="plasma", shading="auto")
        else:
            axur.pcolormesh(X, Y, self._data.absorption, cmap="plasma", shading="auto")

        if self._data.normalized:
            k, k_abs = self._data.genKspace(self.para.edge_pos)
            for i, spectra in enumerate(k_abs[::step, :]):
                axll.plot(k, (spectra - 1) * k**self.k_order)
            axll.axhline(0, ls="dotted", c="black")

            X, Y = np.meshgrid(k, self._data.times)
            k_scal = k**self.k_order
            if self.diff:
                mean = np.mean(k_abs * k_scal, axis=0)
                axlr.pcolormesh(X, Y, k_abs * k_scal - mean, cmap="plasma", shading="auto")
            else:
                axlr.pcolormesh(X, Y, (k_abs-1) * k_scal, cmap="plasma", shading="auto")

        axul.axhline(1)
        for e in (val for val in self.para.pre_edge_range if val is not None):
            axul.axvline(e, ls="dashed", color="grey")
        for e in (val for val in self.para.post_edge_range if val is not None):
            axul.axvline(e, ls="dashed", color="grey")

class EdgeDiffPlotter(Analyzer):
    def _analyse(self):
        X, Y = np.meshgrid(self._data.energies, self._data.times)
        diff = np.diff(self._data.absorption, 1, axis=1)
        plt.subplots()
        max_diff = np.max(np.abs(diff), axis=None)
        plt.pcolormesh(X, Y, diff[:-1], shading="auto", cmap="bwr", vmin=-max_diff, vmax=max_diff)
        plt.xlim(self.para.pre_edge_range[1], self.para.post_edge_range[0])
        plt.axvline(self.para.edge_pos, color="black", ls="dotted")

class Exporter(Analyzer):
    """
    Exports the data to a csv file.
    Args:
        export_path (pathlib.Path): Path to directory where the export data should be exported to. By default the input path will be reused.
        export_name (str): Name of the file (without .csv). By default the name of the first input file will be used.
    """
    path: pathlib.Path
    exp_name: str
    mode: Literal['combined', 'individual'] = 'individual'
    comment: Optional[str] = None
    
    # def model_post_init(self, context):
    #     super().model_post_init(context)
    #     if self.export_path is None:
    #         self.export_path = self.para.path
    #     if self.export_name is None:
    #         self.export_name = self.para.name

    def _analyse(self):
        if self.mode == 'combined':
            self._data.toNORM(self.path, self.exp_name, self.para, self.comment)
        else:
            self._data.toNORMind(self.path, self.exp_name, self.para, self.comment)
        self.logger.info(f"Exported Data in '{self.path}' with name '{self.name}'")
#endregion

PREPROCESSORS = {cls.__name__: cls for cls in Preprocessor.__subclasses__()}
ANALYZERS = {cls.__name__: cls for cls in Analyzer.__subclasses__()}

#region Pipeline
class XASPipeline:
    logger: logging.Logger = logging.getLogger("XAS-Pipeline")
    context: PipelineContext = None
    def __init__(self):
        self._PreProcessors: list[Preprocessor] = []
        self._Analyzers: list[Analyzer] = []

    def _load_global_conf(self, config: dict):
        xp = {k: v for k, v in config.items() if k in inspect.signature(XASPara.__init__).parameters.keys()}
        self.defineXASParas(XASPara(**xp))

        context = {}
        context_vars = ["path", "sample_name", "beamline", "plot"]
        for conf, val in config.items():
            if conf in inspect.signature(XASPara.__init__).parameters.keys():
                continue
            elif conf in context_vars:
                context[conf] = val
            else:
                raise ValueError(f"Global configuration parameter '{conf}' with value '{val}' not recognized")
        
        return context

    def load_config(self, config: dict, cli_context: dict):
        context = self._load_global_conf(config.get("global", {}))
        context.update(cli_context)
        self.context = PipelineContext.model_validate(context)

        for cls_name, cls_config in config.items():
            if cls_name in PREPROCESSORS:
                self._PreProcessors.append(PREPROCESSORS[cls_name].with_context(cls_config or {}, self.context))
            elif cls_name in ANALYZERS:
                self._Analyzers.append(ANALYZERS[cls_name].with_context(cls_config or {}, self.context))

    def addPreProcessor(self, p: Preprocessor):
        if not isinstance(p, Preprocessor):
            raise ValueError(f"processor has to be of type {Preprocessor} not {type(p)}")
        self._PreProcessors.append(p)

    def addAnalyzer(self, a:Analyzer):
        if not isinstance(a, Analyzer):
            raise ValueError(f"processor has to be of type {Analyzer} not {type(a)}")
        self._Analyzer.append(a)

    def defineXASParas(self, paras: XASPara):
        Processor.para = paras

    def run(self, data: XASData):
        self.logger.info("Starting Pipeline Execution...")

        for p in self._PreProcessors:
            data = p.transform(data)

        for a in self._Analyzers:
            a.analyse(data)
        plt.show()
#endregion

def runPipeline(conf_path: pathlib.Path, cli_context: dict):
    logger = logging.getLogger("XAS-Pipeline")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    XAS = XASPipeline()
    with open(conf_path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Problem during parsing of {conf_path}: {exc}")
    XAS.load_config(config, cli_context)

    XAS.run(XASData.extracter(logger, XAS.context.path, XAS.context.exp_name, XAS.context.beamline))

#region main
def main():
    argParser = argparse.ArgumentParser(description="XAS-Pipeline")

    argParser.add_argument("exp_name")
    argParser.add_argument("--path", type=pathlib.Path, help="Path of the raw data. Can also be provided in the config.")
    argParser.add_argument("--beamline", type=str, help="Beamline-mode (e.g. Balder, P65_T, P65_F, P65_SSD) to correctly read the data. Can also be provided in the config.")
    argParser.add_argument("--plot", type=bool, help="Overwrite of default Value for Preprocessor plotting (Default: False). Can also be provided in the config.")
    argParser.add_argument("-c", "--config", default="config.yaml", type=str, help="Path of the .yaml file serving as the config")

    args = argParser.parse_args()

    runPipeline(args.config, {k: v for k, v in vars(args).items() if k not in ["config"] and v is not None})
#endregion

if __name__ == "__main__":
    main()