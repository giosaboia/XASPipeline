# Hey Jannik

from numpy._typing import NDArray
import argparse, h5py, inspect, logging, pathlib, os, re, struct, sys, time, yaml

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt
import scipy as sp
import matplotlib.colors as mcolors

from abc import abstractmethod, ABC
from dataclasses import dataclass
from datetime import datetime
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import Slider
from matplotlib.ticker import AutoMinorLocator
from matplotlib import axes, figure, widgets, text, collections, lines
from multiprocessing.pool import ThreadPool
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
from lmfit import Parameters, minimize, report_fit
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import leastsq
from typing import Annotated, ClassVar, Callable, Union, Literal, Optional, Type, Any, get_origin, get_args, cast


def deltaE2k(deltaE):
    return np.sqrt(2 * sp.constants.m_e * sp.constants.eV * deltaE / np.square(sp.constants.hbar)) * 1E-10

def k2deltaE(k):
    return np.square(k* 1E10) * np.square(sp.constants.hbar) / (2 * sp.constants.m_e * sp.constants.eV)

def abs2AthenaRep(val: float) -> str:
    if val > 0.1:
        return " %.8f" %val
    else:
        return " %17.10E" %val

def readDatCols(logger, file_path, cols) -> tuple[float, npt.NDArray[np.floating[Any]]]:
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
    uC: int = 1
    with open(file_path, 'r') as f:
        for l in f:
            if l[0] == "#":
                headerline = l
            else:
                if not startReadout and isinstance(useCol, str) :
                    try:
                        uC = headerline.split()[1:].index(useCol)
                    except ValueError:
                        uC = 1
                    startReadout = True

                words = l.split()
                energies.append(float(words[0]))
                absorption.append(float(words[uC]))

    return np.array(energies), np.array(absorption)

#region dataclass
class XASPara:
    def __init__(
            self,
            edge:str,
            element: str,
            edge_pos: float,
            pre_edge_range: tuple[float | None, float],
            post_edge_range: tuple[float, float | None]
            ):
        self.edge = edge
        self.element = element
        self.edge_pos = edge_pos
        self._pre_edge_range = pre_edge_range
        self._post_edge_range = post_edge_range
        # self.beamline: str = beamline
        # self.path: pathlib.Path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
        # self.name: str

    @property
    def pre_edge_range(self) -> tuple[float | None, float]:
        e1, e2 = self._pre_edge_range
        return (self.edge_pos + e1 if e1 is not None else None, self.edge_pos + e2)

    @property
    def post_edge_range(self) -> tuple[float, float | None]:
        e1, e2 = self._post_edge_range
        return (self.edge_pos + e1, self.edge_pos + e2 if e2 is not None else None)

@dataclass
class XASData:
    energies: npt.NDArray[np.floating[Any]]
    times: npt.NDArray[np.floating[Any]]
    absorption: npt.NDArray[np.floating[Any]]
    normalized: bool = False

    def __post_init__(self):
        self.removeNan()
        self.validate()

    def __getitem__(self, key):
        assert isinstance(key, slice) | isinstance(key, list)
        return type(self)(np.copy(self.energies, subok=True), np.copy(self.times[key], subok=True), np.copy(self.absorption[key], subok=True))


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
            case _:
                raise ValueError("No beamline specified")
            
        if len(files) == 0:
            raise ValueError(f"No files with name '{name}' in '{directory}' found.")

        return extract_func(logger, files)

    @classmethod
    def extract_data_hdf5(cls: type['XASData'], logger: logging.Logger, files: list[pathlib.Path]) -> 'XASData':
        def conv_time(date_str: str):
            try:
                return datetime.strptime(date_str, r"%Y-%m-%d %H:%M:%S.%f").timestamp()
            except ValueError:
                return np.nan

        energy, times, mu = np.array([]), np.array([]), np.array([])
        for i, file in enumerate(files):
            with h5py.File(file, 'r') as run_datafile:
                e = np.array(cast(h5py.Dataset, run_datafile['energy'])[0,:-1])
                t = np.array([conv_time(str(t)[2:-1]) for t in cast(h5py.Dataset, run_datafile['time'])])
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

        energy, times, mu = np.array([]), [], np.array([])
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
        times = np.array(times)
        if log:
            return cls(times - times[0], energy, np.log(mu))
        else:
            return cls(times - times[0], energy, mu)
        
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
        
    def toNORM(self, path:pathlib.Path, name: str, para:XASPara, comment:str|None):
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

    def maskedTime(self, mask: npt.NDArray[np.bool_]):
        self.absorption = self.absorption[mask]
        self.times = self.times[mask]
        
    def energyRange2idx(self, low, up) -> slice:        
        if isinstance(low, type(None)) & isinstance(up, type(None)):
            return slice(0, None)
        elif isinstance(low, type(None)):
            return slice(0, np.argmax(self.energies > up), None)
        elif isinstance(up, type(None)):
            return slice(np.argmax(self.energies > low), None)
        
        if low > up:
            raise ValueError("energyRange2idx requires the low value to be smaller than the up value")
        
        return slice(np.searchsorted(self.energies, low), np.searchsorted(self.energies, up))
#endregion

#region XASReference
class XASRef(BaseModel):
    _mu: Optional[npt.NDArray] = None
    name: str
    color: Optional[str] = None
    source_e: Optional[npt.NDArray[np.floating[Any]]] = None
    source_mu: Optional[npt.NDArray[np.floating[Any]]] = None
    source_idx: Optional[int] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    color: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def from_conf(cls, conf: list[Any]) -> dict[Any,Any]:
        if not conf:
            raise ValueError("XASRef configuration list cannot be empty")
        ref = conf[0]
        color = conf[1] if len(conf) > 1 else None
        name = conf[2] if len(conf) > 2 else None

        if isinstance(ref, int):
            if name is None: name = f"Spectra {ref}"
            return dict(name=name, source_idx=ref, color = color)
        else:
            path = pathlib.Path(ref).resolve()
            energy, mu = readNorm(path)
            if name is None: name = path.stem
            return dict(name = name, source_e = energy, source_mu = mu, color = color)
    
    @property
    def mu(self) -> npt.NDArray:
        assert self._mu is not None, "Before extracting mu from XASRef it has to be pulled from the data using pull_data and the provided idx or resampled from the provided ref file."
        return self._mu
    
    def pull_data(self, mu: npt.NDArray):
        if self.source_idx is not None:
            self.absorption = mu[self.source_idx]

    def resample(self, target_energy: np.ndarray):
        if self._mu is not None:
            if len(self.mu) == len(target_energy):
                return
        if self.source_e is not None and self.source_mu is not None:
            self._mu = self._rebin(target_energy)
        else:
            raise ValueError("XASRef has not data and no source to resample from!")
    
    def _rebin(self, target_energy: np.ndarray):
        assert self.source_e is not None
        assert self.source_mu is not None

        midpoints = (target_energy[:-1] + target_energy[1:]) / 2
        edges = np.concatenate([
            [target_energy[0] - (target_energy[1] - target_energy[0]) / 2],
            midpoints,
            [target_energy[-1] + (target_energy[-1] - target_energy[-2]) / 2]
        ])

        cum_integral = sp.integrate.cumulative_trapezoid(self.source_mu, x = self.source_e, initial=0)
        resampled_integral = np.interp(edges, self.source_e, cum_integral)
        return np.diff(resampled_integral) / np.diff(edges)

class PipelineContext(BaseModel):
    path: pathlib.Path
    exp_name: str
    beamline: Literal['Balder', 'P65_T', 'P65_F', 'P65_SDD']
    plot: Optional[bool]
#endregion

#region Processor
possible_path = Union[pathlib.Path, int]
RESOLUTION_MAP: dict[type[Any], Any] = {
    XASRef: Union[tuple[possible_path], tuple[possible_path, *tuple[Any, ...]]],
}

class Processor(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    logger: ClassVar[logging.Logger] = logging.getLogger("XAS-Pipeline")

    name: str 
    plot: bool = False

    para: ClassVar[XASPara]
    _data: XASData | None = PrivateAttr(default=None)

    @classmethod
    def with_context(cls, data: dict, context: PipelineContext):
        for field_name, field_info in cls.model_fields.items():
            if (field_name in data) and (field_info.annotation is not None):
                data[field_name] = cls._resolve_paths(field_info.annotation, data[field_name], context.path)
            else:
                val = getattr(context, field_name, None)
                if val is None:
                    continue

                data[field_name] = val

        if data.get("name") is None:
            data["name"] = cls.__name__

        return cls.model_validate(data)
    
    @staticmethod
    def _resolve_paths(annotation: type[Any], val: Any, base_path: pathlib.Path) -> Any:
        origin = get_origin(annotation)
        if origin is Annotated:
            return Processor._resolve_paths(get_args(annotation)[0], val, base_path)
        
        if origin is Union:
            for sub_type in get_args(annotation):
                resolved = Processor._resolve_paths(sub_type, val, base_path)
                if resolved != val:
                    return resolved
            return val
            
        if origin is list:
            if not isinstance(val, list):
                return val
            else:
                return [Processor._resolve_paths(get_args(annotation)[0], item, base_path) for item in val]
        
        if origin is tuple:
            if not isinstance(val, (list, tuple)): return val
            type_args = get_args(annotation)

            if len(val) < len(type_args):
                raise ValueError(f"Expected a tuple/list of length {len(type_args)}, but got {len(val)}.")
            
            if len(type_args) == 2 and type_args[1] is Ellipsis:
                return tuple(Processor._resolve_paths(type_args[0], item, base_path) for item in val)
            a = tuple(Processor._resolve_paths(type_args[i], val[i], base_path) if i < len(type_args) else val[i] for i in range(len(val)))
            return a
        
        if annotation in RESOLUTION_MAP:
            return Processor._resolve_paths(RESOLUTION_MAP[annotation], val, base_path)
        
        if isinstance(val, str) and Processor._is_path_type(annotation):
            return Processor._resolve_relative_paths(val, base_path)

        return val

    @staticmethod
    def _is_path_type(tpl: Any) -> bool:
        path_types = (pathlib.Path, XASRef)
        try:
            if get_origin(tpl) is Annotated:
                tpl = get_args(tpl)[0]
            return isinstance(tpl, type) and issubclass(tpl, path_types)
        except TypeError:
            return False
        
    @staticmethod
    def _resolve_relative_paths(val: str, base_path: pathlib.Path) -> pathlib.Path:
        val_path = pathlib.Path(val)
        if val_path.is_absolute():
            return val_path
        else:
            return (base_path / val_path).resolve()

    @property
    def data(self) -> XASData:
        if self._data is None:
            raise RuntimeError(
                f"Processor '{self.name}' attempted to access data before transform() was called. "
                "Ensure the pipeline is orchestrating calls correctly."
            )
        return self._data

    @data.setter
    def data(self, data: XASData):
        if not isinstance(data, XASData):
            raise RuntimeError(f"Attribute data in {self.name} can only be of type {type(XASData)} not {type(data)}")
        self._data = data
#endregion

#region Preprocessors
class Preprocessor(Processor):

    def transform(self, data: XASData) -> XASData:
        self.data = data
        self.logger.info(f"Preprocessor {self.name} started")
        start_time = time.time()
        
        self._transform()
        try:
            self.data.validate()
        except (ValueError, TypeError) as e:
            error_msg = f"Validation failed after '{self.name}': {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        duration = time.time() - start_time
        self.logger.info(f"Preprocessor {self.name} finished after {round(duration, 4)} s.")
        if self.plot:
            self._plot()
        return self.data

    @abstractmethod
    def _transform(self) -> None:
        pass

    def _plot(self) -> None:
        step = max(int(len(self.data.times)/20), 1)
        fig, ax = plt.subplots()
        times_min = self.data.times / 60
        norm = plt.Normalize(times_min[0], times_min[-1])
        cmap = LinearSegmentedColormap.from_list('royal_firebrick', ['#750002', '#436436'])
        for i, spectra in enumerate(self.data.absorption[::step, :]):
            color = cmap(norm(times_min[step*i]))
            ax.plot(self.data.energies, spectra, color=color, label=f"{times_min[step*i]:.1f}")
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Time (min)')
        plt.xlim(self._data.energies[0], self._data.energies[0] + 900)
        major_ticks = np.linspace(self._data.energies[0], self._data.energies[0] + 900, 7)
        plt.xticks(major_ticks)
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
        plt.title(f"Preprocessor {self.name}")
        plt.legend(frameon=False, loc="lower right", ncols=3)
        plt.show()


class BackgroundModel(BaseModel, ABC):
    _fitted_params: Any = PrivateAttr(None)
    @abstractmethod
    def fit_transform(self, data: XASData, e_slice: slice, t_slice: slice | list = slice(None), e0_idx: npt.NDArray[np.integer] | None = None) -> npt.NDArray[np.floating[Any]]:
        pass

    @abstractmethod
    def transform(self, energies: npt.NDArray[np.floating[Any]], spec_idx: int | None = None) -> npt.NDArray[np.floating[Any]]:
        pass

    def mask(self, selection: Any):
        assert self._fitted_params is not None, "Cannot mask before fitting."
        
        if isinstance(self._fitted_params, np.ndarray):
            self._fitted_params = self._fitted_params[:,selection]
        elif isinstance(self._fitted_params, list):
            self._fitted_params = [item for i, item in enumerate(self._fitted_params) if selection[i]]

    @model_validator(mode="before")
    @classmethod
    def match_class_name(cls, data: Any) -> Any:
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise ValueError()
        if len(data) == 0:
            return data
        if any(key in cls.model_fields.keys() for key in data.keys()):
            return data
        if cls.__name__ not in data.keys():
            raise ValueError(f"{cls} is not in {data.keys()}")
        
        return data[cls.__name__] or {}

class Polynomial(BackgroundModel):
    order: int = 3
    def fit_transform(self, data: XASData, e_slice: slice, t_slice: slice | list = slice(None), e0_idx: npt.NDArray[np.integer] | None = None) -> npt.NDArray[np.floating[Any]]:
        A = np.vander(data.energies, self.order + 1)
        self._fitted_params, _, _, _ = np.linalg.lstsq(A[e_slice], data.absorption[t_slice,e_slice].T)
        return np.dot(A, self._fitted_params).T
    
    def transform(self, energies: npt.NDArray[np.floating[Any]], spec_idx: int | None = None) -> npt.NDArray[np.floating[Any]]:
        assert self._fitted_params is not None, f"cant access Polynomial data without fit"
        if spec_idx is None:
            return np.dot(np.vander(energies, self.order + 1), self._fitted_params).T
        else:
            return np.dot(np.vander(energies, self.order + 1), self._fitted_params[:,spec_idx]).T

        
class Victoreen(BackgroundModel):
    order: int = 3
    def fit_transform(self, data: XASData, e_slice: slice, t_slice: slice | list = slice(None) , e0_idx: npt.NDArray[np.integer] | None = None) -> npt.NDArray[np.floating[Any]]:
        A = np.column_stack([
            (1E10 * sp.constants.h * sp.constants.c / (sp.constants.e * data.energies)) ** self.order, np.ones(len(data.energies))
        ])

        self._fitted_params, _, _, _ = np.linalg.lstsq(A[e_slice], data.absorption[t_slice,e_slice].T)
        return np.dot(A, self._fitted_params).T
    
    def transform(self, energies: npt.NDArray[np.floating[Any]], spec_idx: int | None = None) -> npt.NDArray[np.floating[Any]]:
        A = np.column_stack([
            (1E10 * sp.constants.h * sp.constants.c / (sp.constants.e * energies)) ** self.order, np.ones(len(energies))
        ])

        assert self._fitted_params is not None; f"cant access Polynomial data without fit"
        if spec_idx is None:
            return np.dot(A, self._fitted_params).T
        else:
            return np.dot(A, self._fitted_params[:,spec_idx]).T

class KSpline(BackgroundModel):
    order: int = 3
    weigth: int = 3
    _e0_idx: Optional[npt.NDArray[np.integer]] = PrivateAttr(None)
    _e_slice: Optional[slice] = PrivateAttr(None)
    def fit_transform(self, data: XASData, e_slice: slice, t_slice: slice | list = slice(None), e0_idx: npt.NDArray[np.integer] | None = None) -> npt.NDArray[np.floating[Any]]:
        self._e0_idx = e0_idx
        assert self._e0_idx is not None
        self._e_slice = e_slice
        eV2revA = (2E-20 * sp.constants.e * sp.constants.m_e / sp.constants.hbar**2)

        self._fitted_params = []
        fitted = np.zeros(data.absorption.shape)
        iterator = range(len(data.times[t_slice])) if isinstance(t_slice, slice) else t_slice
        for i in iterator:
            ks = np.sqrt((data.energies[e_slice] - data.energies[self._e0_idx[i]]) * eV2revA)
            knots = np.pad(np.linspace(ks[0], ks[-1], int((ks.max() - ks.min())/2)), (self.order,self.order), "edge")
            self._fitted_params.append(sp.interpolate.make_lsq_spline(
                ks, (data.absorption[i,e_slice] * (ks**self.weigth)),
                knots, k = self.order
            ))
            fitted[i,e_slice] = self._fitted_params[i](ks) * (ks**(-self.weigth))
            fitted[i,:e_slice.start] = fitted[i,e_slice.start]
        
        return fitted
    
    def transform(self, energies: npt.NDArray[np.floating[Any]], spec_idx: int | None = None) -> npt.NDArray[np.floating[Any]]:
        assert self._fitted_params is not None, "Cannot transform before fitting."
        assert self._e0_idx is not None
        assert self._e_slice is not None
        eV2revA = (2E-20 * sp.constants.e * sp.constants.m_e / sp.constants.hbar**2)

        fitted = np.zeros((len(self._fitted_params),len(energies))) if spec_idx is None else np.zeros((1,len(energies)))
        iterator = range(len(self._fitted_params)) if spec_idx is None else [spec_idx]
        for j, i in enumerate(iterator):
            ks = np.sqrt((energies[self._e_slice] - energies[self._e0_idx[i]]) * eV2revA)
            fitted[j,self._e_slice] = self._fitted_params[i](ks) * (ks**(-self.weigth))
            fitted[j,:self._e_slice.start] = fitted[j,self._e_slice.start]
        
        if spec_idx is None:
            return fitted
        else:
            return fitted[0]
        
preEdgeFitModels = Union[Polynomial, Victoreen]
postEdgeFitModels = Union[Polynomial, KSpline]
PREMODELS: dict[str, Type[preEdgeFitModels]] = {cls.__name__: cls for cls in get_args(preEdgeFitModels)}
POSTMODELS: dict[str, Type[postEdgeFitModels]] = {cls.__name__: cls for cls in get_args(postEdgeFitModels)}

class Normalizer(Preprocessor):
    """
    Normlizes the spectra.
    Args:
        pre: ClassName and Args of the Fitting model for the PreEdge (Polynomial or Victoreen)
    """
    pre: preEdgeFitModels = Polynomial(order = 1)
    post: postEdgeFitModels = Polynomial(order = 3)

    @model_validator(mode="before")
    @classmethod
    def validate_structure(cls, data: Any) -> Any:
        for field in ["pre", "post"]:
            val = data.get(field)
            if val is not None:
                if not isinstance(val, dict):
                    raise ValueError(f"'{field}' must be a dictionary.")
                if len(val) != 1:
                    raise ValueError(f"'{field}' must contain exactly one model key.")
        return data
    
    def _transform(self) -> None:
        pre_edge_slice = self.data.energyRange2idx(*self.para.pre_edge_range)
        post_edge_slice = self.data.energyRange2idx(*self.para.post_edge_range)

        if self.para.pre_edge_range[0] is not None and pre_edge_slice.start != 0:
            start_idx = np.searchsorted(self.data.energies, self.para.pre_edge_range[0]) 
            self.data.energies = self.data.energies[start_idx:]
            self.data.absorption = self.data.absorption[:,start_idx:]
            self.data.validate()
            pre_edge_slice = slice(pre_edge_slice.start-start_idx, pre_edge_slice.stop-start_idx)
            post_edge_slice = slice(post_edge_slice.start-start_idx, post_edge_slice.stop-start_idx)
        
        if self.para.post_edge_range[1] is not None and post_edge_slice.stop != len(self.data.energies):
            stop_idx = np.searchsorted(self.data.energies, self.para.post_edge_range[1], side="right")
            self.data.energies = self.data.energies[:stop_idx]
            self.data.absorption = self.data.absorption[:,:stop_idx]
            self.data.validate()
            

        e0_idx = self._e0_idx(pre_edge_slice.stop, post_edge_slice.start)
        pre = self.pre.fit_transform(self.data, pre_edge_slice, e0_idx = e0_idx)
        self.data.absorption -= pre

        post = self.post.fit_transform(self.data, post_edge_slice, e0_idx = e0_idx)
        self.data.absorption /= post

        rows_with_neg = (post < 0).any(axis=1)
        mask = np.invert(rows_with_neg)
        self.logger.info(f"Preprocessor {self.name} removed {np.sum(rows_with_neg)} from {len(self.data.times)} due to negative values in post_line spline")
        self.data.maskedTime(mask)
        # self.data.absorption = self.data.absorption[mask]
        # self.data.times = self.data.times[mask]
        self.pre.mask(mask)
        self.post.mask(mask)

        self.data.normalized = True 
        
    def _e0_idx(self, pre_idx: int, post_idx: int) -> npt.NDArray[np.integer]:
        energy_slice = slice(pre_idx, post_idx)
        deriv = np.gradient(self.data.absorption[:, energy_slice], axis = 1) / np.gradient(self.data.energies[energy_slice])
        return np.argmax(sp.ndimage.gaussian_filter1d(deriv, 3, axis=1), axis=1) + pre_idx

    def _plot(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
        plt.subplots_adjust(bottom=0.25)

        idx0 = 0
        pre_edge = self.pre.transform(self.data.energies, idx0)
        post_edge = self.post.transform(self.data.energies, idx0)

        data_line, = ax1.plot(self.data.energies, self.data.absorption[idx0] * post_edge + pre_edge, label=f'mu @ {self.data.times[idx0]}')
        pre_line, = ax1.plot(self.data.energies, pre_edge, ls="dashed", c="grey")
        post_line, = ax1.plot(self.data.energies, post_edge + pre_edge, ls="dashed", c="grey")
        ax1.set_xlim(self.data.energies[0], self.data.energies[-1])

        norm_line, = ax2.plot(self.data.energies, self.data.absorption[idx0])
        ax2.axhline(1, ls="dotted", color="grey")
        ax2.set_xlim(self.data.energies[0], self.data.energies[-1])
        ax2.set_ylim(0, 1.5)


        for e in (val for val in self.para.pre_edge_range if val is not None):
            plt.axvline(e, ls="dashed", color="grey")
        for e in (val for val in self.para.post_edge_range if val is not None):
            plt.axvline(e, ls="dashed", color="grey")

        ax_slider = plt.axes((0.2, 0.1, 0.65, 0.03))
        slider = widgets.Slider(ax_slider, 'Timepoint', 0, self.data.times.shape[0]-1, 
                        valinit=idx0, valfmt='%d')

        def update(val):
            idx = int(slider.val)
            pre_edge = self.pre.transform(self.data.energies, idx)
            post_edge = self.post.transform(self.data.energies, idx)

            data_line.set_ydata(self.data.absorption[idx] * post_edge + pre_edge)
            data_line.set_label(f'mu @ {self.data.times[idx]}')
            pre_line.set_ydata(pre_edge)
            post_line.set_ydata(post_edge + pre_edge)
            norm_line.set_ydata(self.data.absorption[idx])

            fig.canvas.draw_idle()
        
        slider.on_changed(update)

        plt.legend()
        plt.show()

class NoiseFilter(Preprocessor):
    """
    Filters out Spectra with large noise-to-signal ratio. After normalization the spectra with low signal exhibit large noise.
    Args:
        gate (float): multipier of the median RMS that serves as the cutoff (Default: 3)
        cutoff(float): this multiplied with the median is the largest value displayed in the histogram (Default: 20)
        bin_size(float): size of the bins (relative to the median) in the histogram plot (Default: 0.25)
        log(bool): decide weather the histogram should have a logarithmic y axis or not (Default: False)
    """
    gate: float = 3
    cutoff: float = 20
    bin_size: float = 0.25
    log: bool = False

    def _transform(self) -> None:
        rms_noise = np.sqrt(np.mean(np.square(np.diff(self.data.absorption, n=2, axis=1)), axis=1))

        threshold = self.gate * np.median(rms_noise)
        mask = rms_noise < threshold
        if self.plot:
            scaled_noise = np.clip(rms_noise / np.median(rms_noise), 0, self.cutoff)
            bins = np.arange(0, self.cutoff + self.bin_size + 0.001, self.bin_size, dtype=np.float64)
            print(bins)

            plt.subplots()
            plt.title(f"Preprocessor {self.name}")
            plt.hist(scaled_noise, bins=bins.tolist(), log=self.log)
            plt.axvline(self.gate, color="black", ls="dotted")
            plt.axvline(self.cutoff, color="black")
            plt.show()

        if np.sum(~mask):
            init_num = len(self.data.times)
            self.data.maskedTime(mask)
            # self.data.absorption = self.data.absorption[mask, :]
            # self.data.times = self.data.times[mask]
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
        self.data.absorption = sp.signal.savgol_filter(self.data.absorption, window_length=self.window_length, polyorder=self.polyorder, axis=1)

class Rebinner(Preprocessor):
    """
    Rebins the spectra along the energy axis in defined equienergetical steps in the preEdge and Edge region and equi-k steps in the postEdge
    Args:
        edge_range tuple[float]: Energy range used with edge_bin. If not provided value defined in global (XASPara) will be used.
        pre_edge_bin (float): Bin size of the pre edge region in eV (Default: 10)
        edge_bin (float): Bin size of the edge region in eV (Default: 0.5)
        post_edge_bin (float): Bin size of the post edge region in A-1 (Default: 0.05)
    """
    edge_range: Optional[tuple[float, float]] = None
    # _edge_range: tuple[float, float] = PrivateAttr(default=None)
    pre_edge_bin: float = 10
    edge_bin: float = 1
    post_edge_bin: float = 0.05

    def model_post_init(self, context):
        super().model_post_init(context)

        print("IDK ob ich hier redefinition der Edge zulassen sollte. IDK ob ich automatische erkennung aktivieren sollte")
        if self.edge_range is None:
            self.edge_range = (self.para._pre_edge_range[1], self.para._post_edge_range[0])
    
    @property
    def _edge_range(self) -> tuple[float, float]:
        assert self.edge_range is not None
        return (self.edge_range[0] + self.para.edge_pos, self.edge_range[1] + self.para.edge_pos)

    def _boxcar_average(self, e_start, e_end):
        i_start = np.searchsorted(self.data.energies, e_start)
        i_end = np.searchsorted(self.data.energies, e_end) - 1
        if i_start >= i_end:
            raise ValueError(f"no data in bin [{e_start}, {e_end}]") 
        
        t_start = (e_start - self.data.energies[i_start-1]) / (self.data.energies[i_start] - self.data.energies[i_start-1])
        t_end = (e_end - self.data.energies[i_end]) / (self.data.energies[i_end+1] - self.data.energies[i_end])
        abs_start = self.data.absorption[: ,i_start-1] * (1-t_start) + self.data.absorption[: ,i_start] * t_start
        abs_end = self.data.absorption[: ,i_end] * (1-t_end) + self.data.absorption[: ,i_end+1] * t_end


        areas = (abs_start + self.data.absorption[: ,i_start]) / 2 * (self.data.energies[i_start] - e_start)
        for i in range(i_start, i_end):
            areas += (self.data.absorption[: ,i] + self.data.absorption[: ,i+1]) / 2 * (self.data.energies[i+1] - self.data.energies[i])
        areas += (self.data.absorption[: ,i_end] + abs_end) / 2 * (e_end - self.data.energies[i_end])
        return areas / (e_end - e_start)

    def _transform(self):
        pre_n  = int(np.ceil((self._edge_range[0] - self.data.energies[0]) / self.pre_edge_bin))
        edge_n = int(np.ceil((self._edge_range[1] - self._edge_range[0]) / self.edge_bin))
        post_e = self._edge_range[0] + edge_n * self.edge_bin
        post_k = deltaE2k(post_e - self.para.edge_pos)
              
        post_n = int(np.ceil((deltaE2k(self.data.energies[-1] - post_e) - post_k) / self.post_edge_bin))

        steps = np.concat([
            np.linspace(self._edge_range[0] - pre_n * self.pre_edge_bin, self._edge_range[0], pre_n, False),
            np.linspace(self._edge_range[0], self._edge_range[0] + edge_n * self.edge_bin, edge_n, False),
            k2deltaE(np.linspace(post_k, post_k + post_n * self.post_edge_bin, post_n, False)) + self.para.edge_pos
        ], axis=0)

        self.data.absorption = np.stack([self._boxcar_average(steps[i], steps[i+1]) for i in range(len(steps)-1)], axis = 1)
        self.data.energies = np.array([(steps[i] + steps[i+1]) / 2 for i in range(len(steps)-1)])

class Merger(Preprocessor):
    mode: Literal['all', 'auto', 'manuel'] = "all"
    threshold: float = 0.03
    _groups: npt.NDArray[np.integer[Any]] | None = PrivateAttr(default=None)
    _times: npt.NDArray[np.floating[Any]]| None = PrivateAttr(default=None)

    def _merge_all(self):
        self.data.absorption = self.data.absorption.mean(axis=0)[np.newaxis, :]
        self.data.times = np.arange(1).astype(np.float64)

    def _merge_manuel(self):
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(self.data.energies, np.arange(len(self.data.times)))
        heatmap = ax.pcolormesh(X, Y, self.data.absorption-np.mean(self.data.absorption, axis=0), cmap="bwr", shading="auto")
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
        self.data.absorption = np.stack([np.mean(self.data.absorption[r], axis=0) for r in ranges], axis= 0)
        self.data.times = np.arange(len(ranges)).astype(np.float64)

    def _merge_auto(self):
        dist_matrix = sp.spatial.distance.pdist(self.data.absorption, metric=lambda u,v: np.sqrt(np.mean(np.square(u-v))))

        if self.plot:
            plt.subplots()
            plt.title(f"Preprocessor {self.name}")
            plt.hist(dist_matrix, log=True)
            plt.show()

        Z = linkage(dist_matrix, method='complete')
        self._groups = fcluster(Z, t=self.threshold, criterion='distance') - 1
        assert self._groups is not None

        g_total = max(self._groups) + 1
        group_list: list[list[int]] = [[] for x in range(g_total)]
        for i, g in enumerate(self._groups):
            group_list[g].append(i)

        self.data.absorption = np.stack([np.mean(self.data.absorption[g], axis=0) for g in group_list], axis=0)
        self.data.times = np.arange(g_total)
        self.logger.info(f"Preprocessor {self.name} has identified {g_total} groups")

    def _transform(self):
        self._times = self.data.times
        self.logger.info(f"Preprocessor {self.name} uses mode {self.mode}")
        match self.mode:
            case "all":
                return self._merge_all()
            case "manuel":
                return self._merge_manuel()
            case "auto":
                return self._merge_auto()
    
    def _plot(self):
        assert self._times is not None
        assert self._groups is not None

        if self.mode != "auto":
            return
        plt.subplots()
        plt.title(f"Preprocessor {self.name}")
        cmap = LinearSegmentedColormap.from_list('royal_firebrick', ['royalblue', 'firebrick'])
        norm = plt.Normalize(min(self._groups), max(self._groups))
        for t,g in zip(self._times, self._groups):
            plt.axvline(t, color=cmap(norm(g)))
        
#endregion

#region Analyser
class Analyzer(Processor):

    def analyse(self, data: XASData):
        self.data = data
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
        U, S, Vh = np.linalg.svd(self.data.absorption, full_matrices=False)
        if self.mode == "threshold":
            n_keep = np.sum(S > self.threshold)
        else:
            n_keep = self.n_comp
        U, S, Vh = U[:, :n_keep], S[:n_keep], Vh[:n_keep, :]
        a_approx = U @ np.diag(S) @ Vh

        fig, ((axul, axur), (axll, axlr)) = plt.subplots(2, 2, figsize=(12,8), layout="tight", width_ratios=(1,1))
        fig.suptitle(f"Analyzer {self.name}")
        step = max(int(len(self.data.times)/20), 1)
        for i, spectra in enumerate((self.data.absorption - a_approx)[::step, :]):
            axul.plot(self.data.energies, spectra, label=f"{self.data.times[step*i]:.0f}")

        axur.bar(np.arange(len(S)), S)
        axur.set_yscale('log')

        for i, (contri, comp) in enumerate(zip((U * S).T, np.diag(S)@Vh)):
            axll.plot(self.data.times, contri)
            axlr.plot(self.data.energies, comp)
        
        axll.axhline(0, color="black")

class EdgeLC(Analyzer):
    pre: Optional[float] = None
    post: Optional[float] = None
    refs: list[XASRef]
    def _analyse(self):
        weight = 1000
        if isinstance(self.pre, type(None)):
            self.pre = self.para._pre_edge_range[1]
        if isinstance(self.post, type(None)):
            self.post = self.para._post_edge_range[0]
        e_slice = self.data.energyRange2idx(self.para.edge_pos + self.pre, self.para.edge_pos + self.post)

        ref_objects = []
        for i, r in enumerate(self.refs):
            r.pull_data(self.data.absorption[:, e_slice])
            r.resample(self.data.energies[e_slice])
        
        mu = np.append(self.data.absorption[:,e_slice], np.full((len(self.data.times), 1), weight), axis=1)
        r = [r.mu for r in self.refs]
        refs = np.column_stack([np.append(r.mu, np.array([weight])) for r in self.refs])

        coeffs = np.ones((len(self.data.times), len(self.refs)))
        def fit_nnls(t_idx):
            coeffs[t_idx], _ = sp.optimize.nnls(refs, mu[t_idx])

        with ThreadPool(processes=8) as pool:
            pool.map(fit_nnls, range(len(self.data.times)))


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for i in range(len(self.refs)):
            ax1.plot(self._data.times/60, coeffs[:, i], label=self.refs[i].name.split('_')[0], color=self.refs[i].color)
        ax1.set_xlim(0, self._data.times[-1]/60)
        ax1.set_ylim(-0.01, 1.01)
        ax1.legend(loc = "upper center", bbox_to_anchor=(0.5, 1.075), frameon = False, ncols = 3)
        ax1.set_xlabel(xlabel = "Time on Stream (min)", labelpad = 5, fontsize = 16)
        ax1.set_ylabel(ylabel = "Phase Contribution (%)", labelpad = 5, fontsize = 16)
        ax1.xaxis.set_major_locator(MultipleLocator(30))
        ax1.tick_params(length = 8, width = 1, labelsize = 14)

        cum_coeffs = np.cumsum(coeffs, axis=1)
        hatch_patterns = ['..', '\\', '++', '//']
        width = np.min(np.diff(self._data.times))
        for i in reversed(range(len(self.refs))):
            color = self.refs[i].color
            hatch = hatch_patterns[i % len(hatch_patterns)]
            ax2.fill_between(self._data.times/60,
                             cum_coeffs[:, i],
                             label=self.refs[i].name.split('_')[0],
                             color=color,
                             hatch=hatch,
                             alpha=0.9,
                             edgecolor="black",
                             linewidth=0.1,
                             )
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, self._data.times[-1]/60)
        ax2.legend(loc = "upper center", bbox_to_anchor=(0.5, 1.075), frameon = False, ncols = 3)
        ax2.set_xlabel(xlabel = "Time on Stream (min)", labelpad = 5, fontsize = 16)
        ax2.set_ylabel(ylabel = "Phase Contribution", labelpad = 5, fontsize = 16)
        ax2.xaxis.set_major_locator(MultipleLocator(30))
        ax2.tick_params(length = 8, width = 1, labelsize = 14)


class Plotter(Analyzer):
    diff: bool = True
    ref: int | np.ndarray = 0
    k_order: int = 2
    def _analyse(self):
        if self.data.normalized:
            fig, ((axul, axur), (axll, axlr)) = plt.subplots(2, 2, figsize=(12,8), layout="tight", width_ratios=(1,1))
        else:
            fig, (axul, axur) = plt.subplots(1, 2, figsize=(12,4), layout="tight", width_ratios=(1,1))
            axll = axlr = None
        step = max(int(len(self.data.times)/20), 1)
        norm = plt.Normalize(self._data.times[0], self._data.times[-1])
        cmap = LinearSegmentedColormap.from_list('royal_firebrick', ['royalblue', 'firebrick'])

        for i, spectra in enumerate(self.data.absorption[::step, :]):
            color = cmap(norm(self._data.times[step*i]))
            axul.plot(self.data.energies, spectra, color= color, label=f"{self.data.times[step*i]:.0f}")
        axul.legend(frameon=False, loc="lower right", ncols=3)

        axul.set_xlim(self._data.energies[0], self._data.energies[0] + 900)
        major_ticks = np.linspace(self._data.energies[0], self._data.energies[0]+900, 7)
        axul.set_xticks(major_ticks)
        axul.xaxis.set_minor_locator(AutoMinorLocator(5))
        axul.set_xlabel("Energy (eV)", fontsize = 16, labelpad = 10)
        axul.set_ylabel("Absorption", fontsize = 16, labelpad = 10)
        axul.tick_params(axis = "both", which = "major", labelsize = 14, length = 6)
        axul.tick_params(axis = "both", which = "minor", labelsize = 14, length = 4)

        X, Y = np.meshgrid(self._data.energies, self._data.times/60)
        if self.diff:
            mean = np.mean(self._data.absorption, axis=0)
            axur.pcolormesh(X, Y, self._data.absorption-mean, cmap="copper", shading="gouraud")
        else:
            axur.pcolormesh(X, Y, self._data.absorption, cmap="copper", shading="gouraud")
        axur.set_xlim(self._data.energies[0], self._data.energies[0] + 900)
        major_ticks = np.linspace(self._data.energies[0], self._data.energies[0]+900, 7)
        axur.set_xticks(major_ticks)
        axur.xaxis.set_minor_locator(AutoMinorLocator(5))
        axur.set_xlabel("Energy (eV)", fontsize = 16, labelpad = 10)
        axur.set_ylabel("Time on Stream (min)", fontsize = 16, labelpad = 10)
        axul.tick_params(axis = "both", which = "major", labelsize = 14, length = 6)
        axul.tick_params(axis = "both", which = "minor", labelsize = 14, length = 4)

        if self.data.normalized:
            assert (axll is not None) and (axlr is not None)
            k, k_abs = self.data.genKspace(self.para.edge_pos)
            for i, spectra in enumerate(k_abs[::step, :]):
                color = cmap(norm(self._data.times[step*i]))
                axll.plot(k, (spectra - 1) * k**self.k_order, color=color)
            axll.axhline(0, ls="dotted", c="black")
            axll.set_xlim(0, 12)
            axll.set_xlabel("k (Å⁻¹)", fontsize = 16, labelpad = 10)
            axll.set_ylabel("k²χ(k)", fontsize = 16, labelpad = 10)
            axll.tick_params(axis = "both", labelsize = 14)

            X, Y = np.meshgrid(k, self._data.times/60)

            k_scal = k**self.k_order
            if self.diff:
                mean = np.mean(k_abs * k_scal, axis=0)
                axlr.pcolormesh(X, Y, k_abs * k_scal - mean, cmap="copper", shading="gouraud")
            else:
                axlr.pcolormesh(X, Y, (k_abs-1) * k_scal, cmap="copper", shading="gouraud")
            axlr.set_xlabel("k (Å⁻¹)", fontsize = 16, labelpad = 10)
            axlr.set_ylabel("Time on Stream (min)", fontsize = 16, labelpad = 10)
            axlr.tick_params(axis = "both", labelsize = 14)
            axlr.set_xlim(0, 12)


        axul.axhline(1)
        for e in (val for val in self.para.pre_edge_range if val is not None):
            axul.axvline(e, ls="dashed", color="grey")
        for e in (val for val in self.para.post_edge_range if val is not None):
            axul.axvline(e, ls="dashed", color="grey")

class EdgeDiffPlotter(Analyzer):
    def _analyse(self):
        X, Y = np.meshgrid(self.data.energies, self.data.times)
        diff = np.diff(self.data.absorption, 1, axis=1)
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
    subfolder: bool = False
    exp_name: str
    mode: Literal['combined', 'individual'] = 'individual'
    comment: Optional[str] = None

    def _analyse(self):
        if self.subfolder:
            self.path = self.path / self.exp_name
        if not self.path.exists():
            os.makedirs(self.path)

        if self.mode == 'combined':
            self.data.toNORM(self.path, self.exp_name, self.para, self.comment)
        else:
            self.data.toNORMind(self.path, self.exp_name, self.para, self.comment)
        self.logger.info(f"Exported Data in '{self.path}' with name '{self.exp_name}'")
#endregion

PREPROCESSORS = {cls.__name__: cls for cls in Preprocessor.__subclasses__()}
ANALYZERS = {cls.__name__: cls for cls in Analyzer.__subclasses__()}

#region Pipeline
class XASPipeline:
    logger: logging.Logger = logging.getLogger("XAS-Pipeline")
    context: PipelineContext | None = None

    def __init__(self):
        self._PreProcessors: list[Preprocessor] = []
        self._Analyzers: list[Analyzer] = []

    def _load_global_conf(self, config: dict) -> dict[Any, Any]:
        xp = {k: v for k, v in config.items() if k in inspect.signature(XASPara.__init__).parameters.keys()}
        self.defineXASParas(XASPara(**xp))

        context = {}
        for conf, val in config.items():
            if conf in inspect.signature(XASPara.__init__).parameters.keys():
                continue
            elif conf in PipelineContext.model_fields.keys():
                context[conf] = val
            else:
                raise ValueError(f"Global configuration parameter '{conf}' with value '{val}' not recognized")
        
        return context

    def load_config(self, config: dict, cli_context: dict):
        context = self._load_global_conf(config.get("global", {}))
        context.update(cli_context)
        self.context = PipelineContext.model_validate(context)

        for cls_name, cls_config in config.items():
            conf = cls_config or {}
            if cls_name in PREPROCESSORS:
                self._PreProcessors.append(PREPROCESSORS[cls_name].with_context(conf, self.context))
            elif cls_name in ANALYZERS:
                self._Analyzers.append(ANALYZERS[cls_name].with_context(conf, self.context))

    def addPreProcessor(self, p: Preprocessor):
        if not isinstance(p, Preprocessor):
            raise ValueError(f"processor has to be of type {Preprocessor} not {type(p)}")
        self._PreProcessors.append(p)

    def addAnalyzer(self, a:Analyzer):
        if not isinstance(a, Analyzer):
            raise ValueError(f"processor has to be of type {Analyzer} not {type(a)}")
        self._Analyzers.append(a)

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
    if not conf_path.is_absolute():
        conf_path = pathlib.Path(__file__).parent.resolve() / conf_path
    with open(conf_path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Problem during parsing of {conf_path}: {exc}")
    XAS.load_config(config, cli_context)

    assert XAS.context is not None
    XAS.run(XASData.extracter(logger, XAS.context.path, XAS.context.exp_name, XAS.context.beamline))

def constrainedTextBox(ax: axes.Axes, label: str, initial_value: str ="0.0"):
    pattern = re.compile(r'^-?\d*\.?\d*$|^-?\.\d*$')
    allowed_pattern = re.compile(r'^[-.\d]*$')
    last_valid_value = initial_value

    textbox = widgets.TextBox(ax, label, initial=initial_value)
    def on_text_change(val):
        nonlocal last_valid_value
        
        if not allowed_pattern.match(val) and val != "":
            textbox.set_val(last_valid_value)
            return

        if pattern.match(val):
            last_valid_value = val
        elif val != "" and val not in ["-", ".", ""]:
            textbox.set_val(last_valid_value)
    textbox.on_text_change(on_text_change)
    
    l = ax.get_children()[0]
    assert isinstance(l, text.Text)
    l.set_position((0.5,1))
    l.set_verticalalignment("bottom")
    l.set_horizontalalignment("center")
    
    return textbox

widgetTypes = Union[widgets.RadioButtons]
class widget_store(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pre_start: widgets.TextBox
    pre_stop: widgets.TextBox
    post_start: widgets.TextBox
    post_stop: widgets.TextBox
    Tslider: widgets.Slider
    preType: widgets.RadioButtons
    postType: widgets.RadioButtons

    pre_conf: dict[str, widgetTypes] = {}
    post_conf: dict[str, widgetTypes] = {}

    def position_labels(self):
        for w, opt in zip([self.preType, self.postType], [preEdgeFitModels, postEdgeFitModels]):
            n_options = len(get_args(opt))
            for i,p in enumerate(w.ax.get_children()[:n_options]):
                assert isinstance(p, text.Text)
                p.set_position(((i+1)/(n_options+1),.5))
                p.set_verticalalignment("center")
                p.set_horizontalalignment("left")

            c = w.ax.get_children()[n_options]
            assert isinstance(c, collections.PathCollection), f"type is {type(c)} not collections.PathCollection"
            c.set_offsets([[(i+1)/(n_options+1) -0.05, 0.5] for i in range(n_options)])

    def position_conf_labels(self):
        for a in [self.pre_conf, self.post_conf]:
            for w in a.values():
                n_options = 3
                ### using n_options with set offset is vulnarable to injections like legends; scanning for all leading text.Text might be better
                for i,p in enumerate(w.ax.get_children()[:n_options]):
                    assert isinstance(p, text.Text), f"element is {type(p)} not text.Text"
                    p.set_position(((i+1)/(n_options+1),.5))
                    p.set_verticalalignment("center")
                    p.set_horizontalalignment("left")

                c = w.ax.get_children()[n_options]
                assert isinstance(c, collections.PathCollection), f"type is {type(c)} not collections.PathCollection"
                c.set_offsets([[(i+1)/(n_options+1) -0.05, 0.5] for i in range(n_options)])
        
    def add_CallBacks(self, updateT: Callable[[Any],None], updateBorder: Callable[[Any], None], updateFit: Callable[[Any], None]):
        self.Tslider.on_changed(updateT)
        for w in [self.pre_start, self.pre_stop, self.post_start, self.post_stop]:
            w.on_submit(updateBorder)
        for w in [self.preType, self.postType, *self.pre_conf.values(), *self.post_conf.values()]:
            w.on_clicked(updateFit)

class GUINorm(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: XASData
    para: XASPara
    pre: preEdgeFitModels = Polynomial(order = 1)
    post: postEdgeFitModels = Polynomial(order = 3)
    _widgets: Optional[widget_store] = PrivateAttr(None)

    def interactiveNormalizer(self):

        self._fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
        plt.subplots_adjust(top=0.75, bottom=0.05, right=0.965, left=0.035, wspace=0.075)

        idx0 = 0
        pre0 = list(self.para._pre_edge_range)
        if pre0[0] is None:
            pre0[0] = self.data.energies[0]-self.para.edge_pos
        post0 = list(self.para._post_edge_range)
        if post0[1] is None:
            post0[1] = self.data.energies[-1]-self.para.edge_pos

        self._pre_edge_slice = self.data.energyRange2idx(*self.para.pre_edge_range)
        self._post_edge_slice = self.data.energyRange2idx(*self.para.post_edge_range)

        self._widgets = widget_store.model_validate({
            "pre_start": constrainedTextBox(plt.axes((.05, .9, .05, .05)), "Pre Start: ", "%.2f" %pre0[0]),
            "pre_stop": constrainedTextBox(plt.axes((.15, .9, .05, .05)), "Pre Stop: ", "%.2f" %pre0[1]),
            "Tslider": widgets.Slider(plt.axes((.3, .9, .4, .05)),'Timepoint', 0, self.data.times.shape[0]-1, valinit=idx0, valfmt='%d'),
            "post_start": constrainedTextBox(plt.axes((.8, .9, .05, .05)), "Post Start: ", "%.2f" %post0[0]),
            "post_stop": constrainedTextBox(plt.axes((.9, .9, .05, .05)), "Post Stop: ", "%.2f" %post0[1]),
            "preType": widgets.RadioButtons(plt.axes((.05, .85, .4, .05)), [m.__name__ for m in get_args(preEdgeFitModels)]),
            "postType": widgets.RadioButtons(plt.axes((.55, .85, .4, .05)), [m.__name__ for m in get_args(postEdgeFitModels)]),
        })
        self.update_model_conf("pre")
        self.update_model_conf("post")
        self._widgets.position_labels()

        mu0, norm, pre0, post0 = self.get_line_data(self._pre_edge_slice, self._post_edge_slice)

        self._data_line, = ax1.plot(self.data.energies, mu0, label=f'mu @ {self.data.times[idx0]}')
        self._pre_line, = ax1.plot(self.data.energies, pre0, ls="dashed", c="grey")
        self._post_line, = ax1.plot(self.data.energies, post0 + pre0, ls="dashed", c="grey")
        ax1.set_xlim(self.data.energies[0], self.data.energies[-1])

        self._norm_line, = ax2.plot(self.data.energies, norm)
        ax2.axhline(1, ls="dotted", color="grey")
        ax2.set_xlim(self.data.energies[0], self.data.energies[-1])
        ax2.set_ylim(0, 1.5)
        
        self._widgets.add_CallBacks(self.update_T, self.update_borders, self.update_fit)

        plt.show()
    
    def update_T(self, val):
        assert self._widgets is not None
        idx = int(self._widgets.Tslider.val)

        mu, norm, pre, post = self.get_line_data(self._pre_edge_slice, self._post_edge_slice)

        self._data_line.set_ydata(mu)
        self._data_line.set_label(f'mu @ {self.data.times[idx]}')
        self._pre_line.set_ydata(pre)
        self._post_line.set_ydata(post + pre)
        self._norm_line.set_ydata(norm)

        self._fig.canvas.draw_idle()
    
    def update_borders(self, val):
        assert self._widgets is not None
        self._pre_edge_slice = self.data.energyRange2idx(
            self.para.edge_pos + float(self._widgets.pre_start.text),
            self.para.edge_pos + float(self._widgets.pre_stop.text),
        )
        self._post_edge_slice = self.data.energyRange2idx(
            self.para.edge_pos + float(self._widgets.post_start.text),
            self.para.edge_pos + float(self._widgets.post_stop.text),
        )
        self.update_T(None)

    def update_model_conf(self, region: Literal["pre", "post"]):
        assert self._widgets is not None

        model_conf: dict[str, widgetTypes] = getattr(self._widgets, region + "_conf")

        while model_conf:
            w, w_val = model_conf.popitem()
            assert isinstance(w_val, widgetTypes)
            w_val.disconnect_events()
            w_val.ax.remove()
            del w_val

        model = getattr(self, region)
        assert isinstance(model, BackgroundModel)
        for i, (attr, info) in enumerate(model.__class__.model_fields.items()):
            x_pos = .05 if region == "pre" else .55
            if info.annotation == int:
                conf_option = ["1","2","3"]
            else:
                print("lolololol")
                print(info)
                conf_option = ["1","2","3"]
            active = conf_option.index(str(getattr(model, attr)))
            ax = plt.axes((x_pos, .80 - .05*i, .4, .05))
            model_conf[attr] = widgets.RadioButtons(ax, conf_option, active=active)
            # this is not displayed, i dont know why
            label = ax.text(0.05, 0.5, attr, va="center", ha="left", visible=True)
        
        self._widgets.position_conf_labels()
        self._widgets.add_CallBacks(self.update_T, self.update_borders, self.update_fit)


    def update_fit(self, val: str):
        assert self._widgets is not None
        pre_model = self._widgets.preType.value_selected
        post_model = self._widgets.postType.value_selected

        if pre_model != self.pre.__class__.__name__:
            new_pre = PREMODELS[pre_model].model_validate({})
            self.pre = new_pre
            ### update Slider logic
            self.update_model_conf("pre")

        if post_model != self.post.__class__.__name__:
            self.post = POSTMODELS[post_model].model_validate({})
            ### slider update logic
            self.update_model_conf("post")

        for w_attr, w_val in self._widgets.pre_conf.items():
            assert isinstance(w_val, widgets.RadioButtons)
            try:
                val_int = int(w_val.value_selected)
            except:
                continue
            if getattr(self.pre, w_attr) != val_int:
                setattr(self.pre, w_attr, val_int)
        
        for w_attr, w_val in self._widgets.post_conf.items():
            assert isinstance(w_val, widgets.RadioButtons)
            try:
                val_int = int(w_val.value_selected)
            except:
                continue
            if getattr(self.post, w_attr) != val_int:
                setattr(self.post, w_attr, val_int)

        self.update_T(None)
        pass

    def get_line_data(self, pre_slice, post_slice) -> tuple[
        npt.NDArray[np.floating[Any]],
        npt.NDArray[np.floating[Any]],
        npt.NDArray[np.floating[Any]],
        npt.NDArray[np.floating[Any]]
        ]:
        assert isinstance(self._widgets, widget_store)
        t_idx = int(self._widgets.Tslider.val)
        data = self.data[[t_idx]]

        deriv = np.gradient(data.absorption[:,slice(pre_slice.stop, post_slice.start)], axis=1)
        e0_idx = np.argmax(sp.ndimage.gaussian_filter1d(deriv, 3, axis=1), axis=1) + pre_slice.stop

        pre = self.pre.fit_transform(data, pre_slice, e0_idx = e0_idx)
        data.absorption -= pre

        post = self.post.fit_transform(data, post_slice, e0_idx = e0_idx)
        data.absorption /= post
        return self.data.absorption[t_idx], data.absorption[0], pre[0], post[0]

    def _e0_idx(self, pre_idx: int, post_idx: int) -> npt.NDArray[np.integer]:
        energy_slice = slice(pre_idx, post_idx)
        deriv = np.gradient(self.data.absorption[:, energy_slice], axis = 1) / np.gradient(self.data.energies[energy_slice])
        return np.argmax(sp.ndimage.gaussian_filter1d(deriv, 3, axis=1), axis=1) + pre_idx
    
def normGUI(conf_path: pathlib.Path, cli_context: dict):
    XAS = XASPipeline()
    if not conf_path.is_absolute():
        conf_path = pathlib.Path(__file__).parent.resolve() / conf_path
    with open(conf_path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Problem during parsing of {conf_path}: {exc}")
        
    config = config.get("global", {})

    xp = {k: v for k, v in config.items() if k in inspect.signature(XASPara.__init__).parameters.keys()}

    context = {}
    for conf, val in config.items():
        if conf in inspect.signature(XASPara.__init__).parameters.keys():
            continue
        elif conf in PipelineContext.model_fields.keys():
            context[conf] = val
        else:
            raise ValueError(f"Global configuration parameter '{conf}' with value '{val}' not recognized")
    context.update(cli_context)
    context = PipelineContext.model_validate(context)

    guiNorm = GUINorm.model_validate({
        "data": XASData.extracter(logging.Logger("XAS-Pipeline"), context.path, context.exp_name, context.beamline),
        "para": XASPara(**xp)
        })
    guiNorm.interactiveNormalizer()

#region main
def main():
    argParser = argparse.ArgumentParser(description="XAS-Pipeline")

    argParser.add_argument("exp_name")
    argParser.add_argument("--path", type=pathlib.Path, help="Path of the raw data. Can also be provided in the config.")
    argParser.add_argument("--beamline", type=str, help="Beamline-mode (e.g. Balder, P65_T, P65_F, P65_SSD) to correctly read the data. Can also be provided in the config.")
    argParser.add_argument("--plot", type=bool, help="Overwrite of default Value for Preprocessor plotting (Default: False). Can also be provided in the config.")
    argParser.add_argument("-c", "--config", default="config.yaml", type=str, help="Path of the .yaml file serving as the config")
    argParser.add_argument("-g", action="store_true", default=False, help="Enter the GUI modus for the Normalizer")

    args = argParser.parse_args()

    if args.g:
        normGUI(pathlib.Path(args.config), {k: v for k, v in vars(args).items() if k not in ["config", ""] and v is not None})
    else:
        runPipeline(pathlib.Path(args.config), {k: v for k, v in vars(args).items() if k not in ["config", ""] and v is not None})
#endregion

if __name__ == "__main__":
    main()