"""
Data layer : modalities (lazy-loaded volumes reoriented to RSA), subjects and datasets.
Every modality knows how to turn itself into napari layer tuples (data, kwargs, layer_type).
"""

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
import warnings

import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
import numpy as np
from skimage.measure import marching_cubes

def reorient(image, axcodes=("R", "S", "A")):
    """Return a copy of `img` whose array axes follow `axcodes`"""
    ornt= ornt_transform(io_orientation(image.affine), axcodes2ornt(axcodes))
    return image.as_reoriented(ornt)

# Abstract class as a base for all the visualisation classes
class Modality(ABC): 
    layer_type : str = "image"

    def __init__(self, path=None, data=None, spacing=(1,1,1), name = None) :
        self.path, self._data, self.spacing = path, data, tuple(spacing)
        self.name = name or (Path(path).name if path else type(self).__name__)
        self.affine = None  # voxel (of `data`, RSA) -> mm, set by load() ; unknown for in-memory data

    def to_mm(self, voxel) :
        if self.affine is None :
            raise ValueError(f"{self.name} : no affine (data not loaded from a file)")
        return tuple(float(v) for v in (self.affine @ [*voxel, 1.0])[:3])

    @cached_property
    def data(self) : 
        return self._data if self._data is not None else self.load()

    @abstractmethod
    def load(self) : ...

    def crop(self,slices):
        return type(self)(data=self.data[slices], spacing=self.spacing, name = self.name)

    def style(self) -> dict: 
        return {}

    def layers(self, mode="2d") -> list :
        data = self.data  # load first : load() sets the real spacing
        kw = dict(name=self.name, scale=self.spacing) | self.style()
        return [(data, kw, self.layer_type)]


# Conceptual layers (only loading the data)
class ScalarVolume(Modality) :
    def load(self) :
        img = reorient(nib.load(self.path), ("R","S","A"))
        self.spacing = img.header.get_zooms()
        self.affine = img.affine
        return np.asarray(img.dataobj, dtype=np.float32)

class LabelVolume(ScalarVolume) : 
    layer_type = "labels"
    def load(self) :
        return super().load().astype(np.int32)

class Surface(Modality):
    layer_type = "surface"
    def load(self) :
        ...

    def crop(self, slices) :
        return self

### Effective dataclasses

class T1w(ScalarVolume):
    def style(self) :
        lo, hi = np.percentile(self.data[self.data > 0], [1, 99.5])
        return dict(colormap="gray", contrast_limits=(lo,hi))

class ZScore(ScalarVolume):
    def style(self) :
        m = np.percentile(np.abs(self.data), 99)
        return dict(colormap="bwr", contrast_limits=(-m,m))

    def binarize(self, threshold=2.0) :
        return BinaryMask(data=(self.data > threshold).astype(np.int8),
                          spacing = self.spacing,
                          name = f"{self.name}>{threshold}")
    def layers(self, mode="2d") : 
        if mode == "3d" : 
            return self.binarize().layers("3d")
        return super().layers(mode)

class BinaryMask(LabelVolume):
    def layers(self, mode="2d"):
        if mode=="3d" :
            data = self.data  # load first : load() sets the real spacing
            verts, faces, *_ = marching_cubes(data, 0.5, spacing=self.spacing)
            return [((verts, faces), dict(name=self.name, shading="smooth"), "surface")]
        return super().layers(mode)

MODALITIES_TYPES = {
    "t1w" : T1w,
    "zscore" : ZScore,
    "acapulco" : LabelVolume,
    "mask" : BinaryMask,
}

# ------------------------------------------------------------------------
# ------------------ Subjects and Dataset Management ---------------------
# ------------------------------------------------------------------------

class Subject :
    def __init__(self, sid : str, sub_path : Path | str, modalities : dict[str, Modality]) :
        self.sid = sid
        self.sub_path = Path(sub_path)
        self.mod = modalities

    def __getitem__(self, key) :
        return self.mod[key]

    def _check_paths(self) :
        for key, mod in self.mod.items() :
            if mod.path is not None and self.sub_path not in Path(mod.path).parents :
                raise ValueError(f"Modality path : {key} ({mod.path}) does "
                                 f"not match subject root : {self.sub_path}")

    def layers(self, mode="2d", keys=None):
        keys = keys or self.mod.keys()
        return [ld for k in keys for ld in self.mod[k].layers(mode)]

DEFAULT_LAYOUT = {
    "t1w" : "{sid}/bbox_cerebellum/{sid}_t1w_cropped.nii.gz",
    "zscore" : "{sid}/sheetness/{sid}_local_z_score_sheetness_filter5mm.nii.gz",
    "acapulco" : "{sid}/acapulco/{sid}_seg_acapulco_cropped.nii.gz",
}

SES_LAYOUT = {
    "t1w" : "{ses}/{sid}/bbox_cerebellum/{sid}_t1w_cropped.nii.gz",
    "zscore" : "{ses}/{sid}/sheetness/{sid}_local_z_score_sheetness_filter5mm.nii.gz",
    "acapulco" : "{ses}/{sid}/acapulco/{sid}_seg_acapulco_cropped.nii.gz",
}

class Dataset :
    """
    `layout` maps a modality key to a path pattern RELATIVE to `root`
    (Path.glob rejects absolute patterns). Patterns may contain {sid},
    {ses} and glob wildcards, and must resolve to exactly one file.
    A key absent from MODALITIES_TYPES, or one whose class must be
    overridden for this dataset, takes a (pattern, cls) tuple.
    """
    def __init__(self, root : str | Path, name : str, layout : dict = DEFAULT_LAYOUT):
        self.path = Path(root)
        self.name = name
        self.layout = layout

    def _spec(self, key) :
        if key not in self.layout :
            known = ("known modality, but not provided by this dataset" if key in MODALITIES_TYPES
                     else "unknown modality")
            raise ValueError(f"{key}: {known}. {self.name} provides : {sorted(self.layout)}")
        entry = self.layout[key]
        if isinstance(entry, tuple) :
            return entry
        if key not in MODALITIES_TYPES :
            raise ValueError(f"{key}: no class registered in MODALITIES_TYPES, "
                             f"give a (pattern, cls) tuple in the {self.name} layout")
        return entry, MODALITIES_TYPES[key]

    def _resolve(self, pattern, sid, ses):
        if "{ses}" in pattern and ses is None :
            raise ValueError(f"pattern {pattern} needs a session : pass ses=...")
        rel = pattern.format(sid=sid, ses=ses)
        matches = sorted(self.path.glob(rel))
        if len(matches) != 1 :
            raise FileNotFoundError(f"{len(matches)} matches for {self.path / rel} : {matches}")
        return matches[0]


    def load_sub(self, sid : str, modalities : list[str] | None = None, 
                 ses : str | None = None, strict : bool = False) :
        keys = modalities or list(self.layout)
        mods = {}
        for key in keys :
            pattern, cls = self._spec(key)
            try :
                path = self._resolve(pattern, sid, ses)
            except FileNotFoundError as e:
                if strict : 
                    raise 
                warnings.warn(f"{self.name} : skipping {key} ({e})")
                continue
            mods[key] = cls(path=path, name=key)
        return Subject(sid, self.path / sid, mods)

    def list_subjects(self, pattern = "*"):
        # "*" : sids may be bare ids (HCP) as well as BIDS "sub-XXXX" ; hidden folders skipped
        return sorted(p.name for p in self.path.glob(pattern) if p.is_dir() and not p.name.startswith("."))
