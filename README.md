# cerebellum-light-env

A light, portable working environment for the cerebellum project: napari for visualisation,
nibabel for NIfTI, and the `cerebellum_viz` package (viewer + blinded vermis annotation tool).
No BrainVISA (Anatomist, Morphologist, AIMS), so it installs with plain conda or mamba on
**Linux** and **macOS Apple Silicon**.

`cerebellum_viz` here is a copy of the package in `working-env` (carafon, 2026-10-24). The two are
independent: changes made in one are not carried over to the other.

## Install

```bash
git clone <this repo> && cd cerebellum-light-env
mamba env create -f environment.yml      # or: conda env create -f environment.yml
mamba activate cerebellum-light
cerebellum-viz --help
```

After editing `environment.yml`: `mamba env update -f environment.yml --prune`.

## Use

```bash
cerebellum-viz view <dataset_root> <sid> [-m t1w zscore] [--ses ses-01] [--3d]
cerebellum-viz annotate <project_root> <rater>
cerebellum-viz status <project_root>
```

From Python or a notebook:

```python
from cerebellum_viz import view
viewer, sub = view("/path/to/dataset", "sub-XXXX", ["t1w", "zscore"])
```

Images are read with nibabel and reoriented to RSA (superior up, anterior left in sagittal views).

## What is included

| Purpose | Packages |
|---|---|
| Visualisation | napari 0.7, PyQt, magicgui |
| Images / analysis | nibabel, numpy, scipy, pandas, scikit-image, matplotlib, SimpleITK, trimesh |
| Notebooks | jupyter, ipykernel |
| Pipelines | snakemake 9, tqdm |

Not included, on purpose: the BrainVISA suite, gudhi / POT, dcm2niix and the shell tools of
`working-env`. Add a package under `dependencies:` in `environment.yml` if you need it.

## Data

No imaging data or subject identifiers in this repo (`.gitignore` blocks NIfTI and arrays).
Point the viewer at data on the machine you work on.
