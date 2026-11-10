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

With pixi instead (`pixi.toml` + `pixi.lock`, same packages, solved for both platforms):

```bash
pixi install
pixi run viz view <dataset_root> <sid>     # or `pixi shell`, then cerebellum-viz ...
```

Keep `environment.yml` and `pixi.toml` in sync when adding a package (`pixi add <pkg>` updates the
lock).

## Use

```bash
cerebellum-viz view <dataset_root> <sid> [-m t1w zscore] [--ses ses-01] [--3d]
cerebellum-viz annotate <project_root> <rater> [--data-root <dataset_root>]
cerebellum-viz status <project_root>
```

### Where the images are read from

An annotation project stores the dataset root it was created with (`dataset_root` in
`project.json`). When the data are mounted at another place on your computer, give that place with
`--data-root`; the project itself is left unchanged, so raters with different mounts can share it:

```bash
cerebellum-viz annotate /path/to/project rater_name --data-root /path/to/dataset/on/this/computer
```

The data root is checked before napari opens. If it cannot be read (not found, not a folder, no
permission, or a network mount that has dropped), the command stops with one line saying which path
failed and where it came from (`--data-root` or `project.json`), for example:

```
cerebellum-viz: error: Data root not found : /path/to/dataset (from project.json). Check that the disk is mounted, or point to the data with --data-root.
```

From Python, the same override is `annotate(project_root, rater, data_root=...)` or
`AnnotationProject.load(project_root).session(rater, data_root=...)`.

From Python or a notebook:

```python
from cerebellum_viz import view
viewer, sub = view("/path/to/dataset", "sub-XXXX", ["t1w", "zscore"])
```

Images are read with nibabel and reoriented to RSA (superior up, anterior left in sagittal views).

## Protocol v2 : landmarks

`protocols/protocol_v2.json` keeps the v1 inferior and lingula items. The superior item is reduced
to 1 or 2 folia, with a "bifid" tick per folium : the 1.5 configurations come post hoc from the
folium lengths. A ticked answer is stored as one code, base + "+tick" : `2+bifid1`.
It adds landmarks placed on one mid-sagittal slice : fastigium, culmen connection, B5 root and
branching (always), B6 / B7 attachments, SZFg folia connections, tips and split points (of a bifid
folium), lingula root and tip. Which landmarks a case needs depends on its codes (none for ND or S).

In `annotate`, choosing the codes shows exactly the landmarks they need (small cross-hair pointers :
four arms, a gap and a centre dot), parked in a labelled column left of the image ; changing a code removes or adds
landmarks (a landmark hidden and shown again keeps its place). The only action on them is to drag
them into place, one at a time : adding, deleting and multi-selecting are undone, and napari's
layer panels are hidden (contrast and reset view are in the form). The landmarks follow the
sagittal slice shown, and the slice they are saved on is the case's mid-sagittal slice. A case is
saved only when every landmark has been dragged off its parking place, inside the image.
Landmarks go to `answers/<rater>_landmarks.csv` (voxel of the RSA image + mm through its affine),
under the same `save_id` as the codes. Distances in mm, defined in the protocol's `measures` :

```python
from cerebellum_viz import AnnotationProject
AnnotationProject.load("/path/to/project").session("rater").measures()
```

A new project uses v2 with `AnnotationProject.create(..., protocol_path=protocol_file(2))`.

## Protocol v3

`protocols/protocol_v3.json` is v2 with these changes (the numbering follows the vault's protocol note):

- **Two superior folia are always placed.** Folium 1 is the first folium of the SZFg, and folium 2
  is the next folium above it. The code says where folium 2 is attached: 2 = on the SZFg,
  1 = on the culmen branch. Folium 2's position therefore measures where each rater puts the
  culmen-side end of the SZFg.
- **Bifid tick on both folia** whatever the code (`1+bifid2` is valid).
- **Landmarks are placed for S** and hidden only for ND, on every item.
- **Three SZFg folia** are coded S with the comment "3 folia".
- **User definitions** of the B5 root (start of the white-matter thinning), the B5 branching
  (VIIAf / VIIAt / VI division) and the culmen connection (proximal end of the culmen branch).
- **Three new measures:** `sup1_conn_from_fastigium`, `sup2_conn_from_fastigium` and
  `sup2_conn_from_culmen`.

Use it with `protocol_file(3)`. Projects created with v2 keep their v2 answers. To move an existing
project to v3, use `AnnotationProject.update_protocol(protocol_file(3))`; later saves are then
checked against v3.

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
