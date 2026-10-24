"""
Pack the files `cerebellum-viz view` opens for a list of subjects into one archive, keeping the
dataset's folder structure (<sid>/bbox_cerebellum/..., <sid>/sheetness/...), so the extracted folder is
a dataset root the viewer reads directly.

Per subject : the cropped T1w, the local z-scored sheetness and the cropped ACAPULCO labels (found
wherever they are under <sid>/ : `acapulco/` or `bbox_cerebellum/` depending on the pipeline version).
The NIfTI are already gzipped, so the archive is a plain .tar (re-gzipping gains nothing).

    python scripts/pack_viewer_files.py --root <dataset_root> --subjects subjects.txt --out hcp30_viewer.tar
    tar -xf hcp30_viewer.tar -C <dest>        # then : cerebellum-viz view <dest>/<root name> <sid>

`--subjects` : a text file, one sid per line (or a csv with a `sid` column, first --n rows).
"""
import argparse
import tarfile
from pathlib import Path

import pandas as pd

PATTERNS = {
    "t1w": "{sid}/bbox_cerebellum/{sid}_t1w_cropped.nii.gz",
    "zscore": "{sid}/sheetness/{sid}_local_z_score_sheetness_filter5mm.nii.gz",
    "acapulco": "{sid}/**/{sid}_seg_acapulco_cropped.nii.gz",
}


def read_subjects(path: Path, n: int | None) -> list[str]:
    if path.suffix == ".csv":
        sids = pd.read_csv(path, dtype=str)["sid"].tolist()
    else:
        sids = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    return sids[:n] if n else sids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="dataset root holding one folder per subject")
    ap.add_argument("--subjects", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None, help="keep the first n subjects")
    ap.add_argument("--out", type=Path, required=True, help="archive path (.tar)")
    args = ap.parse_args()

    sids = read_subjects(args.subjects, args.n)
    files, missing = [], []
    for sid in sids:
        for key, pattern in PATTERNS.items():
            hits = sorted(args.root.glob(pattern.format(sid=sid)))
            if len(hits) == 1:
                files.append(hits[0])
            else:
                missing.append(f"{sid}:{key} ({len(hits)} matches)")
    if missing:
        print(f"{len(missing)} file(s) not found or ambiguous, skipped:\n  " + "\n  ".join(missing))

    arc_root = args.root.name  # archive paths start with the dataset folder name
    with tarfile.open(args.out, "w") as tar:
        for f in files:
            tar.add(f, arcname=f"{arc_root}/{f.relative_to(args.root)}")
    size = args.out.stat().st_size / 1e6
    print(f"{len(files)} files from {len(sids)} subjects -> {args.out} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
