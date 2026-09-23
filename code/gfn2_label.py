"""Label XYZ frames with GFN2-xTB energies + forces -> extxyz for mace_run_train.

    python gfn2_label.py in.xyz out.xyz --workers 4

Deliberately torch-free: torch and tblite bundle separate libomp runtimes that
segfault when loaded into one process (that is what killed the earlier mixed
test, not tblite threading itself). This script never imports torch/mace, and
parallelism comes from N single-threaded worker processes (OMP=1 each), which
is faster than OMP threads for ~100-atom frames anyway (thread overhead eats
the parallel gains at that size).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # per worker; see docstring

import argparse
import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


def read_frames(path: str) -> list[Atoms]:
    """Plain multi-frame XYZ; MDTraj-style names (O1x) -> element (O)."""
    lines = Path(path).read_text().splitlines()
    frames, i = [], 0
    while i < len(lines):
        n = int(lines[i].split()[0])
        syms, pos = [], []
        for line in lines[i + 2:i + 2 + n]:
            f = line.split()
            syms.append(re.match(r"[A-Za-z]+", f[0]).group(0))
            pos.append([float(x) for x in f[1:4]])
        frames.append(Atoms(syms, positions=pos))
        i += 2 + n
    return frames


def _label(args):
    idx, syms, pos = args
    from tblite.ase import TBLite
    atoms = Atoms(syms, positions=pos)
    atoms.calc = TBLite(method="GFN2-xTB", verbosity=0)
    atoms.info["REF_energy"] = atoms.get_potential_energy()
    atoms.arrays["REF_forces"] = atoms.get_forces()
    atoms.info["frame"] = idx
    atoms.calc = None
    return atoms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    frames = read_frames(args.input)
    print(f"{len(frames)} frames -> GFN2 labels with {args.workers} workers")

    from concurrent.futures import ProcessPoolExecutor
    jobs = [(i, list(a.symbols), a.positions.tolist())
            for i, a in enumerate(frames)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        labeled = list(ex.map(_label, jobs))

    write(args.output, labeled := labeled, format="extxyz")
    e = np.array([a.info["REF_energy"] for a in labeled])
    print(f"wrote {args.output}  E range: {e.min():.3f} .. {e.max():.3f} eV")


if __name__ == "__main__":
    main()