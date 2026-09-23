"""Latent-distance OOD check for a trained SchNetPack (PaiNN) model.

Same signal as ../mace/code/activation_ood.py, applied in reverse: there the
reference distribution was MACE-OFF23's broad small-molecule training data and
rotaxanes were the novel chemistry; here the model has only ever seen one
rotaxane, so ordinary molecules are the out-of-distribution case.

Signal: each atom's final PaiNN scalar feature (``scalar_representation``, the
per-atom vector the energy readout acts on) is unit-normalized and scored by
cosine distance to the nearest same-element atom in a reference pool built from
the training frames. A structure's score is the mean over its atoms (``max``
is reported too). Atoms of an element absent from training score 1.0 (the
maximum) and are counted separately.

The in-distribution scale is calibrated on the held-out validation frames of
the same rotaxane; the threshold is their maximum score.

    python ood_spk.py ../output/rot250_painn/best_model

Writes ``ood_pool.npz`` and ``ood_scores.csv`` next to the model.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import read

import schnetpack as spk
import schnetpack.transform as trn

from gfn2_label import read_frames   # MDTraj-style XYZ (dethread frames)

_DATA = Path(__file__).resolve().parent.parent / "data"


def load_s66(path=_DATA / "ood" / "S66_psi4.py"):
    """S66 dimers and both monomers from the Psi4 database module
    (Řezáč, Riley, Hobza, JCTC 2011). Parsing as in ../mace/code/ood_datasets.py."""
    txt = Path(path).read_text()
    pat = re.compile(r"GEOS\['%s-%s-dimer'\s*%\s*\(dbse,\s*'(\d+)'\)\]\s*=\s*"
                     r"qcdb\.Molecule\(\"\"\"(.*?)\"\"\"", re.S)

    def frag(text):
        syms, pos = [], []
        for line in text.strip().splitlines():
            p = line.split()
            if len(p) >= 4 and p[0] in chemical_symbols:
                syms.append(p[0])
                pos.append([float(x) for x in p[1:4]])
        return Atoms(syms, positions=pos)

    dimers, monomers = [], []
    for m in pat.finditer(txt):
        fa, fb = re.split(r"^--\s*$", m.group(2), flags=re.M)[:2]
        a, b = frag(fa), frag(fb)
        dimers.append(a + b)
        monomers += [a, b]
    return dimers, monomers


def test_sets():
    """name -> list of Atoms, ordered from in-distribution to far OOD."""
    bench = _DATA / "rotaxane_bench"
    parts = lambda sub, n: [read(d / f"{n}.xyz") for d in sorted((bench / sub).iterdir())]
    s66_dimers, s66_monomers = load_s66()
    return {
        "rot250 valid (held out)": read(_DATA / "rot250_gfn2_valid.xyz", ":"),
        "rot1 GFN2 MD/normal modes": read(_DATA / "rot1_gfn2.xyz", ":"),
        "bench whole: dimer": parts("whole", "dimer"),
        "bench whole: rod": parts("whole", "rod"),
        "bench whole: wheel": parts("whole", "wheel"),
        "bench stacking: dimer": parts("stacking", "dimer"),
        "bench stacking: rod": parts("stacking", "rod"),
        "bench stacking: wheel": parts("stacking", "wheel"),
        "dethread1 (112-atom system)": read_frames(str(_DATA / "ood" / "dethread1_10.xyz")),
        "S66 dimers": s66_dimers,
        "S66 monomers": s66_monomers,
        "OFF23 test sample": read(_DATA / "ood" / "off23_test_sample300.xyz", ":"),
        "rosuvastatin": [read(_DATA / "ood" / "rosuvastatin.xyz")],
    }


class LatentScorer:
    def __init__(self, model_path, cutoff=5.0):
        self.model = torch.load(model_path, map_location="cpu", weights_only=False).double().eval()
        self.conv = spk.interfaces.AtomsConverter(
            neighbor_list=trn.MatScipyNeighborList(cutoff), dtype=torch.float64, device="cpu")
        self.pool = {}

    @torch.no_grad()
    def latents(self, atoms):
        """Unit-normalized final scalar features, one row per atom."""
        inputs = self.conv(atoms)
        for m in self.model.input_modules:
            inputs = m(inputs)
        q = self.model.representation(inputs)["scalar_representation"]
        q = q.reshape(len(atoms), -1).numpy()
        return q / np.linalg.norm(q, axis=1, keepdims=True)

    def build_pool(self, frames):
        feats, zs = [], []
        for a in frames:
            feats.append(self.latents(a))
            zs.append(a.numbers)
        feats, zs = np.concatenate(feats), np.concatenate(zs)
        self.pool = {int(z): feats[zs == z] for z in np.unique(zs)}

    def score(self, atoms):
        """Per-atom nearest-neighbour cosine distance to the same-element pool."""
        x = self.latents(atoms)
        d = np.ones(len(atoms))
        for z in np.unique(atoms.numbers):
            if int(z) in self.pool:
                idx = atoms.numbers == z
                d[idx] = 1.0 - (x[idx] @ self.pool[int(z)].T).max(axis=1)
        unseen = int(sum(z not in self.pool for z in atoms.numbers))
        return d, unseen


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="path to a SchNetPack best_model")
    p.add_argument("--train", nargs="+", default=[str(_DATA / "rot250_gfn2_train.xyz")],
                   help="training frames that define the reference pool")
    p.add_argument("--cutoff", type=float, default=5.0)
    a = p.parse_args(argv)
    out = Path(a.model).parent

    sc = LatentScorer(a.model, a.cutoff)
    sc.build_pool([f for path in a.train for f in read(path, ":")])
    np.savez(out / "ood_pool.npz", **{chemical_symbols[z]: v for z, v in sc.pool.items()})
    print("pool:", {chemical_symbols[z]: len(v) for z, v in sc.pool.items()})

    rows, threshold = [], None
    print(f"\n{'set':30s} {'n':>4s} {'mean score: min':>16s} {'median':>7s} {'max':>7s} "
          f"{'>thr':>6s} {'unseen-el':>9s}")
    for name, frames in test_sets().items():
        means = []
        for i, atoms in enumerate(frames):
            d, unseen = sc.score(atoms)
            means.append(d.mean())
            rows.append({"set": name, "index": i, "formula": atoms.get_chemical_formula(),
                         "n_atoms": len(atoms), "mean": d.mean(), "max": d.max(),
                         "unseen_element_atoms": unseen})
        means = np.array(means)
        if threshold is None:            # first set = held-out in-distribution frames
            threshold = means.max()
        n_unseen = sum(r["unseen_element_atoms"] > 0 for r in rows if r["set"] == name)
        print(f"{name:30s} {len(means):4d} {means.min():16.4f} {np.median(means):7.4f} "
              f"{means.max():7.4f} {np.mean(means > threshold):6.0%} {n_unseen:9d}")
    print(f"\nthreshold (max of held-out rot250 frames): {threshold:.4f}")

    with open(out / "ood_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'ood_pool.npz'} and {out / 'ood_scores.csv'}")


if __name__ == "__main__":
    main()
