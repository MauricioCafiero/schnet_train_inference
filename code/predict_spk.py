"""Run a trained SchNetPack model on XYZ frames or a SMILES string.

Single-point energy + forces, and optionally a geometry optimization.

    # energies (and MAEs if the file carries REF_energy / REF_forces)
    python predict_spk.py ../output/rot250_painn/best_model --xyz ../data/rot250_gfn2_valid.xyz

    # relax every frame, write the relaxed structures
    python predict_spk.py ../output/rot250_painn/best_model --xyz frame.xyz --optimize -o relaxed.xyz

    # SMILES -> RDKit 3D conformer (lowest of several MMFF-minimized) -> relax
    python predict_spk.py ../output/rot250_painn/best_model --smiles "c1ccccc1O" --optimize

Energies are total GFN2-scale energies in eV (the saved model re-adds the
GFN2 atom references), forces in eV/Ang. The model runs in float64 on CPU.

Scope: a model trained on rot250 has only seen one rotaxane (H, C, N, O, F).
It will return numbers for anything, but other elements are meaningless
(their atom reference is zero) and other chemistry is extrapolation.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse

import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS

import schnetpack as spk
import schnetpack.transform as trn

TRAINED_Z = {1, 6, 7, 8, 9}   # elements in the rot250 training set
_OPTIMIZERS = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}


def smiles_to_atoms(smiles, n_conformers=5, seed=42):
    """SMILES -> add Hs -> embed n_conformers (ETKDGv3, RMS-pruned) -> MMFF
    (UFF if MMFF lacks parameters) -> lowest-energy conformer as Atoms.
    Same recipe as ../mace/code/mace_calc.smiles_to_atoms."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    if not cids:
        raise RuntimeError(f"RDKit failed to embed any conformers for {smiles!r}")

    props = AllChem.MMFFGetMoleculeProperties(mol)
    energies = []
    for cid in cids:
        ff = (AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid) if props
              else AllChem.UFFGetMoleculeForceField(mol, confId=cid))
        ff.Minimize(maxIts=500)
        energies.append((ff.CalcEnergy(), cid))
    _, best = min(energies)

    atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()],
                  positions=mol.GetConformer(best).GetPositions())
    atoms.info.update(smiles=smiles, forcefield="mmff" if props else "uff",
                      n_conformers_sampled=len(energies))
    return atoms


def get_calculator(model_path, cutoff=5.0):
    return spk.interfaces.SpkCalculator(
        str(model_path), neighbor_list=trn.MatScipyNeighborList(cutoff),
        energy_unit="eV", position_unit="Ang", device="cpu", dtype=torch.float64)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="path to a SchNetPack best_model")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--xyz", help="XYZ / extxyz file (all frames are used)")
    src.add_argument("--smiles", help="SMILES string; 3D structure built with RDKit")
    p.add_argument("--optimize", action="store_true", help="relax each structure")
    p.add_argument("--fmax", type=float, default=0.01, help="force convergence, eV/Ang")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--optimizer", choices=list(_OPTIMIZERS), default="FIRE")
    p.add_argument("--cutoff", type=float, default=5.0, help="must match training")
    p.add_argument("-o", "--out", help="write structures + predicted energy/forces (extxyz)")
    a = p.parse_args(argv)

    frames = read(a.xyz, ":") if a.xyz else [smiles_to_atoms(a.smiles)]
    calc = get_calculator(a.model, a.cutoff)

    results, de, df = [], [], []
    for i, atoms in enumerate(frames):
        extra = set(atoms.numbers) - TRAINED_Z
        if extra:
            print(f"WARNING frame {i}: elements Z={sorted(extra)} not in training data")
        ref_e = atoms.info.get("REF_energy")
        ref_f = atoms.arrays.get("REF_forces")
        atoms.calc = calc
        e0 = atoms.get_potential_energy()
        f0 = atoms.get_forces()
        if ref_e is not None:
            de.append((e0 - ref_e) / len(atoms))
        if ref_f is not None:
            df.append(np.abs(f0 - ref_f).ravel())

        row = {"frame": i, "formula": atoms.get_chemical_formula(), "energy": e0,
               "fmax": float(np.linalg.norm(f0, axis=1).max())}
        if a.optimize:
            start = atoms.positions.copy()
            converged = _OPTIMIZERS[a.optimizer](atoms, logfile=None).run(fmax=a.fmax, steps=a.steps)
            f = atoms.get_forces()
            row.update(energy_opt=atoms.get_potential_energy(), converged=bool(converged),
                       fmax_opt=float(np.linalg.norm(f, axis=1).max()),
                       rmsd=float(np.sqrt(((atoms.positions - start) ** 2).sum(1).mean())))
        results.append(row)

        # keep the (possibly relaxed) structure with the model's energy/forces
        atoms.info["energy"] = atoms.get_potential_energy()
        atoms.arrays["forces"] = atoms.get_forces()
        atoms.calc = None

    for r in results:
        line = f"{r['frame']:4d} {r['formula']:14s} E = {r['energy']:14.6f} eV  fmax {r['fmax']:7.3f}"
        if a.optimize:
            line += (f"  | relaxed E = {r['energy_opt']:14.6f} eV  dE = "
                     f"{r['energy_opt'] - r['energy']:9.4f}  fmax {r['fmax_opt']:6.3f}  "
                     f"RMSD {r['rmsd']:5.2f} A  {'converged' if r['converged'] else 'NOT converged'}")
        print(line)
    if de:
        print(f"energy MAE vs REF_energy: {1000 * np.mean(np.abs(de)):.2f} meV/atom ({len(de)} frames)")
    if df:
        print(f"forces MAE vs REF_forces: {1000 * np.concatenate(df).mean():.1f} meV/Ang")

    if a.out:
        for atoms in frames:   # the REF_* keys describe the input geometry, not a relaxed one
            if a.optimize:
                atoms.info.pop("REF_energy", None)
                atoms.arrays.pop("REF_forces", None)
        write(a.out, frames, format="extxyz")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
