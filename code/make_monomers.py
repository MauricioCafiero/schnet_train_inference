"""Split rotaxane frames into their rod (axle) and wheel (macrocycle) monomers.

The two components of a rotaxane are mechanically interlocked but not
covalently bonded, so each frame splits into exactly two connected components
of the covalent graph (ASE natural cutoffs). Each monomer keeps the geometry it
has inside the rotaxane: exactly the structures an interaction energy
E(rotaxane) - E(rod) - E(wheel) needs. Label them afterwards with
gfn2_label.py.

    python make_monomers.py ../data/rot250_gfn2_train.xyz ../data/rot250_gfn2_valid.xyz

For each input X.xyz this writes X_rod.xyz and X_wheel.xyz (plain XYZ, frame
order preserved). It checks that every frame gives the same two molecules
(RDKit canonical SMILES from the 3D bonding), and draws both as
../output/monomers.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs
from scipy.sparse.csgraph import connected_components

_REPO = Path(__file__).resolve().parent.parent


def split(atoms):
    """-> (rod, wheel): the larger component is the rod."""
    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)
    n, lab = connected_components(nl.get_connectivity_matrix())
    assert n == 2, f"expected 2 covalent components, got {n}"
    a, b = (atoms[np.where(lab == k)[0]] for k in range(2))
    return (a, b) if len(a) > len(b) else (b, a)


def to_rdkit(atoms):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    xyz = f"{len(atoms)}\n\n" + "\n".join(
        f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions))
    mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    return mol


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+")
    a = p.parse_args(argv)

    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw

    smiles = {"rod": set(), "wheel": set()}
    first = None
    for path in a.inputs:
        rods, wheels = [], []
        for f in read(path, ":"):
            rod, wheel = split(f)
            rods.append(rod)
            wheels.append(wheel)
            for name, m in (("rod", rod), ("wheel", wheel)):
                smiles[name].add(Chem.MolToSmiles(Chem.RemoveHs(to_rdkit(m))))
            first = first or (rod, wheel)
        stem = Path(path).with_suffix("")
        write(f"{stem}_rod.xyz", rods, format="xyz")
        write(f"{stem}_wheel.xyz", wheels, format="xyz")
        print(f"{path}: {len(rods)} frames -> {stem}_rod.xyz ({rods[0].get_chemical_formula()}), "
              f"{stem}_wheel.xyz ({wheels[0].get_chemical_formula()})")

    for name, s in smiles.items():
        print(f"{name}: {len(s)} distinct topolog{'y' if len(s) == 1 else 'ies'}")
        for x in s:
            print("   ", x)

    mols, legends = [], []
    for name, m in zip(("rod", "wheel"), first):
        mol = Chem.RemoveHs(to_rdkit(m))
        AllChem.Compute2DCoords(mol)
        mols.append(mol)
        legends.append(f"{name}: {m.get_chemical_formula()} ({len(m)} atoms)")
    img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(700, 500), legends=legends)
    out = _REPO / "output" / "monomers.png"
    img.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
