"""Generate reference structures + forces with GFN2-xTB (TBLite) via ASE.

This module is the *data-generation* half of a MACE fine-tuning workflow:

    GFN2-xTB reference (this module)  -->  extxyz with E & F  -->  mace_run_train
    (ASE front-end via TBLite)             (energy + forces)        (fine-tune a
                                                                    MACE foundation)

It wraps the ASE interface to TBLite (``tblite.ase.TBLite``) and provides a
single high-level entry point :func:`generate_dataset` that, for each input
molecule,

    1. relaxes the geometry to a GFN2 minimum,
    2. samples diverse configurations with NVT Langevin MD at one or more
       temperatures,
    3. (optionally) perturbs along GFN2 normal modes for extra coverage,
    4. recomputes GFN2 energy + forces on every collected frame, and
    5. writes an extended XYZ file ready for ``mace_run_train``.

The output uses both the plain keys (``energy`` / ``forces``) and the
``REF_*`` keys (``REF_energy`` / ``REF_forces``) so you can choose the
fine-tuning flags freely:

    mace_run_train --train_file train.xyz --valid_file valid.xyz \
        --energy_key=REF_energy --forces_key=REF_forces ...

Energies are in eV, forces in eV/Angstrom (ASE/MACE native units). GFN2-xTB
covers the full periodic table up to Z=86, so it is suitable for generating
training data for organic, main-group, and many transition-metal systems.

Example
-------
>>> from gfn2_data import generate_dataset
>>> from ase.build import molecule
>>> mols = [molecule("H2O"), molecule("CH3OH"), molecule("NH3")]
>>> generate_dataset(mols, outfile="gfn2_data.xyz",
...                  temperatures=[100, 300, 500], md_steps=2000,
...                  sample_every=50)
"""

from __future__ import annotations

import os

# macOS / Apple-Silicon stability guards (see mace_calc.py for rationale).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from ase import Atoms, units
from ase.io import Trajectory, read, write
from ase.optimize import FIRE
from ase.md import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.vibrations import Vibrations

log = logging.getLogger("gfn2_data")

GFN2_METHOD = "GFN2-xTB"


# ---------------------------------------------------------------------------
# Calculator factory
# ---------------------------------------------------------------------------
def get_gfn2_calculator(
    method: str = GFN2_METHOD,
    charge: Optional[float] = None,
    multiplicity: Optional[int] = None,
    accuracy: float = 1.0,
    max_iterations: int = 250,
    verbosity: int = 0,
    solvation: Optional[tuple] = None,
    **kwargs,
):
    """Build an ASE-compatible TBLite calculator (default: GFN2-xTB).

    Parameters
    ----------
    method : str
        ``"GFN2-xTB"`` (default), ``"GFN1-xTB"``, or ``"IPEA1-xTB"``.
    charge, multiplicity : optional
        Total charge (e) and spin multiplicity (2S+1) of the system.
    accuracy : float
        TBLite numerical accuracy (1.0 = default, higher = tighter).
    solvation : tuple, optional
        e.g. ``("alpb", "water")`` or ``("gbsa", "water")`` to generate data
        in solution.
    """
    from tblite.ase import TBLite

    return TBLite(
        method=method,
        charge=charge,
        multiplicity=multiplicity,
        accuracy=accuracy,
        max_iterations=max_iterations,
        verbosity=verbosity,
        solvation=solvation,
        **kwargs,
    )


def attach_gfn2(atoms: Atoms, **calc_kw) -> Atoms:
    """Attach a GFN2 calculator to ``atoms`` (in-place)."""
    atoms.calc = get_gfn2_calculator(**calc_kw)
    return atoms


# ---------------------------------------------------------------------------
# Low-level tasks
# ---------------------------------------------------------------------------
def gfn2_singlepoint(atoms: Atoms, **calc_kw) -> dict:
    """Single-point GFN2 energy (eV) and forces (eV/Angstrom)."""
    if atoms.calc is None or not _is_tblite(atoms.calc):
        attach_gfn2(atoms, **calc_kw)
    return {
        "energy": float(atoms.get_potential_energy()),
        "forces": np.asarray(atoms.get_forces()),
    }


def gfn2_optimize(
    atoms: Atoms,
    fmax: float = 0.01,  # eV/Angstrom
    steps: int = 500,
    **calc_kw,
) -> dict:
    """Geometry optimization at the GFN2 level. Mutates ``atoms`` in place."""
    if atoms.calc is None or not _is_tblite(atoms.calc):
        attach_gfn2(atoms, **calc_kw)
    dyn = FIRE(atoms, logfile=None)
    dyn.run(fmax=fmax, steps=steps)
    return {
        "energy": float(atoms.get_potential_energy()),
        "forces": np.asarray(atoms.get_forces()),
        "fmax": float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
    }


def _is_tblite(calc) -> bool:
    return type(calc).__name__ == "TBLite"


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------
def sample_md(
    atoms: Atoms,
    temperatures_K: Sequence[float] = (300.0,),
    md_steps: int = 2000,
    timestep_fs: float = 0.5,
    friction: float = 0.01,  # 1/fs
    sample_every: int = 50,
    seed: Optional[int] = None,
    **calc_kw,
) -> list[Atoms]:
    """Sample configurations from NVT Langevin MD at each temperature.

    Returns a list of Atoms copies (without calculator) at sampled intervals.
    The input ``atoms`` should already be reasonably relaxed; it is left at the
    end of the last MD run.
    """
    if atoms.calc is None or not _is_tblite(atoms.calc):
        attach_gfn2(atoms, **calc_kw)

    frames: list[Atoms] = []
    rng = np.random.default_rng(seed)
    for T in temperatures_K:
        MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=rng, force_temp=True)
        Stationary(atoms)
        ZeroRotation(atoms)
        dyn = Langevin(
            atoms, timestep_fs * units.fs,
            temperature_K=T, friction=friction / units.fs, logfile=None, rng=rng,
        )
        collected: list[Atoms] = []

        def _collect():
            collected.append(atoms.copy())

        dyn.attach(_collect, interval=sample_every)
        dyn.run(md_steps)
        for a in collected:
            a.calc = None
            a.info["temperature_K"] = T
            frames.append(a)
        log.info("MD @ %g K: collected %d frames", T, len(collected))
    return frames


def sample_normal_modes(
    atoms: Atoms,
    n_modes: Optional[int] = None,
    n_displacements: int = 3,
    amplitude_A: float = 0.03,  # Angstrom, ~ root-mean-square displacement
    seed: Optional[int] = None,
    name: str = "gfn2_vib",
    **calc_kw,
) -> list[Atoms]:
    """Perturb along harmonic normal modes to cover off-equilibrium geometries.

    Uses GFN2 finite-difference vibrations. For each of the first ``n_modes``
    non-imaginary modes, displaces by +/- random fractions of ``amplitude_A``.
    Returns a list of Atoms copies (without calculator). This is the classic
    "normal-mode sampling" recipe for ML potential training data.
    """
    if atoms.calc is None or not _is_tblite(atoms.calc):
        attach_gfn2(atoms, **calc_kw)
    work = atoms.copy()
    work.calc = atoms.calc
    vib = Vibrations(work, name=name)
    vib.run()
    freqs = np.asarray(vib.get_frequencies())
    freqs_real = np.real(freqs)  # imag part is ~0 for stable modes; safe cast
    masses = atoms.get_masses()
    natoms = len(atoms)
    rng = np.random.default_rng(seed)

    # ASE returns mass-weighted mode vectors; convert to Cartesian displacement
    # via e_cart = e / sqrt(masses), then normalize to a unit vector.
    def _cart_mode(i: int) -> np.ndarray:
        mw = np.asarray(vib.get_mode(i)).reshape(natoms, 3)
        cart = mw / np.sqrt(masses)[:, None]
        n = np.linalg.norm(cart)
        return cart / n if n > 0 else cart

    # Pick real, positive-frequency modes (skip translations/rotations/imag)
    real_idx = [i for i, f in enumerate(freqs_real) if f > 50.0]  # cm^-1
    if n_modes is not None:
        real_idx = real_idx[:n_modes]
    log.info("Normal-mode sampling: %d real modes selected", len(real_idx))

    frames: list[Atoms] = []
    for i in real_idx:
        mode_vec = _cart_mode(i)
        if not np.any(mode_vec):
            continue
        for _ in range(n_displacements):
            scale = rng.uniform(-amplitude_A, amplitude_A)
            disp = atoms.copy()
            disp.positions = atoms.positions + scale * mode_vec
            disp.calc = None
            disp.info["mode_index"] = int(i)
            disp.info["mode_freq_cm1"] = float(freqs_real[i])
            frames.append(disp)
    # clean up vib scratch files
    try:
        vib.clean()
    except Exception:  # pragma: no cover
        pass
    return frames


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def _stamp(atoms: Atoms, config_type: str, label: str) -> Atoms:
    """Attach a fresh GFN2 calculator, compute E & F, and tag info/arrays."""
    a = atoms.copy()
    a.calc = None
    # Strip transient MD data (momenta etc.) so the training frame is clean.
    for k in ("momenta", "initial_charges", "initial_magmoms"):
        if k in a.arrays:
            del a.arrays[k]
    attach_gfn2(a)
    energy = float(a.get_potential_energy())
    forces = np.asarray(a.get_forces()).copy()
    a.calc = None
    a.info["energy"] = energy
    a.info["REF_energy"] = energy
    a.info["config_type"] = config_type
    a.info["label"] = label
    a.arrays["forces"] = forces
    a.arrays["REF_forces"] = forces
    return a


def generate_dataset(
    molecules: Iterable[Atoms],
    outfile: Union[str, Path] = "gfn2_data.xyz",
    relax: bool = True,
    relax_fmax: float = 0.01,
    relax_steps: int = 500,
    temperatures_K: Sequence[float] = (100.0, 300.0, 500.0),
    md_steps: int = 2000,
    timestep_fs: float = 0.5,
    sample_every: int = 50,
    do_normal_modes: bool = True,
    nmodes_per_mol: int = 6,
    nmodes_displacements: int = 3,
    nmodes_amplitude_A: float = 0.03,
    valid_fraction: float = 0.1,
    train_file: Optional[Union[str, Path]] = None,
    valid_file: Optional[Union[str, Path]] = None,
    seed: Optional[int] = 0,
    label_attr: str = "label",
    **calc_kw,
) -> dict:
    """Generate a GFN2 energy/forces dataset for MACE fine-tuning.

    Parameters
    ----------
    molecules : iterable of Atoms
        Input geometries. Each is relaxed (if ``relax``) and then sampled.
    outfile : path
        Full dataset (train+valid) written here as extended XYZ.
    valid_fraction : float
        Fraction of frames reserved as the validation set. If
        ``train_file``/``valid_file`` are given they are written separately;
        otherwise the split is implicit in ``outfile`` (use MACE's
        ``--valid_file`` pointing at ``outfile`` is *not* recommended -- pass
        ``train_file`` and ``valid_file``).
    label_attr : str
        Info key used to name each molecule (defaults to ``"label"``; falls
        back to a counter).

    Returns
    -------
    dict
        Counts: ``{"n_total", "n_train", "n_valid", "outfile", ...}``.
    """
    rng = np.random.default_rng(seed)
    all_frames: list[Atoms] = []

    for idx, mol in enumerate(molecules):
        atoms = mol.copy()
        atoms.calc = None
        label = str(atoms.info.get(label_attr) or atoms.info.get("config_type") or f"mol{idx}")

        if relax:
            try:
                gfn2_optimize(atoms, fmax=relax_fmax, steps=relax_steps, **calc_kw)
            except Exception as e:  # pragma: no cover
                log.warning("Relaxation failed for %s: %s", label, e)
                continue
            # the relaxed minimum itself is a useful training point
            all_frames.append(_stamp(atoms, "gfn2_min", label))

        # MD sampling
        md_frames = sample_md(
            atoms,
            temperatures_K=temperatures_K,
            md_steps=md_steps,
            timestep_fs=timestep_fs,
            sample_every=sample_every,
            seed=int(rng.integers(1 << 31)) if seed is not None else None,
            **calc_kw,
        )
        for fr in md_frames:
            all_frames.append(_stamp(fr, "gfn2_md", label))

        # Normal-mode sampling (start from the relaxed geometry)
        if do_normal_modes:
            nm_frames = sample_normal_modes(
                atoms,
                n_modes=nmodes_per_mol,
                n_displacements=nmodes_displacements,
                amplitude_A=nmodes_amplitude_A,
                seed=int(rng.integers(1 << 31)) if seed is not None else None,
                **calc_kw,
            )
            for fr in nm_frames:
                all_frames.append(_stamp(fr, "gfn2_nm", label))

        log.info("%s: %d frames total so far", label, len(all_frames))

    if not all_frames:
        raise RuntimeError("No frames generated; check inputs and GFN2 setup.")

    # Shuffle and split
    order = rng.permutation(len(all_frames))
    all_frames = [all_frames[i] for i in order]
    n_valid = max(1, int(round(valid_fraction * len(all_frames))))
    valid = all_frames[:n_valid]
    train = all_frames[n_valid:]

    write(str(outfile), all_frames, format="extxyz")
    if train_file is None:
        train_file = Path(outfile).with_name(Path(outfile).stem + "_train.xyz")
    if valid_file is None:
        valid_file = Path(outfile).with_name(Path(outfile).stem + "_valid.xyz")
    write(str(train_file), train, format="extxyz")
    write(str(valid_file), valid, format="extxyz")

    log.info("Wrote %s (%d), %s (%d), %s (%d)",
             outfile, len(all_frames), train_file, len(train), valid_file, len(valid))

    return {
        "n_total": len(all_frames),
        "n_train": len(train),
        "n_valid": len(valid),
        "outfile": str(outfile),
        "train_file": str(train_file),
        "valid_file": str(valid_file),
    }