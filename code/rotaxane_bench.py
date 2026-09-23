"""Rotaxane interaction-energy benchmark -- the target-chemistry test set.

Adapted from ../mace/code/rotaxane_bench.py to score SchNetPack models. Two
sets, both from the rotaxanes-results study
(https://github.com/MauricioCafiero/rotaxanes-results):

**stacking** -- capped wheel/rod fragment pairs at the center-ring stacked well
and at both stoppers, with a **DLPNO-CCSD(T)/aug-cc-pVTZ** reference plus DFT,
GFN2-xTB and stock MACE-OFF23 columns from that study
(``QM_comparisons/README.md``, ``DETHREADING_REPORT.md`` §4.1/§4.1a). The GFN2
column matters most here: it is the level these models were trained on, so a
good model should land on it.

**whole** -- four ~128-atom double-stack geometries scored without capping,
referenced against UMA (``ROT3QM/README.md`` §4a). UMA agrees with CCSD(T) to
~0.7 kcal/mol on the stacking set, so it is a usable reference; the ordering of
the four and the size of the central-ring/iso-side outlier gap are the
scientifically meaningful quantities.

E_int = E(dimer) - E(rod) - E(wheel), in kcal/mol. SchNetPack models are run
in float64 on CPU. MACE numbers are not recomputed: they are read from
``data/mace_bench_results.json`` (copied from the mace repo) for comparison.

    python rotaxane_bench.py --spk-model painn=../output/rot250_painn/best_model \
        --mace ft-medium ft-large
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_BENCH = _REPO / "data" / "rotaxane_bench"
_KCAL = 23.0605487415
OUT = _REPO / "output" / "bench_results.json"
MACE_RES = _REPO / "data" / "mace_bench_results.json"

# DETHREADING_REPORT.md §4.1 / §4.1a and QM_comparisons/README.md
STACK_REF = {
    "center_4F":          {"ccsdt": -10.10, "dft": -10.09, "gfn2": -10.26, "mace": -10.67},
    "center_2F":          {"ccsdt":  -9.63, "dft":  -9.30, "gfn2":  -9.90, "mace":  -9.77},
    "center_0F":          {"ccsdt":  -7.59, "dft":  -7.28, "gfn2":  -8.58, "mace":  -8.17},
    "weak_stopper_real":  {"ccsdt":  -8.09, "dft":  -8.23, "gfn2":  -7.84, "mace":  -8.09},
    "strong_stopper_real": {"ccsdt":  None, "dft": -11.53, "gfn2": -11.71, "mace": -11.51},
    "strong_stopper":     {"ccsdt":  None, "dft": -11.52, "gfn2": -12.33, "mace": -10.71},
    "weak_stopper":       {"ccsdt":  None, "dft":  -9.50, "gfn2": -10.52, "mace":  -8.99},
}
# ROT3QM/README.md §4a (UMA whole-structure)
WHOLE_REF = {"cf3_ring": -38.17, "iso_ring": -36.73,
             "central_cf3side": -36.05, "central_isoside": -30.62}


def _calc(path, cutoff):
    import torch
    import schnetpack as spk
    import schnetpack.transform as trn
    return spk.interfaces.SpkCalculator(
        str(path), neighbor_list=trn.MatScipyNeighborList(cutoff),
        energy_unit="eV", position_unit="Ang", device="cpu", dtype=torch.float64)


def _eint(d, calc):
    from ase.io import read
    es = []
    for n in ("dimer", "rod", "wheel"):
        at = read(d / f"{n}.xyz")
        at.calc = calc
        es.append(float(at.get_potential_energy()))
        at.calc = None
    return (es[0] - es[1] - es[2]) * _KCAL


def run(models, cutoff, sets=("stacking", "whole")):
    """models: {name: path to a SchNetPack best_model}. Always rescored (a
    resumed run keeps its name); results are stored in OUT."""
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, path in models.items():
        calc = None
        for sub, ref in (("stacking", STACK_REF), ("whole", WHOLE_REF)):
            if sub not in sets:
                continue
            res.setdefault(sub, {}).setdefault(name, {})
            for key in ref:
                d = _BENCH / sub / key
                if not d.exists():
                    continue
                calc = calc or _calc(path, cutoff)
                res[sub][name][key] = _eint(d, calc)
                print(f"  {name:12s} {sub:8s} {key:20s} {res[sub][name][key]:8.2f}",
                      flush=True)
                OUT.parent.mkdir(exist_ok=True)
                OUT.write_text(json.dumps(res, indent=1))
    return res


def report(models, mace_models):
    res = json.loads(OUT.read_text())
    mace = json.loads(MACE_RES.read_text())
    for sub in ("stacking", "whole"):
        for m in mace_models:
            res.setdefault(sub, {})[f"mace:{m}"] = mace[sub][m]
    models = list(models) + [f"mace:{m}" for m in mace_models]

    print("\n=== stacking fragments (kcal/mol) — CCSD(T)/aug-cc-pVTZ reference ===")
    hdr = f"  {'pairing':22s} {'CCSD(T)':>8s} {'DFT':>7s} {'GFN2':>7s} {'OFF23':>7s}"
    print(hdr + "".join(f" {m:>14s}" for m in models))
    for key, r in STACK_REF.items():
        if key not in res.get("stacking", {}).get(models[0], {}):
            continue
        c = f"{r['ccsdt']:8.2f}" if r["ccsdt"] is not None else f"{'--':>8s}"
        print(f"  {key:22s} {c} {r['dft']:7.2f} {r['gfn2']:7.2f} {r['mace']:7.2f}"
              + "".join(f" {res['stacking'][m][key]:14.2f}" for m in models))
    for label, col in (("vs CCSD(T)", "ccsdt"), ("vs GFN2 (trained level)", "gfn2")):
        line = f"  {label:22s} {'':8s} {'':7s} {'':7s} {'':7s}"
        for m in models:
            e = [res["stacking"][m][k] - STACK_REF[k][col] for k in STACK_REF
                 if STACK_REF[k][col] is not None
                 and k in res["stacking"].get(m, {})]
            line += f" {np.abs(e).mean():14.2f}" if e else f" {'--':>14s}"
        print(line + "   <- MAE")

    if "whole" in res:
        print("\n=== whole-structure double stacks (kcal/mol) — UMA reference ===")
        order = sorted(WHOLE_REF, key=lambda k: WHOLE_REF[k], reverse=True)
        print(f"  {'geometry':22s} {'UMA':>8s}" + "".join(f" {m:>14s}" for m in models))
        for key in order:
            print(f"  {key:22s} {WHOLE_REF[key]:8.2f}"
                  + "".join(f" {res['whole'][m][key]:14.2f}" for m in models))
        gap = lambda d: d["central_isoside"] - d["central_cf3side"]
        print(f"  {'outlier gap':22s} {gap(WHOLE_REF):8.2f}"
              + "".join(f" {gap(res['whole'][m]):14.2f}" for m in models))
        print(f"  {'ordering matches UMA':22s} {'--':>8s}"
              + "".join(f" {str(sorted(res['whole'][m], key=lambda k: res['whole'][m][k], reverse=True) == order):>14s}"
                        for m in models))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spk-model", action="append", required=True, metavar="NAME=PATH",
                   help="SchNetPack best_model to score (repeatable)")
    p.add_argument("--mace", nargs="*", default=["off-medium", "ft-medium", "ft-large"],
                   help="stored MACE columns to show: off-medium off-large "
                        "ft-medium ft-large ft-dimer")
    p.add_argument("--cutoff", type=float, default=5.0, help="must match training")
    p.add_argument("--sets", nargs="*", default=["stacking", "whole"])
    a = p.parse_args(argv)
    models = dict(spec.partition("=")[::2] for spec in a.spk_model)
    run(models, a.cutoff, sets=tuple(a.sets))
    report(list(models), a.mace)


if __name__ == "__main__":
    main()
