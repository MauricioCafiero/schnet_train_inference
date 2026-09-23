"""Which PaiNN layer gives the best OOD signal? (cf. ../mace/code/layer_sweep.py)

Every per-atom layer is captured with forward hooks in one pass per structure
and scored with the ood_spk.py recipe: unit-normalize, cosine distance to the
nearest same-element atom of a reference pool built from the training frames,
mean over atoms. Each vector layer is also scored after subtracting the
per-element pool mean ("centered"). The mean removes the component every atom
of an element shares, which otherwise compresses cosine distances toward 0.

Candidate layers (PaiNN, 3 blocks, 128 features):
    emb            element embedding (a function of Z only: degenerate by construction)
    int{k}, mix{k} scalar features after the message / update step of block k
    vnorm{k}       per-channel norm of the vector features after block k (invariant)
    readout        hidden layer of the energy readout MLP (64-d)
    eps_z          per-atom energy: |eps_i - mean_Z| / std_Z  (MACE's ezMean analogue)
mix3 is what ood_spk.py uses (``scalar_representation``).

Two tests per layer:
  separation   median score of each set / max score of the held-out rot250
               frames, and AUROC(held-out vs set). OFF23 is restricted to
               frames whose elements were all in training.
  error        Spearman correlation between the score and the model's actual
               per-frame force MAE and |energy error| per atom on rot1_gfn2
               (221 GFN2-labelled frames of the same rotaxane).

    python layer_sweep_spk.py ../output/rot250_painn/best_model_ep200
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import read

import schnetpack as spk
import schnetpack.transform as trn

from ood_spk import test_sets

_DATA = Path(__file__).resolve().parent.parent / "data"
TRAINED_Z = {1, 6, 7, 8, 9}


class Capture:
    """Run the full model (energy + forces) and keep every per-atom layer."""

    def __init__(self, model_path, cutoff=5.0):
        self.model = torch.load(model_path, map_location="cpu", weights_only=False).double().eval()
        self.conv = spk.interfaces.AtomsConverter(
            neighbor_list=trn.MatScipyNeighborList(cutoff), dtype=torch.float64, device="cpu")
        self.acts = {}
        rep, out = self.model.representation, self.model.output_modules[0].outnet
        hook = lambda name, fn: (lambda m, i, o: self.acts.__setitem__(name, fn(o)))
        rep.embedding.register_forward_hook(hook("emb", lambda o: o))
        for k, (it, mx) in enumerate(zip(rep.interactions, rep.mixing), 1):
            it.register_forward_hook(hook(f"int{k}", lambda o: o[0].squeeze(1)))
            mx.register_forward_hook(hook(f"mix{k}", lambda o: o[0].squeeze(1)))
            mx.register_forward_hook(hook(f"vnorm{k}", lambda o: torch.linalg.norm(o[1], dim=1)))
        out[0].register_forward_hook(hook("readout", lambda o: o))
        out[1].register_forward_hook(hook("eps", lambda o: o))

    def __call__(self, atoms):
        self.acts.clear()
        res = self.model(self.conv(atoms))
        acts = {k: v.detach().reshape(len(atoms), -1).numpy().astype(np.float32)
                for k, v in self.acts.items()}
        return acts, float(res["energy"].detach()), res["forces"].detach().numpy()


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def auroc(neg, pos):
    """P(score of a random pos > score of a random neg) (Mann-Whitney)."""
    s = np.concatenate([neg, pos])
    r = s.argsort().argsort() + 1.0
    return (r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--cutoff", type=float, default=5.0)
    a = p.parse_args(argv)
    cap = Capture(a.model, a.cutoff)

    # reference pool: every layer, every training atom
    pool, pz = {}, []
    for f in read(_DATA / "rot250_gfn2_train.xyz", ":"):
        acts, _, _ = cap(f)
        for k, v in acts.items():
            pool.setdefault(k, []).append(v)
        pz.append(f.numbers)
    pz = np.concatenate(pz)
    pool = {k: np.concatenate(v) for k, v in pool.items()}
    elements = [int(z) for z in np.unique(pz)]
    mean = {k: {z: v[pz == z].mean(0) for z in elements} for k, v in pool.items()}
    eps_sd = {z: pool["eps"][pz == z].std() for z in elements}
    vector_layers = [k for k in pool if k != "eps"]
    variants = [(k, c) for k in vector_layers for c in (False, True)]
    ref = {(k, c): {z: unit(pool[k][pz == z] - (mean[k][z] if c else 0)) for z in elements}
           for k, c in variants}

    def score(acts, numbers):
        s = {}
        for k, c in variants:
            d = np.ones(len(numbers))
            for z in np.unique(numbers):
                if int(z) in ref[(k, c)]:
                    idx = numbers == z
                    x = unit(acts[k][idx] - (mean[k][int(z)] if c else 0))
                    d[idx] = 1 - (x @ ref[(k, c)][int(z)].T).max(1)
            s[f"{k}{'-c' if c else ''}"] = d.mean()
        ez = [abs(acts["eps"][i, 0] - mean["eps"][int(z)][0]) / eps_sd[int(z)]
              if int(z) in eps_sd else np.nan for i, z in enumerate(numbers)]
        s["eps_z"] = float(np.nanmean(ez))
        return s

    sets = test_sets()
    keep = {"rot250 valid (held out)": "valid", "rot1 GFN2 MD/normal modes": "rot1",
            "bench whole: rod": "rod", "bench whole: wheel": "wheel",
            "dethread1 (112-atom system)": "dethread", "S66 monomers": "S66",
            "OFF23 test sample": "OFF23"}
    scores, err = {}, {"F": [], "E": []}
    for name, short in keep.items():
        frames = sets[name]
        if short == "OFF23":
            frames = [f for f in frames if set(f.numbers) <= TRAINED_Z]
        rows = []
        for f in frames:
            acts, e, forces = cap(f)
            rows.append(score(acts, f.numbers))
            if short == "rot1":
                err["F"].append(np.abs(forces - f.arrays["REF_forces"]).mean())
                err["E"].append(abs(e - f.info["REF_energy"]) / len(f))
        scores[short] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        print(f"scored {short}: {len(frames)} structures", flush=True)

    layers = list(scores["valid"])
    ood = ["rot1", "rod", "wheel", "dethread", "S66", "OFF23"]
    table = {}
    for k in layers:
        v = scores["valid"][k]
        thr = v.max()
        table[k] = {
            "valid_max": float(thr),
            **{f"ratio_{s}": float(np.median(scores[s][k]) / thr) if thr > 0 else float("nan") for s in ood},
            **{f"auc_{s}": float(auroc(v, scores[s][k])) for s in ood},
            "rho_F_rot1": spearman(scores["rot1"][k], err["F"]),
            "rho_E_rot1": spearman(scores["rot1"][k], err["E"]),
        }

    print(f"\nrot1 per-frame errors: force MAE {1000*np.min(err['F']):.0f}-{1000*np.max(err['F']):.0f} meV/A, "
          f"|dE| {1000*np.min(err['E']):.2f}-{1000*np.max(err['E']):.2f} meV/atom")
    print("\nratio = median(set score) / max(held-out score);  AUROC held-out vs set;  "
          "rho = Spearman(score, error) on rot1")
    hdr = (f"{'layer':12s} {'valid max':>10s} " + "".join(f"{'r:'+s:>9s}" for s in ood)
           + "".join(f"{'auc:'+s:>11s}" for s in ("rot1", "rod", "wheel", "S66"))
           + f"{'rho F':>7s}{'rho E':>7s}")
    print(hdr)
    for k, t in table.items():
        print(f"{k:12s} {t['valid_max']:10.4f} " + "".join(f"{t['ratio_'+s]:9.1f}" for s in ood)
              + "".join(f"{t['auc_'+s]:11.2f}" for s in ("rot1", "rod", "wheel", "S66"))
              + f"{t['rho_F_rot1']:7.2f}{t['rho_E_rot1']:7.2f}")

    out = Path(a.model).parent / "layer_sweep.json"
    out.write_text(json.dumps(table, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
