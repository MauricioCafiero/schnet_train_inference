"""Train a SchNetPack potential (PaiNN or SchNet) on GFN2-xTB extxyz data.

    python train_spk.py --train ../data/rot250_gfn2_train.xyz \
        --valid ../data/rot250_gfn2_valid.xyz --name rot250_painn --device mps

Inputs are the extxyz files written by ``gfn2_label.py`` / ``gfn2_data.py``
(``REF_energy`` in eV, ``REF_forces`` in eV/Ang). Everything a run produces
goes to ``output/<name>/``:

    data.db, split.npz   ASE database + fixed train/valid split
    nbl_cache/           cached neighbor lists (built on the first epoch)
    best_model           best inference model (by val_loss), loadable with
                         SpkCalculator / torch.load
    lightning/           Lightning checkpoints + CSV log (metrics.csv;
                         resumed runs log to lightning/resume_<k>/)
    metrics.json         final float64 CPU evaluation of best_model on valid

MPS notes: Apple's MPS has no float64. SchNetPack's ``AddOffsets`` normally
fills its ``mean`` buffer with float64 dataset statistics, which then cannot
move to the device, so the per-atom mean is computed here in float32 and passed
in explicitly. The ``CastTo64`` postprocessor from the stock configs is left out
for the same reason; the final evaluation upcasts on CPU instead.

Memory: training on forces needs a second backward pass, and on MPS its
transient peak is large (measured for 144-atom rotaxane frames, PaiNN-128: MPS
driver peak ~1.0 / 1.7 / 3.4 GB at 1 / 2 / 4 frames per pass, vs ~1.3 GB process
RSS on CPU at 4). ``--micro-batch`` splits each ``--batch-size`` batch into
smaller passes with gradient accumulation -- identical gradients when all frames
have the same atom count, at a fraction of the peak. ``MemoryLog`` records process
RSS + MPS driver memory to metrics.csv every 10 steps.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from ase.io import read

import schnetpack as spk
import schnetpack.transform as trn

_REPO = Path(__file__).resolve().parent.parent

# GFN2-xTB isolated-atom energies (eV), from ../mace finetune_mace.GFN2_E0S
GFN2_E0S = {1: -10.707211, 6: -48.798080, 7: -70.908087, 8: -102.521807,
            9: -125.698643, 15: -64.604696, 16: -85.619453, 17: -121.975722,
            35: -110.160925, 53: -102.848978}


class MemoryLog(pl.Callback):
    """Log process RSS + MPS driver memory (GB) every 10 (micro-)batches."""

    def __init__(self):
        self.proc = psutil.Process()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 10:
            return
        rss = self.proc.memory_info().rss / 2**30
        mps = torch.mps.driver_allocated_memory() / 2**30 if torch.backends.mps.is_available() else 0.0
        pl_module.log_dict({"mem_rss_gb": rss, "mem_mps_gb": mps}, on_step=True, on_epoch=False)
        print(f"step {trainer.global_step:6d}  rss {rss:5.2f} GB  mps {mps:5.2f} GB", flush=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self.t0 = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"epoch {trainer.current_epoch} train time {time.time() - self.t0:.1f} s", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", nargs="+", required=True, help="training extxyz file(s)")
    p.add_argument("--valid", nargs="+", required=True, help="validation extxyz file(s)")
    p.add_argument("--name", required=True, help="run name -> output/<name>/")
    p.add_argument("--model", choices=["painn", "schnet"], default="painn")
    p.add_argument("--features", type=int, default=128)
    p.add_argument("--interactions", type=int, default=3)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4, help="effective batch per optimizer step")
    p.add_argument("--micro-batch", type=int, default=None,
                   help="frames per forward/backward; gradients are accumulated up to "
                        "--batch-size (same result for equal-size frames, lower peak memory)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--energy-weight", type=float, default=0.05)
    p.add_argument("--device", default="mps", help="mps | cpu | cuda")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-steps", type=int, default=-1, help="for quick probes")
    p.add_argument("--grad-clip", type=float, default=None,
                   help="clip the gradient norm to this value (guards against loss spikes)")
    p.add_argument("--resume", action="store_true",
                   help="continue output/<name>/ from lightning/last.ckpt up to --epochs "
                        "(total); optimizer, LR schedule and best-model score are restored")
    a = p.parse_args(argv)
    micro = a.micro_batch or a.batch_size
    assert a.batch_size % micro == 0, "--batch-size must be a multiple of --micro-batch"

    out = _REPO / "output" / a.name
    out.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(a.seed)

    train = [f for path in a.train for f in read(path, ":")]
    valid = [f for path in a.valid for f in read(path, ":")]

    # ASE db with a fixed split: first len(train) rows train, rest valid
    db = out / "data.db"
    if not db.exists():
        ds = spk.data.ASEAtomsData.create(
            str(db), distance_unit="Ang",
            property_unit_dict={"energy": "eV", "forces": "eV/Ang"})
        ds.add_systems([{"energy": np.array([f.info["REF_energy"]]),
                         "forces": f.arrays["REF_forces"]} for f in train + valid],
                       train + valid)
    split = out / "split.npz"
    np.savez(split, train_idx=np.arange(len(train)),
             val_idx=np.arange(len(train), len(train) + len(valid)),
             test_idx=np.array([], dtype=int))

    atomref = torch.zeros(100)
    for z, e in GFN2_E0S.items():
        atomref[z] = e
    # per-atom mean of the binding energy E - sum(E0), float32 (see MPS notes)
    mean = np.mean([(f.info["REF_energy"] - sum(GFN2_E0S[z] for z in f.numbers)) / len(f)
                    for f in train])
    mean = torch.tensor([mean], dtype=torch.float32)

    data = spk.data.AtomsDataModule(
        str(db), batch_size=micro, val_batch_size=micro,
        num_train=len(train), num_val=len(valid), split_file=str(split),
        transforms=[
            trn.CachedNeighborList(str(out / "nbl_cache"),
                                   trn.MatScipyNeighborList(cutoff=a.cutoff),
                                   keep_cache=True),
            trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=True,
                              atomrefs=atomref, property_mean=mean),
            trn.CastTo32(),
        ],
        num_workers=0, pin_memory=False)

    rbf = spk.nn.GaussianRBF(n_rbf=20, cutoff=a.cutoff)
    cut = spk.nn.CosineCutoff(a.cutoff)
    if a.model == "painn":
        rep = spk.representation.PaiNN(a.features, a.interactions, rbf, cut)
    else:
        rep = spk.representation.SchNet(a.features, a.interactions, rbf, cut)
    model = spk.model.NeuralNetworkPotential(
        representation=rep,
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[spk.atomistic.Atomwise(a.features, output_key="energy"),
                        spk.atomistic.Forces()],
        postprocessors=[trn.AddOffsets("energy", add_mean=True, add_atomrefs=True,
                                       atomrefs=atomref, property_mean=mean)])

    mae = lambda: {"MAE": torchmetrics.MeanAbsoluteError()}
    task = spk.task.AtomisticTask(
        model,
        [spk.task.ModelOutput("energy", torch.nn.MSELoss(), a.energy_weight, metrics=mae()),
         spk.task.ModelOutput("forces", torch.nn.MSELoss(), 1 - a.energy_weight, metrics=mae())],
        optimizer_args={"lr": a.lr},
        scheduler_cls=spk.train.ReduceLROnPlateau,
        scheduler_args={"factor": 0.5, "patience": 25, "min_lr": 1e-6},
        scheduler_monitor="val_loss")

    # Lightning's CSVLogger deletes an existing metrics.csv, so a resumed run
    # logs to its own subfolder (lightning/resume_<k>/metrics.csv)
    ckpt, version = None, ""
    if a.resume:
        ckpt = str(out / "lightning" / "last.ckpt")
        version = f"resume_{len(list((out / 'lightning').glob('resume_*'))) + 1}"
    trainer = pl.Trainer(
        accelerator=a.device, devices=1, max_epochs=a.epochs, max_steps=a.max_steps,
        accumulate_grad_batches=a.batch_size // micro,
        gradient_clip_val=a.grad_clip,
        default_root_dir=str(out / "lightning"),
        logger=CSVLogger(str(out / "lightning"), name="", version=version),
        callbacks=[spk.train.ModelCheckpoint(model_path=str(out / "best_model"),
                                             dirpath=str(out / "lightning"),
                                             monitor="val_loss", save_last=True),
                   pl.callbacks.EarlyStopping("val_loss", patience=100, strict=False),
                   pl.callbacks.LearningRateMonitor(),
                   MemoryLog()],
        enable_progress_bar=False, log_every_n_steps=10)
    t0 = time.time()
    # our own checkpoint; it pickles the model via save_hyperparameters, so a
    # weights-only load (Lightning default) refuses it
    trainer.fit(task, datamodule=data, ckpt_path=ckpt, weights_only=False if ckpt else None)
    wall = time.time() - t0

    if not (out / "best_model").exists():   # e.g. --max-steps probe ended before validation
        print(f"no best_model (stopped at step {trainer.global_step}); skipping evaluation")
        return

    # Final check: best model, float64 on CPU, through the ASE calculator
    calc = spk.interfaces.SpkCalculator(
        str(out / "best_model"), neighbor_list=trn.MatScipyNeighborList(a.cutoff),
        energy_unit="eV", position_unit="Ang", device="cpu", dtype=torch.float64)
    de, df, formula = [], [], []
    for f in valid:
        at = f.copy()
        at.calc = calc
        de.append((at.get_potential_energy() - f.info["REF_energy"]) / len(f))
        df.append(np.abs(at.get_forces() - f.arrays["REF_forces"]).ravel())
        formula.append(f.get_chemical_formula())
        at.calc = None
    res = {"name": a.name, "args": vars(a), "n_train": len(train), "n_valid": len(valid),
           "epochs_run": trainer.current_epoch, "wall_s": round(wall, 1),
           "valid_energy_mae_meV_per_atom": 1000 * float(np.mean(np.abs(de))),
           "valid_forces_mae_meV_per_A": 1000 * float(np.concatenate(df).mean()),
           "n_params": sum(p.numel() for p in model.parameters())}
    if len(set(formula)) > 1:   # mixed validation set (e.g. rotaxane + monomers): per molecule
        res["valid_by_formula"] = {
            fm: {"n": formula.count(fm),
                 "energy_mae_meV_per_atom": 1000 * float(np.mean([abs(e) for e, x in zip(de, formula) if x == fm])),
                 "forces_mae_meV_per_A": 1000 * float(np.concatenate([d for d, x in zip(df, formula) if x == fm]).mean())}
            for fm in sorted(set(formula))}
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
