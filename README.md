# SchNetPack on GFN2-xTB rotaxane data

Train [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack)
potentials (PaiNN / SchNet) from scratch on GFN2-xTB energies + forces for a
144-atom rotaxane, and score them on a rotaxane interaction-energy benchmark
against DLPNO-CCSD(T), UMA and fine-tuned MACE-OFF23.

This is the from-scratch counterpart to the `../mace` project (fine-tuning the
MACE-OFF23 foundation model on the same data). The data, the GFN2 labelling
code and the benchmark were copied from there so this repo is self-contained.

## Layout

```
code/
  train_spk.py        train PaiNN/SchNet on extxyz (REF_energy / REF_forces) -> output/<name>/
  predict_spk.py      inference: XYZ or SMILES -> energy/forces, optional geometry optimization
  rotaxane_bench.py   interaction-energy benchmark for SchNetPack models (+ stored MACE columns)
  gfn2_label.py       label XYZ frames with GFN2-xTB energies + forces (tblite; torch-free)
  gfn2_data.py        generate GFN2 datasets: relax, Langevin MD, normal-mode sampling
data/
  rot250_gfn2{,_train,_valid}.xyz   250 GFN2-labelled rotaxane frames, 225/25 split (the MACE split)
  rot1_gfn2{,_train,_valid}.xyz     221 frames from GFN2 MD + normal modes (same rotaxane)
  rot1_sampled_250.xyz              the unlabelled 250-frame sample behind rot250
  rotaxane_bench/                   stacking (fragment) and whole-structure dimer/rod/wheel geometries
  mace_bench_results.json           MACE results on the benchmark (from ../mace, not recomputed)
examples/
  run_rot250_painn.sh               train PaiNN on rot250, then run the benchmark
output/                             run directories, logs, benchmark results
```

## Setup

```sh
python3.12 -m venv .venv
.venv/bin/pip install schnetpack tblite psutil
```

Installed here: Python 3.12, schnetpack 2.2.0 (built from the GitHub source),
torch 2.14, pytorch-lightning 2.x, tblite 0.7.0, psutil. `tblite` is only
needed for labelling new data (`gfn2_label.py`, `gfn2_data.py`).

## Usage

Train (outputs go to `output/<name>/`):

```sh
cd code
../.venv/bin/python train_spk.py --train ../data/rot250_gfn2_train.xyz \
    --valid ../data/rot250_gfn2_valid.xyz --name rot250_painn --device cpu --epochs 200
```

Main options: `--model painn|schnet`, `--features 128`, `--interactions 3`,
`--cutoff 5.0`, `--batch-size 4`, `--micro-batch N`, `--lr 5e-4`,
`--energy-weight 0.05` (force weight is `1 - energy-weight`), `--device cpu|mps|cuda`.
Several `--train` / `--valid` files can be given (e.g. add `rot1_gfn2_train.xyz`).

A run directory holds `best_model` (inference model; load with
`schnetpack.interfaces.SpkCalculator`), `lightning/metrics.csv` (per-step and
per-epoch losses, MAEs, learning rate and memory), and `metrics.json` (float64
CPU evaluation of `best_model` on the validation frames).

Inference with a trained model (float64, CPU; energies in eV on the GFN2 scale):

```sh
# energy + forces for every frame; MAEs are printed if the file has REF_energy / REF_forces
../.venv/bin/python predict_spk.py ../output/rot250_painn/best_model --xyz ../data/rot250_gfn2_valid.xyz

# relax, write the relaxed structures with predicted energy/forces
../.venv/bin/python predict_spk.py ../output/rot250_painn/best_model --xyz frame.xyz --optimize -o relaxed.xyz

# from SMILES: RDKit ETKDGv3 conformers, MMFF-minimized, lowest kept, then relaxed with the model
../.venv/bin/python predict_spk.py ../output/rot250_painn/best_model --smiles "c1ccccc1O" --optimize
```

Options: `--fmax 0.01` (eV/Å), `--steps 500`, `--optimizer FIRE|BFGS|LBFGS`,
`--cutoff 5.0` (must match training). The script prints a warning for elements
the model never saw in training. A model trained on one rotaxane is
extrapolating on any other chemistry.

Benchmark a trained model:

```sh
../.venv/bin/python rotaxane_bench.py --spk-model painn=../output/rot250_painn/best_model
```

The whole example (train, then benchmark) runs as a detached job:

```sh
~/miniforge3/bin/python -c "import subprocess; subprocess.Popen(['zsh','/Users/cafierom/python_mac/schnet/examples/run_rot250_painn.sh'], stdout=open('/Users/cafierom/python_mac/schnet/output/rot250_painn.log','ab'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)"
```

`EPOCHS`, `DEVICE` and `NAME` environment variables override the defaults
(200 / cpu / rot250_painn in the run below).

Label new frames with GFN2-xTB:

```sh
../.venv/bin/python gfn2_label.py frames.xyz frames_gfn2.xyz --workers 4
```

## Apple Silicon notes (MPS, memory)

- **float64 on MPS.** MPS has no float64. SchNetPack's `AddOffsets` fills its
  `mean` buffer with float64 dataset statistics, so moving the model to MPS
  fails with `Cannot convert a MPS Tensor to float64`. `train_spk.py`
  computes the per-atom mean in float32 and passes it explicitly, and leaves
  out the `CastTo64` postprocessor that the stock SchNetPack configs add.
- **Memory.** Training on forces needs a second backward pass. Its transient
  peak is not a leak (memory is flat step to step), but on MPS it costs far
  more than on CPU. Measured for these 144-atom frames (about 3,800 neighbor
  pairs each at 5 Å) with PaiNN-128:

  | frames per pass | CPU process RSS | MPS driver peak |
  |---|---|---|
  | 1 | – | 1.0 GB |
  | 2 | – | 1.7 GB |
  | 4 | 1.2–2.0 GB | 3.4 GB |

  At batch 4, MPS pushed an 8 GB machine into swap. `--micro-batch 2` halves
  the MPS peak with gradient accumulation. The gradients are the same because
  every frame has the same atom count.
- **Speed.** On CPU (4 OpenMP threads) PaiNN-128 takes about 24 s per epoch
  for 225 frames and peaks at about 2 GB. On this 8 GB machine, MPS with
  micro-batch 2 still pushed free memory below 20%, so the runs here use CPU.

## Results

_Pending: the 200-epoch run is in progress._
