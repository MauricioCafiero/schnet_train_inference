# SchNetPack on GFN2-xTB rotaxane data

Train [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack)
potentials (PaiNN / SchNet) from scratch on GFN2-xTB energies + forces for a
144-atom rotaxane, and score them on a rotaxane interaction-energy benchmark
against DLPNO-CCSD(T), UMA and fine-tuned MACE-OFF23.

This is the from-scratch counterpart to the `../mace` project (fine-tuning the
MACE-OFF23 foundation model on the same data). The data, the GFN2 labelling
code and the benchmark were copied from there so this repo is self-contained.

## The physics behind the models

### Energy as a sum of local atomic contributions

SchNet and PaiNN are machine-learned interatomic potentials. They learn a map
from nuclear positions and element types to the potential energy surface (PES)
of a reference method, here GFN2-xTB. Both use the same central
approximation, the one behind Behler–Parrinello networks, MACE and most other
modern MLIPs: the total energy is a sum of atomic energies,

$$E(\{\mathbf{r}_i, Z_i\}) = \sum_i \varepsilon_i ,$$

where each atomic energy $\varepsilon_i$ depends only on the atoms inside a
cutoff sphere around atom $i$ (5 Å here). This locality makes the cost scale
linearly with system size, and it lets a model trained on one environment
transfer to others made of similar local environments.

**Forces** are not predicted separately. They are the exact negative
gradient of the predicted energy, computed by automatic differentiation:

$$\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i} .$$

The force field is therefore conservative by construction (a curl-free
gradient field), so MD run with it conserves total energy, and forces are
always consistent with energies. Training uses both energies and forces,
with the loss weighted 0.05 : 0.95. Each 144-atom frame supplies 432 force
components but only one energy, so forces carry most of the information about
the shape of the PES. Fitting forces means differentiating a loss that
already contains a gradient: a second backward pass through the network. That
pass is where most of the training memory goes (see the Apple Silicon notes).

**Energy reference.** GFN2 total energies for this rotaxane are about
−7,364 eV, but the chemically meaningful variation between conformers is of
order 1 eV. Before training, the GFN2 isolated-atom energies $E_0(Z)$ and a
per-atom mean binding energy are subtracted, so the network learns only the
remaining binding energy. The saved model adds these offsets back, so it
predicts total GFN2 energies. This plays the same role as `--E0s` in MACE.

### Symmetries

The energy of an isolated molecule must be unchanged by translating it,
rotating it, or relabeling identical atoms. Both architectures build these
symmetries in rather than learning them from data:

- **Translation invariance:** the inputs are interatomic vectors
  $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$, not absolute positions.
- **Permutation invariance:** information from neighbors is combined by
  summation, which does not depend on neighbor order, and atomic energies are
  summed.
- **Rotation:** SchNet uses only distances $r_{ij}$, so every internal
  feature is rotationally *invariant*. PaiNN also carries vector features
  that rotate with the molecule (*equivariant*). Its energy is read from the
  scalar channel only, so it stays invariant, and the forces, as gradients of
  an invariant energy, rotate correctly with the molecule.

### Describing distances: radial basis and smooth cutoff

Each distance $r_{ij}$ is expanded in 20 Gaussians with fixed centers evenly
spaced from 0 to 5 Å. This is a smooth, rich encoding a network can work
with, similar in spirit to a Fourier expansion. Every pair term is
multiplied by a cosine cutoff

$$f_c(r) = \tfrac{1}{2}\left[\cos\left(\pi r / r_c\right) + 1\right], \qquad r < r_c ,$$

which goes smoothly to zero at $r_c$ = 5 Å. An atom crossing the cutoff
during MD or an optimization therefore causes no jump in energy or force.

### SchNet: continuous-filter convolutions

SchNet (Schütt et al., 2017/2018) gives each atom a feature vector
$\mathbf{x}_i$, initialized from a learned embedding of its element, and
refines it through several interaction blocks. The core of each block is a
**continuous-filter convolution**:

$$\mathbf{x}_i \leftarrow \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot \mathbf{W}(r_{ij}),$$

where the filter $\mathbf{W}(r_{ij})$ is produced by a small neural network
acting on the radial basis of the distance. It generalizes the filters of an
image CNN from a fixed pixel grid to arbitrary continuous atomic positions.
Each message depends on a single pair distance, so one SchNet layer is a
**2-body** operation. Angular information (bond angles, dihedrals, stacking
geometry) comes in only indirectly, as several layers compose pair terms. The
activation is shifted softplus, $\ln(0.5\,e^x + 0.5)$, which is smooth and
keeps second derivatives (forces and their training gradients) well
behaved.

### PaiNN: equivariant vector features

PaiNN (Schütt, Unke & Gastegger, 2021) keeps a scalar feature
$\mathbf{s}_i$ and a **vector feature** $\vec{\mathbf{v}}_i$ (3 × 128 here)
for each atom. The vector features carry directional information that a
distance-only model cannot represent directly. Each of the 3 layers has two
steps.

*Message* (between atoms):

$$\Delta\mathbf{s}_i = \sum_j \phi_s(\mathbf{s}_j) \odot \mathbf{W}_s(r_{ij}), \qquad
\Delta\vec{\mathbf{v}}_i = \sum_j \Big[\phi_{vv}(\mathbf{s}_j) \odot \mathbf{W}_{vv}(r_{ij}) \odot \vec{\mathbf{v}}_j
+ \phi_{vs}(\mathbf{s}_j) \odot \mathbf{W}_{vs}(r_{ij})\, \hat{\mathbf{r}}_{ij}\Big]$$

The $\hat{\mathbf{r}}_{ij}$ term puts bond directions into the vector
channel, and the first term carries neighbors' vectors along.

*Update* (within an atom), with learned linear maps $\mathbf{U}$ and
$\mathbf{V}$ acting on the vector channels:

$$\Delta\vec{\mathbf{v}}_i = \mathbf{a}_{vv} \odot \mathbf{U}\vec{\mathbf{v}}_i, \qquad
\Delta\mathbf{s}_i = \mathbf{a}_{ss} + \mathbf{a}_{sv} \odot \langle \mathbf{U}\vec{\mathbf{v}}_i, \mathbf{V}\vec{\mathbf{v}}_i \rangle ,$$

where the coefficients $\mathbf{a}$ come from a network acting on
$[\mathbf{s}_i, \lVert\mathbf{V}\vec{\mathbf{v}}_i\rVert]$. The norms and inner
products of vectors are rotation-invariant scalars that encode angles
between bonds. Feeding them back into $\mathbf{s}_i$ gives PaiNN
**3-body** (angular) information within a single layer. Only operations that
preserve equivariance are used: scaling vectors by scalars, linear mixing of
vector channels, and inner products. So the vector features rotate exactly
with the molecule. The activation is SiLU. After the last layer, a small
network maps each $\mathbf{s}_i$ to $\varepsilon_i$.

The model trained here uses 3 layers, 128 features and a 5 Å cutoff:
589,057 parameters. Through message passing, information travels up to
3 × 5 = 15 Å, but each individual interaction sees only 5 Å.

### What these models cannot capture here

- **Physics beyond the cutoff.** GFN2-xTB includes D4 dispersion and
  charge-dependent electrostatics that extend well past 5 Å. The model sees
  them only as far as they can be absorbed into local environments within
  the 15 Å message-passing range. Close π-stacking contacts (about
  3.4–3.8 Å) are inside the cutoff, but the long tail of the dispersion
  attraction between a wheel and a rod is truncated. That bears directly on
  the interaction-energy benchmark. SchNetPack has Coulomb/Ewald modules for
  explicit long-range terms (`schnetpack.atomistic.electrostatic`); they are
  not used here.
- **Body order compared with MACE.** MACE builds up to 4-body correlations
  in each layer from higher-order spherical-harmonic features. SchNet is
  2-body per layer, and PaiNN is effectively 3-body with only first-order
  (vector) angular features. MACE-OFF23 is also pretrained on a large organic
  dataset before fine-tuning. The MACE comparison in this repo therefore
  mixes two effects, architecture and pretraining.
- **Data scope.** The training set is 225 frames of one rotaxane near one
  region of conformational space. Nothing in it constrains the model for
  separated fragments. The benchmark's isolated rod and wheel monomers are
  exactly that case, so interaction energies $E_\text{dimer} - E_\text{rod}
  - E_\text{wheel}$ test extrapolation, not interpolation.

### References

- K. T. Schütt et al., *SchNet: A continuous-filter convolutional neural network for modeling quantum interactions*, NeurIPS 2017; *J. Chem. Phys.* **148**, 241722 (2018).
- K. T. Schütt, O. T. Unke, M. Gastegger, *Equivariant message passing for the prediction of tensorial properties and molecular spectra* (PaiNN), ICML 2021.
- K. T. Schütt et al., *SchNetPack 2.0: A neural network toolbox for atomistic machine learning*, *J. Chem. Phys.* **158**, 144801 (2023).
- C. Bannwarth, S. Ehlert, S. Grimme, *GFN2-xTB*, *J. Chem. Theory Comput.* **15**, 1652 (2019).
- I. Batatia et al., *MACE*, NeurIPS 2022; D. P. Kovács et al., *MACE-OFF23*, arXiv:2312.15211.

## Layout

```
code/
  train_spk.py        train PaiNN/SchNet on extxyz (REF_energy / REF_forces) -> output/<name>/
  predict_spk.py      inference: XYZ or SMILES -> energy/forces, optional geometry optimization
  rotaxane_bench.py   interaction-energy benchmark for SchNetPack models (+ stored MACE columns)
  gfn2_label.py       label XYZ frames with GFN2-xTB energies + forces (tblite; torch-free)
  gfn2_data.py        generate GFN2 datasets: relax, Langevin MD, normal-mode sampling
  ood_spk.py          latent-distance OOD score vs the training frames (PaiNN final layer)
  layer_sweep_spk.py  which PaiNN layer gives the best OOD signal
  make_monomers.py    split rotaxane frames into rod + wheel monomers, check topology, draw
data/
  rot250_gfn2{,_train,_valid}.xyz   250 GFN2-labelled rotaxane frames, 225/25 split (the MACE split)
  rot1_gfn2{,_train,_valid}.xyz     221 frames from GFN2 MD + normal modes (same rotaxane)
  rot1_sampled_250.xyz              the unlabelled 250-frame sample behind rot250
  rot250_{rod,wheel}_gfn2_{train,valid}.xyz   GFN2-labelled monomers of the rot250 frames
  rot250_gfn2_{train,valid}_{rod,wheel}.xyz   the same monomers, unlabelled (make_monomers.py)
  rotaxane_bench/                   stacking (fragment) and whole-structure dimer/rod/wheel geometries
  mace_bench_results.json           MACE results on the benchmark (from ../mace, not recomputed)
  ood/                              OOD test sets: S66 (Psi4 module), dethread1 frames, rosuvastatin,
                                    OFF23 test sample (local only)
examples/
  run_rot250_painn.sh               train (or resume) PaiNN on rot250, then benchmark + OOD check
  run_rot250mono_painn.sh           label monomers, train on rotaxane + rod + wheel, benchmark + OOD
output/                             run directories, logs, benchmark results
```

## Setup

```sh
python3.12 -m venv .venv
.venv/bin/pip install schnetpack tblite psutil rdkit
```

Installed here: Python 3.12, schnetpack 2.2.0 (built from the GitHub source),
torch 2.14, pytorch-lightning 2.x, tblite 0.7.0, psutil, rdkit (SMILES input to `predict_spk.py`). `tblite` is only
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
(500 / cpu / rot250_painn). The run below used `EPOCHS=200`. `RESUME=1`
continues an existing run from `lightning/last.ckpt` up to `EPOCHS` total
(`train_spk.py --resume`). Resumed runs log to `lightning/resume_<k>/`,
because Lightning's CSV logger would otherwise delete the existing
`metrics.csv`. Note that `last.ckpt` is only refreshed when validation improves,
so a resume restarts from the best epoch.

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

Run `rot250_painn`: PaiNN with 128 features, 3 layers, 5 Å cutoff, 589k
parameters. Trained from scratch on the 225 rot250 frames and validated on
the same 25 held-out frames used for the MACE fine-tunes. CPU, batch 4, Adam,
energy : force loss weights 0.05 : 0.95, about 24.5 s per epoch, peak process
memory 2.1 GB. Everything is in `output/rot250_painn/`.

The run happened in two stages:

1. **Epochs 0–200** at a constant learning rate of 5e-4 (82 min). Best
   epoch 188. This model is kept as `best_model_ep200`, with
   `metrics_ep200.json`, `bench_ep200.txt` and `ood_ep200.txt`.
2. **Resumed** from the epoch-188 checkpoint (`--resume`), targeting 400
   epochs. At **epoch 204 training diverged**: the validation force MAE jumped
   from 0.067 to 0.50 eV/Å in one epoch, still at a learning rate of 5e-4.
   The plateau scheduler then halved the rate at epochs 224, 250 and 276
   (to 6.25e-5). The model recovered only to 0.085 eV/Å, and early stopping
   (100 epochs without improvement) ended the run at epoch 297 (45 min). The
   final `best_model` is **epoch 197**, from before the divergence. The
   per-epoch log is `lightning/resume_1/metrics.csv`.

### Fit to GFN2-xTB (held-out validation frames)

Numbers are from the float64 CPU evaluation of each model in `metrics*.json`.

| model | training | energy MAE | force MAE |
|---|---|---|---|
| **PaiNN, epoch 188** (`best_model_ep200`) | from scratch, 225 frames | **0.50 meV/atom** | **60.7 meV/Å** |
| **PaiNN, epoch 197** (`best_model`) | from scratch, 225 frames | **0.53 meV/atom** | **59.8 meV/Å** |
| MACE-OFF23 medium + replay | fine-tuned, same frames | 0.5 meV/atom | 30.6 meV/Å |
| MACE-OFF23 large + replay | fine-tuned, same frames | 0.6 meV/atom | 25.7 meV/Å |

Validation curve of the first stage (per structure; the energy MAE is for the
whole 144-atom frame):

| epoch | energy MAE (eV) | force MAE (eV/Å) |
|---|---|---|
| 0 | 2.68 | 0.537 |
| 25 | 0.48 | 0.182 |
| 100 | 0.12 | 0.087 |
| 150 | 0.20 | 0.070 |
| 200 | 0.10 | 0.064 |

The energies match the fine-tuned MACE models; the forces are about 2× worse.
The force error had plateaued near 60 meV/Å by epoch 190. The next attempt at
5e-4 blew up rather than improving, so more epochs at this learning rate are
not the way to close the gap. The standard fixes are gradient clipping
(`gradient_clip_val` in the Lightning trainer), a decaying learning-rate
schedule from the start instead of waiting for a plateau, and SchNetPack's
`ExponentialMovingAverage` callback for the weights. How much of the gap to
MACE is closable at all is open, because MACE starts from a foundation model
pretrained on a large organic dataset.

### Rotaxane interaction-energy benchmark

$E_\text{int} = E_\text{dimer} - E_\text{rod} - E_\text{wheel}$ in kcal/mol,
from `output/rot250_painn/bench_ep200.txt` and `bench.txt`. MACE columns come
from the `../mace` project (`data/mace_bench_results.json`).

**Stacking fragments** (capped wheel/rod pairs; DLPNO-CCSD(T)/aug-cc-pVTZ reference):

| pairing | CCSD(T) | DFT | GFN2 | PaiNN ep188 | PaiNN ep197 | MACE-OFF23 med | MACE ft-med | MACE ft-large |
|---|---|---|---|---|---|---|---|---|
| center_4F | −10.10 | −10.09 | −10.26 | −4.18 | −3.11 | −10.67 | −11.17 | −14.99 |
| center_2F | −9.63 | −9.30 | −9.90 | −4.45 | −3.51 | −9.77 | −10.71 | −14.87 |
| center_0F | −7.59 | −7.28 | −8.58 | −3.44 | −2.70 | −8.17 | −9.23 | −13.04 |
| weak_stopper_real | −8.09 | −8.23 | −7.84 | −3.19 | −2.11 | −8.09 | −8.78 | −11.19 |
| strong_stopper_real | – | −11.53 | −11.71 | −6.40 | −5.01 | −11.51 | −12.35 | −16.71 |
| strong_stopper | – | −11.52 | −12.33 | −4.00 | −2.19 | −10.71 | −13.20 | −16.69 |
| weak_stopper | – | −9.50 | −10.52 | +0.41 | +1.74 | −8.99 | −10.72 | −13.49 |
| **MAE vs CCSD(T)** | | | | **5.04** | **5.99** | 0.32 | 1.12 | 4.67 |
| **MAE vs GFN2** | | | | **6.55** | **7.75** | 0.65 | 0.72 | 4.27 |

**Whole-structure double stacks** (about 128 atoms, uncapped; UMA reference):

| geometry | UMA | PaiNN ep188 | PaiNN ep197 | MACE-OFF23 med | MACE ft-med | MACE ft-large |
|---|---|---|---|---|---|---|
| central_isoside | −30.62 | −14.94 | −11.99 | −26.21 | −31.41 | −38.42 |
| central_cf3side | −36.05 | −17.62 | −14.68 | −31.28 | −32.87 | −42.06 |
| iso_ring | −36.73 | −19.20 | −15.33 | −30.69 | −34.05 | −46.28 |
| cf3_ring | −38.17 | −20.20 | −16.02 | −33.52 | −36.08 | −44.59 |
| outlier gap | 5.43 | 2.67 | 2.69 | 5.06 | 1.46 | 3.64 |
| ordering matches UMA | – | yes | yes | no | yes | no |

### Interpretation

- **The from-scratch model underbinds systematically.** The epoch-188
  model recovers roughly 40% of the GFN2 stacking interaction energy it was
  trained toward (MAE 6.6 kcal/mol vs GFN2), and `weak_stopper` comes out
  repulsive. On the whole structures its ratio to UMA is strikingly constant,
  0.49–0.53 across all four geometries.
- **The binding scale is essentially unconstrained by the training data.**
  The epoch-188 and epoch-197 models fit the held-out rotaxane frames equally
  well (60.7 vs 59.8 meV/Å, 0.50 vs 0.53 meV/atom). Yet their interaction
  energies differ by about 20%: the whole-structure ratio to UMA drops from
  about 0.5 to about 0.4, and the stacking MAE vs GFN2 rises from 6.6 to
  7.8 kcal/mol. Two models that agree on everything they were trained on
  disagree about the binding energy. That is the signature of extrapolation:
  the fit says nothing about separated fragments. The OOD check below
  confirms that the isolated rod and wheel lie outside the training
  distribution.
- **Why.** An interaction energy needs the isolated rod and wheel, and the
  training set contains only the assembled rotaxane. The 5 Å cutoff adds to
  this: the dispersion attraction between wheel and rod beyond 5 Å can only
  be captured indirectly through the 15 Å message-passing range (see "What
  these models cannot capture").
- **Ordering survives.** Both PaiNN models reproduce the UMA ranking of the
  four whole-structure geometries, as does fine-tuned MACE medium, while stock
  MACE-OFF23 medium and fine-tuned MACE large do not. The central/iso-side
  outlier gap is compressed from 5.4 to 2.7 kcal/mol, consistent with the
  overall underbinding. The model gets the *relative* strength of the
  contacts but not the absolute scale.
- **Pretraining matters for this test.** The fine-tuned MACE models inherit
  a physically sensible description of separated molecules and non-covalent
  interactions from their foundation model. With 225 frames of one assembled
  rotaxane, a model trained from scratch cannot learn that.

## Out-of-distribution check

`code/ood_spk.py` applies the latent-distance signal from the `../mace`
project (`activation_ood.py`, `OOD_NOTES.md`) in reverse. There, the
reference distribution was MACE-OFF23's broad small-molecule training data and
rotaxanes were the novel chemistry. Here the model has only ever seen one
rotaxane, so ordinary molecules are the out-of-distribution case.

**Method.** (Run 1 scores below use the final layer, `mix3`. After the layer
sweep, the default is now `mix1`; see Run 2.) Each atom's final PaiNN scalar
feature (`scalar_representation`, the 128-d vector the energy readout acts
on) is unit-normalized. It is scored
by cosine distance to the nearest same-element atom in a pool built from the
225 training frames (32,400 atoms). A structure's score is the mean over its
atoms. Atoms of an element absent from training (S, Cl, …) score 1.0. The
in-distribution scale is set by the 25 held-out rotaxane frames: the
threshold is their maximum score.

```sh
../.venv/bin/python ood_spk.py ../output/rot250_painn/best_model   # -> ood_scores.csv, ood_pool.npz
```

The test sets (`data/ood/`, plus the benchmark geometries and `rot1_gfn2`)
run from the same molecule to foreign chemistry. S66 is parsed from the Psi4
database module. The OFF23 sample is 300 frames drawn at random from the
MACE-OFF23 test split. It is kept local (gitignored) because its
redistribution terms have not been checked.

Scores for the epoch-188 model (`ood_ep200.txt`; the epoch-197 model is within
about 10% everywhere, `ood.txt`):

| set | n | median score | × threshold | above threshold |
|---|---|---|---|---|
| rot250 validation (held out) | 25 | 0.0017 | – (threshold 0.0020) | – |
| rot1 GFN2 MD / normal modes (same rotaxane) | 221 | 0.0035 | 1.8 | 98% |
| benchmark whole: dimer / rod / wheel | 4 each | 0.0051 / 0.0048 / 0.0042 | 2.6 / 2.4 / 2.1 | 100% |
| dethread1 (112-atom dethreading system) | 10 | 0.0043 | 2.2 | 100% |
| benchmark stacking: dimer / rod / wheel | 7 each | 0.010 / 0.009 / 0.012 | 5–6 | 100% |
| S66 monomers / dimers | 132 / 66 | 0.013 / 0.014 | 6.5–7 | 100% |
| OFF23 test sample | 300 | 0.042 | 21 | 100% (147 with unseen elements) |
| rosuvastatin | 1 | 0.037 | 19 | contains S |

- **The ranking is physically sensible.** Held-out frames score lowest.
  Next come the same rotaxane under different sampling (500 K MD and
  normal-mode displacements), then the whole-structure fragments and a
  related interlocked system. The capped stacking fragments are further out,
  then ordinary small molecules, and molecules with unseen elements are at
  the top.
- **The rod and wheel monomers the benchmark relies on are out of
  distribution** (2–2.4× the threshold; the capped fragments 5–6×). This is
  the direct evidence behind the underbinding interpretation above.
- **As a pass/fail flag, the threshold is too tight.** 98% of rot1 frames,
  the *same molecule*, exceed it: 25 frames from one sampling run define a
  very narrow in-distribution region. Use scores as ratios to the threshold,
  not as a binary flag.
- **The absolute scale is small** (0.002–0.06, compared with 0.04–0.24 for
  MACE's signal). The final PaiNN features of each element share a large
  common component, so cosine distances are compressed. The layer sweep below
  tests centering as a remedy.

### Layer sweep

As in the MACE study (`layer_sweep.py` there), every per-atom layer was
scored as a candidate signal (`code/layer_sweep_spk.py`, epoch-188 model,
`layer_sweep_ep200.txt`, `layer_sweep.json`). The candidates are the element
embedding; the scalar features after the message (`int`) and update (`mix`)
step of each block; the per-channel norms of PaiNN's vector features after
each block (`vnorm`, rotation-invariant); the 64-d readout hidden layer; and
a per-atom energy z-score (`eps_z`, the analogue of MACE's `ezMean`). Each
vector layer was also scored after subtracting the per-element mean
(`-c`). Two tests:

- **Separation:** median score of each set ÷ held-out maximum, and AUROC of
  held-out vs set. OFF23 is restricted to the 153 frames whose elements were
  all in training.
- **Error prediction:** Spearman ρ between the score and the model's actual
  per-frame force MAE / |energy error| on the 221 GFN2-labelled rot1 frames.
  Those errors range from 73 to 488 meV/Å and 1.6 to 17 meV/atom.

| layer | ratio rot1 | ratio rod | ratio wheel | ratio S66 | ratio OFF23 | AUROC rod / wheel / S66 | ρ force | ρ energy |
|---|---|---|---|---|---|---|---|---|
| emb | 1.0 | 0.7 | 0.9 | 2.1 | 1.6 | 0.00 / 0.00 / 0.85 | 0.04 | 0.06 |
| int1 | 1.4 | 4.4 | 3.5 | **14.8** | **15.5** | 1.00 / 1.00 / 1.00 | 0.74 | 0.83 |
| mix1 | 1.5 | 4.2 | 3.2 | 13.2 | 14.0 | 1.00 / 1.00 / 1.00 | 0.76 | 0.84 |
| vnorm1 | 1.5 | 2.8 | 2.4 | 8.1 | 8.4 | 1.00 / 1.00 / 1.00 | 0.77 | **0.86** |
| int2 | 1.4 | **4.7** | 3.3 | 12.4 | 13.8 | 1.00 / 1.00 / 1.00 | 0.75 | 0.84 |
| mix2 | 1.6 | 3.6 | 2.8 | 10.0 | 11.4 | 1.00 / 1.00 / 1.00 | **0.79** | 0.81 |
| int3 | 1.6 | 3.8 | 2.9 | 10.1 | 11.7 | 1.00 / 1.00 / 1.00 | 0.78 | 0.82 |
| mix3 (used by `ood_spk.py`) | 1.8 | 2.4 | 2.1 | 6.5 | 9.1 | 1.00 / 1.00 / 1.00 | 0.78 | 0.84 |
| vnorm3 | 1.4 | 1.8 | 1.9 | 4.4 | 4.2 | 1.00 / 1.00 / 1.00 | 0.76 | 0.85 |
| readout | 1.9 | 2.5 | 1.9 | 7.1 | 18.7 | 1.00 / 1.00 / 1.00 | 0.70 | 0.65 |
| mix3-c (centered) | 1.3 | 1.5 | 2.1 | 3.4 | 3.7 | 1.00 / 1.00 / 1.00 | 0.74 | 0.84 |
| eps_z | 1.2 | 1.4 | 1.0 | 2.0 | 3.2 | 1.00 / 0.75 / 0.93 | 0.65 | 0.72 |

(Selected rows; the full table, including all centered variants, is in `layer_sweep_ep200.txt`.)

- **The embedding is degenerate,** as in MACE: it is a function of the
  element alone. Every other layer separates the held-out frames from the
  rod/wheel monomers, S66 and OFF23 perfectly (AUROC 1.00).
- **Depth runs the opposite way to MACE.** In MACE the signal sharpened
  with depth. Here the *first* block gives the widest dynamic range (int1:
  S66 and OFF23 at about 15× the threshold), and it shrinks with depth
  (mix3: 6.5–9×). A plausible reason: the first block encodes the local
  chemical environment most directly. Deeper blocks are shaped by
  regression toward one molecule's energies and compress everything not in
  it.
- **Centering does not help separation.** It raises the absolute scores
  (held-out maximum 0.002 → 0.078 for mix3), which makes them easier to
  read. But it raises the held-out spread as much as the OOD scores, so all
  ratios fall.
- **Unlike MACE, the score tracks the model's own errors within the
  rotaxane.** Spearman ρ is 0.74–0.79 for forces and 0.81–0.86 for energies
  across all non-degenerate layers. In the MACE study no signal predicted
  error. The caveat is that this is within one molecule, where frames further
  from the training sampling are both more unusual and harder. It shows the
  score is a useful "how far from training" measure here, not that it would
  predict errors across chemistries.
- **Recommendation:** mix3 is adequate, but a first-block layer (int1 or
  mix1) gives about twice the separation at the same error correlation.
  vnorm1 has the highest energy-error correlation (0.86).

## Run 2: rotaxane + rod and wheel monomers

Run 1's benchmark and OOD check both pointed to the same gap: the model never
saw a separated rod or wheel. Run 2 adds them.

**Monomer data.** `code/make_monomers.py` splits every rot250 frame into its
two covalent components. The rod and wheel of a rotaxane are mechanically
interlocked but not bonded, so the covalent graph (ASE natural cutoffs) has
exactly two connected components. Every monomer keeps the geometry it has
inside the rotaxane, which is exactly what $E_\text{int} = E_\text{rotaxane} -
E_\text{rod} - E_\text{wheel}$ needs. The script checks that all 250 frames
give the same two molecules (RDKit bond perception from the 3D coordinates)
and draws them (`output/monomers.png`, `output/monomers_explicit_H.png`):

- **wheel:** 24-crown-8, C16H32O8 (56 atoms)
- **rod:** C44H26F14N2O2 (88 atoms), a neutral bis-amide axle. A
  para-quaterphenyl core carries one aryl F on each terminal ring, linked by
  amides to 3,5-bis(trifluoromethyl)benzyl stoppers:
  `O=C(NCc1cc(C(F)(F)F)cc(C(F)(F)F)c1)c1ccc(-c2ccc(-c3ccc(-c4ccc(C(=O)NCc5cc(C(F)(F)F)cc(C(F)(F)F)c5)c(F)c4)cc3)cc2)cc1F`

The 500 monomers were labelled with `gfn2_label.py` in 80 s. The split
follows the rotaxane split (monomers of the 25 held-out frames are held out),
so nothing leaks from training into validation. Files:
`data/rot250_gfn2_{train,valid}_{rod,wheel}.xyz` (unlabelled) and
`data/rot250_{rod,wheel}_gfn2_{train,valid}.xyz` (labelled).

Because the rotaxane and its monomers are labelled at identical geometries,
the data also gives a GFN2 **in-distribution interaction energy** for each
frame: −23.7 ± 6.3 kcal/mol on the training frames and **−22.2 ± 4.4
kcal/mol** on the 25 held-out frames.

**Training** (`examples/run_rot250mono_painn.sh`): same PaiNN-128
architecture, from scratch. Trained on 675 frames (225 rotaxanes + 225 rods +
225 wheels) and validated on 75. 300 epochs, CPU, batch 4, Adam at 5e-4
(halved once by the plateau scheduler, to 2.5e-4), **gradient-norm clipping
at 10** (`--grad-clip`). There were no loss spikes, unlike run 1's divergence.
About 47 s per epoch, 4.4 h wall time (including about 36 min of sleep
pauses, see the note below), peak 1.95 GB. Best epoch 294.

### Fit to GFN2 (held-out frames)

| model | rotaxane energy | rotaxane forces | rod forces | wheel forces |
|---|---|---|---|---|
| run 1, rotaxane only (epoch 188) | 0.50 meV/atom | 60.7 meV/Å | – | – |
| **run 2, + monomers (epoch 294)** | **0.29 meV/atom** | **26.9 meV/Å** | 21.3 meV/Å | 19.5 meV/Å |
| MACE-OFF23 medium, fine-tuned | 0.5 meV/atom | 30.6 meV/Å | – | – |
| MACE-OFF23 large, fine-tuned | 0.6 meV/atom | 25.7 meV/Å | – | – |

Run 2 halves the rotaxane force error and **matches the fine-tuned MACE
models**, with lower energy error. Several changes landed at once: the
monomer frames (more and simpler environments of the same chemistry),
gradient clipping, and 100 more epochs of stable training. This run does not
separate their effects.

### In-distribution interaction energy (25 held-out frames)

| | $E_\text{int}$ (kcal/mol) | MAE vs GFN2 | mean error | correlation with GFN2 |
|---|---|---|---|---|
| GFN2-xTB | −22.20 ± 4.44 | – | – | – |
| run 1, rotaxane only | −13.77 ± 4.56 | 8.42 | +8.42 | 0.95 |
| **run 2, + monomers** | **−22.69 ± 4.73** | **1.04** | **−0.49** | **0.97** |

This is the cleanest test of the underbinding hypothesis, and it confirms it.
Run 1 already ranked the frames correctly (r = 0.95) but recovered only about
62% of the binding: the "relative right, absolute wrong" pattern seen on the
benchmark. Trained with the monomers, the model reproduces GFN2's
interaction energies to 1 kcal/mol with essentially no bias.

### Rotaxane benchmark

The benchmark systems are **different molecules** from the training
rotaxane. The whole-structure wheel is C24H32O8, consistent with
dibenzo-24-crown-8, and the rod is C28H22F6N2O6. The stacking set uses small
capped fragments. This is therefore a transfer test.

| | run 1 | run 2 | MACE ft-medium | MACE ft-large |
|---|---|---|---|---|
| stacking MAE vs GFN2 (kcal/mol) | 6.55 | 6.65 | 0.72 | 4.27 |
| stacking MAE vs CCSD(T) | 5.04 | 4.97 | 1.12 | 4.67 |
| whole structures, ratio to UMA | 0.49–0.53 | **0.69–0.90** | 0.91–1.03 | 1.17–1.26 |
| whole structures, ordering matches UMA | yes | no | yes | no |
| whole structures, outlier gap (UMA 5.43) | 2.67 | 0.94 | 1.46 | 3.64 |

Whole-structure values for run 2: central_isoside −27.47, central_cf3side
−28.41, iso_ring −30.91, cf3_ring −26.50 (UMA −30.62, −36.05, −36.73,
−38.17). Full tables: `output/rot250mono_painn/bench.txt`.

- **The large, rotaxane-like systems improve a lot.** Binding on the whole
  structures goes from about half of UMA to 69–90%. Learning what separated
  components look like transfers to a related crown-ether/amide rotaxane.
  However, the ranking of the four geometries is now wrong: cf3_ring comes
  out weakest instead of strongest, and the outlier gap collapses.
- **The small capped stacking fragments do not improve** (MAE vs GFN2
  unchanged at about 6.6 kcal/mol). They are far from anything in the
  training data (5–8× the OOD threshold, below), and no amount of data on one
  rotaxane teaches the model small capped aromatic dimers.

### OOD check with the monomers in the reference pool

`ood_spk.py` now scores the `mix1` layer by default (first-block update,
following the layer sweep), with `--layer` to choose another. Median score
÷ held-out maximum, from `ood_scores_mix1.csv` (run 1) and
`output/rot250mono_painn/ood_scores.csv` (run 2):

| set | run 1 (pool: rotaxane) | run 2 (pool: rotaxane + rod + wheel) |
|---|---|---|
| rot1, same rotaxane, other sampling | 1.5× | 1.7× |
| benchmark whole: rod / wheel | 4.2× / 3.2× | 3.1× / 3.0× |
| benchmark whole: dimer | 4.4× | 3.7× |
| dethread1 | 3.4× | 3.3× |
| benchmark stacking: rod / wheel | 7.6× / 11.0× | 5.0× / 8.5× |
| S66 monomers | 13.2× | 12.1× |
| OFF23 sample | 37.8× | 21.2× |

The benchmark components move closer to the training distribution, most for
the stacking fragments (by 23–34%), but all stay well outside it. The
largest drop is OFF23 (38× → 21×). The ordering of the sets is unchanged. As
before, 98–99% of the rot1 frames exceed the threshold that 25 held-out
frames set, so the ratios are the useful quantity.

### Note on the pauses during this run

Epochs 58, 135 and 212 took 495, 1014 and 791 s instead of about 47 s. The
power log (`pmset -g log`) shows the cause. The previous job's `caffeinate`
exited with that job at 18:01, and with nothing holding it the Mac
idle-slept at 18:08. From then on it ran only in Power Nap DarkWakes, which
macOS ends with a "Maintenance Sleep" about every hour whatever `caffeinate`
asserts. `caffeinate -u` (declare user activity) brought it back to FullWake
at 22:33, and there were no pauses after that. Both runners now end with a
30-minute `sleep 1800` grace period, as in the Rotaxanes runners, so
`caffeinate` outlives the job.

## Next steps

- Train on monomers of more rotaxanes, and on partially dethreaded
  geometries (`dethread1`), so interaction energies transfer beyond this one
  system. The whole-structure ordering regressed in run 2.
- Add small capped fragments (or S66-like dimers) labelled with GFN2 if the
  stacking benchmark matters. They are far outside the current data.
- Separate the effects of the monomer data and gradient clipping (a
  rotaxane-only run with clipping).
- Try a larger cutoff (6–7 Å) or an explicit long-range term to capture more
  of the dispersion tail.
