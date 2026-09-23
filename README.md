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
(500 / cpu / rot250_painn). The run below used `EPOCHS=200`.

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
the same 25 held-out frames used for the MACE fine-tunes. 200 epochs on CPU,
batch 4, Adam at 5e-4, energy : force loss weights 0.05 : 0.95. Wall time
82 min (about 24.5 s per epoch). Peak process memory 2.1 GB. Everything is in
`output/rot250_painn/`.

### Fit to GFN2-xTB (held-out validation frames)

Numbers are from the float64 CPU evaluation of `best_model` (epoch 188) in
`metrics.json`.

| model | training | energy MAE | force MAE |
|---|---|---|---|
| **PaiNN (this repo)** | from scratch, 225 frames | **0.50 meV/atom** | **60.7 meV/Å** |
| MACE-OFF23 medium + replay | fine-tuned, same frames | 0.5 meV/atom | 30.6 meV/Å |
| MACE-OFF23 large + replay | fine-tuned, same frames | 0.6 meV/atom | 25.7 meV/Å |

Validation curve (per structure; the energy MAE is for the whole 144-atom
frame):

| epoch | energy MAE (eV) | force MAE (eV/Å) |
|---|---|---|
| 0 | 2.68 | 0.537 |
| 25 | 0.48 | 0.182 |
| 100 | 0.12 | 0.087 |
| 150 | 0.20 | 0.070 |
| 200 | 0.10 | 0.064 |

The energies match the fine-tuned MACE models. The forces are about 2× worse.
The run was not converged: validation loss was still falling at epoch 200,
the best epoch was 188, and the learning-rate scheduler never triggered (the
rate stayed at 5e-4 throughout). Longer training with learning-rate decay
should improve the forces further. How much of the gap to MACE would close is
open, because MACE starts from a foundation model pretrained on a large
organic dataset.

### Rotaxane interaction-energy benchmark

$E_\text{int} = E_\text{dimer} - E_\text{rod} - E_\text{wheel}$ in kcal/mol,
from `output/rot250_painn/bench.txt`. MACE columns come from the `../mace`
project (`data/mace_bench_results.json`).

**Stacking fragments** (capped wheel/rod pairs; DLPNO-CCSD(T)/aug-cc-pVTZ reference):

| pairing | CCSD(T) | DFT | GFN2 | PaiNN | MACE-OFF23 med | MACE ft-med | MACE ft-large |
|---|---|---|---|---|---|---|---|
| center_4F | −10.10 | −10.09 | −10.26 | −4.18 | −10.67 | −11.17 | −14.99 |
| center_2F | −9.63 | −9.30 | −9.90 | −4.45 | −9.77 | −10.71 | −14.87 |
| center_0F | −7.59 | −7.28 | −8.58 | −3.44 | −8.17 | −9.23 | −13.04 |
| weak_stopper_real | −8.09 | −8.23 | −7.84 | −3.19 | −8.09 | −8.78 | −11.19 |
| strong_stopper_real | – | −11.53 | −11.71 | −6.40 | −11.51 | −12.35 | −16.71 |
| strong_stopper | – | −11.52 | −12.33 | −4.00 | −10.71 | −13.20 | −16.69 |
| weak_stopper | – | −9.50 | −10.52 | +0.41 | −8.99 | −10.72 | −13.49 |
| **MAE vs CCSD(T)** | | | | **5.04** | 0.32 | 1.12 | 4.67 |
| **MAE vs GFN2** | | | | **6.55** | 0.65 | 0.72 | 4.27 |

**Whole-structure double stacks** (about 128 atoms, uncapped; UMA reference):

| geometry | UMA | PaiNN | MACE-OFF23 med | MACE ft-med | MACE ft-large |
|---|---|---|---|---|---|
| central_isoside | −30.62 | −14.94 | −26.21 | −31.41 | −38.42 |
| central_cf3side | −36.05 | −17.62 | −31.28 | −32.87 | −42.06 |
| iso_ring | −36.73 | −19.20 | −30.69 | −34.05 | −46.28 |
| cf3_ring | −38.17 | −20.20 | −33.52 | −36.08 | −44.59 |
| outlier gap | 5.43 | 2.67 | 5.06 | 1.46 | 3.64 |
| ordering matches UMA | – | yes | no | yes | no |

### Interpretation

- **The from-scratch model underbinds systematically, by about a factor of
  two.** On the stacking set it recovers roughly 40% of the GFN2 interaction
  energy it was trained toward (MAE 6.6 kcal/mol vs GFN2). One pairing,
  `weak_stopper`, comes out slightly repulsive (+0.41). On the whole
  structures the ratio to UMA is strikingly constant, 0.49–0.53 across all
  four geometries.
- **This is the extrapolation problem described under "What these models
  cannot capture".** An interaction energy needs the isolated rod and wheel,
  and the training set contains only the assembled rotaxane. The model has
  never seen a separated fragment, so it has no information about how the
  energy changes as the components come apart. The 5 Å cutoff adds to this:
  the dispersion attraction between wheel and rod beyond 5 Å can only be
  captured indirectly through the 15 Å message-passing range. The constant
  ratio suggests the model gets the *relative* strength of the contacts right
  but the absolute scale of the binding wrong.
- **Ordering is still right.** On the whole-structure set, PaiNN reproduces
  the UMA ranking of the four geometries (as does fine-tuned MACE medium,
  while stock MACE-OFF23 medium and fine-tuned MACE large do not). The
  central/iso-side outlier gap is compressed from 5.4 to 2.7 kcal/mol,
  consistent with the overall halving.
- **Pretraining matters for this test.** The fine-tuned MACE models inherit
  a physically sensible description of separated molecules and non-covalent
  interactions from their foundation model. With 225 frames of one assembled
  rotaxane, a model trained from scratch cannot learn that.

### Next steps

- Add separated rod and wheel conformers, and partially dethreaded
  geometries, to the training data. They can be labelled with
  `gfn2_label.py` / `gfn2_data.py`.
- Train longer with learning-rate decay (not converged at 200 epochs).
- Try a larger cutoff (6–7 Å) or an explicit long-range term to capture more
  of the dispersion tail.
