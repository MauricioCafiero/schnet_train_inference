# Plan: a comprehensive multi-rotaxane training set (Run 3)

Status: **planned, not started.** Waiting on trajectory samples (see
[What to provide](#what-to-provide-trajectories)). Written 2026-09-23 after Run 2.

## Why

Run 2 (rotaxane + its rod and wheel monomers) fixed interaction energies *for
the training molecule*. On the 25 held-out rot1 frames it reproduces GFN2
E_int to 1.0 kcal/mol (Run 1: 8.4), and rotaxane forces reach 26.9 meV/Å,
matching fine-tuned MACE. Transfer to *other* rotaxanes is only partial. On
the rot3 whole structures, binding improved from about 0.5 to 0.69–0.90 of
UMA, but the ranking of the four geometries is wrong. The small capped
stacking fragments did not improve at all. The OOD check agrees: every
benchmark component is still 3–8× outside the training distribution.

Run 3 trains on several rotaxanes (different axles, both 24C8 and DB24C8
wheels) and on real dethreading geometries, so the model sees how binding
changes across chemistry and as the components separate.

## Sources in `rotaxanes-results` (already available)

Inventory of <https://github.com/MauricioCafiero/rotaxanes-results> (commit
049cca9). Full `.dcd` trajectories are not in the repo; these are the frame
dumps.

| system | wheel | atoms | frames in the repo | path |
|---|---|---|---|---|
| rot1 (= our rot250 molecule) | 24C8 | 144 | 300 shuttle-coordinate snapshots (d −11 … +11 Å, 3 replicas per station) | `qm_rescoring/frames/*.xyz` |
| rot2htpuma (short diamide axle) | 24C8 | 114 | 183 snapshots | `qm_rescoring/frames_rot2/*.xyz` |
| dethread1 (PFP-ester axle) | DB24C8 | 112 | 494 solute-only dethreading frames | `metad_dethread/checkin_frames2/{dethreading_event (250), trace_movie (138)}.pdb`, `metad_dethread/checkin_frames_cube/{dethreading_transition (61), new_region_frames (45)}.pdb` |
| dethread2 / 3 / 4 | 24C8 / DB24C8 / DB24C8 | ~110–120 | start structures only (solvated `complex.pdb`, seeds) | `outputs_dethread/dethread*_cube/`, `*_seed.xyz` |
| rot3 (ester axle, the benchmark whole-structure system) | DB24C8 | 128 | 8 whole structures, ~17 solvated umbrella windows | `ROT3QM/uma_whole_structure/`, `metad_run_rot3/umbrella_starts/w*.pdb` |
| capped stacking fragments (the stacking benchmark) | – | 19–39 | 7 pairs | `QM_comparisons/fragments/` |

Rod and wheel SMILES for each system are in `<name>.txt` at the root of
`rotaxanes-results`, and for rot3 in `Rotaxanes/rot3.txt`.

To check when building:
- Which dethread system the `checkin_frames*` dumps belong to. The folder
  names and 112 atoms say dethread1; confirm by F count against the SMILES.
- Whether the rot1 shuttle snapshots duplicate geometries already in rot250.

## What to provide (trajectories)

Priority order:

1. **dethread2, dethread3, dethread4 MetaD runs.** We currently have only
   start structures for these. They bring the 24C8 and fluorinated-rod
   dethreading variants, including partly threaded and separated states.
2. **dethread1 full run.** The repo dumps are concentrated around the
   dethreading event.
3. **rot2htpuma MetaD.** Broader than the 183 shuttle snapshots.
4. **rot1 MetaD.** Fills shuttle positions between stations (lowest priority:
   550 rot1 frames already).
5. **rot3 MetaD.** Only if we decide to train on rot3 (see Decisions).

**Format:** the solute-only `pymol_rotaxane.pdb` + `pymol_rotaxane.dcd` pair
written by `analyze_metad.py` (rod-centred, no water), **plus the `COLVAR`**
for each run. Solvated `.dcd` + topology also works, since the solvent is
stripped anyway. Drop them in `data/traj/<system>/` in this repo, or give the
path. They are read only.

## Dataset build

1. **Frame selection.** About 250–300 frames per system, **stratified along
   the CV** from COLVAR (shuttle or dethreading coordinate), so rare states
   (transitions, separated geometries) are not swamped by the wells. Use a
   minimum time spacing to avoid near-duplicates. Record the source run,
   frame index and CV value of every frame in `atoms.info`.
2. **Monomers.** Run `code/make_monomers.py` on every complex frame to get
   the rod and wheel at the in-complex geometry. It asserts two covalent
   components and checks the topology against the SMILES.
3. **Labels.** Everything goes through GFN2-xTB with `code/gfn2_label.py`,
   locally: 4 workers, estimated **10–15 min** for about 4–5k frames (the 500
   Run-2 monomers took 80 s). One level of theory throughout.
4. **Screening.** The trajectories come from the classical force field, so a
   few frames may be strained at GFN2. Drop frames with a GFN2 force outlier
   (max |F| far above the system's distribution) and report how many were
   dropped per system.
5. **Split.** Per system, by **contiguous blocks** of the trajectory or CV,
   not at random (consecutive MD/MetaD frames are strongly correlated). About
   10% validation, and monomers follow their complex's split.
6. **Size.** About 1.2–1.5k complexes + 2.4–3k monomers ≈ **4–5k frames**,
   roughly 4× Run 2's atoms per epoch.

New code: `code/build_dataset.py`: read `.dcd`/multi-model PDB/XYZ (adds
`mdtraj` or `MDAnalysis` to the venv), strip solvent, CV-stratified selection,
block splits, write unlabelled XYZ + provenance, then call `make_monomers.py`.

## Training on Modal (~$3)

On the local CPU this set would take about 3 min per epoch (about 15 h for
300 epochs), so train on a Modal GPU.

- **`code/modal_train_spk.py`**, modelled on `../mace/code/modal_finetune.py`
  (image with pip installs, `add_local_file` for code and data, a Volume for
  outputs, a `gpu` argument; the MACE runs used an A10G). It wraps
  `train_spk.py`: PaiNN-128, 3 blocks, 5 Å cutoff, `--grad-clip 10`,
  float32, batch ~16 on the GPU.
- **Budget guards:**
  1. A **2-epoch timing probe** first (a few cents) to measure s/epoch.
  2. The Modal `timeout` set to the budget (≈ 2.5 h on an A10G) as a **hard
     cost cap**.
  3. Checkpoints to the Volume every epoch, so a timeout loses nothing and
     `--resume` continues.
  4. If the probe says 300 epochs won't fit, drop to ~200 epochs or raise the
     batch size.
- **Cost guide:** approximately A10G $1.10/h, L4 $0.80/h. *Check Modal's
  current pricing before running.* $3 ≈ 2.7 h A10G or 3.7 h L4. 300 epochs
  fits if an epoch takes ≤ 30–45 s on the GPU (a guess until the probe).

## Memory on Modal

Locally the constraint was an 8 GB Mac shared with other work. On Modal it is
a 24 GB GPU (A10G or L4) billed by the second, so the aim is to use memory
efficiently without paying for idle GPU time.

**No longer needed**

- **Micro-batching** (`--micro-batch`). It existed because Apple's MPS
  allocator peaked at about 0.85 GB per 144-atom frame. On CUDA, raise the
  batch size instead. Guide: CPU measured about 0.3 GB/frame, so batch 16 ≈
  5 GB; there is plenty of headroom even if CUDA peaks are 2–3× that. The probe
  measures the real value.
- **The free-memory watchdog and `caffeinate`.** They protected the laptop.
  On Modal the guards are the function `timeout` (cost cap) and a clean CUDA
  out-of-memory error, which fails fast instead of swapping.
- **The MPS float64 workaround** (explicit float32 `AddOffsets` mean). It is
  harmless on CUDA, so leave it in.

**Changes to make**

1. **CUDA memory logging.** Extend `MemoryLog` in `train_spk.py` to log
   `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` (it
   currently reports process RSS + MPS driver memory).
2. **Worst-case probe.** Frames range from about 20 to 144 atoms, so memory
   varies per batch. Before training, run one forward + backward pass (with
   forces) on a batch made only of the largest frames at the chosen batch
   size. If it fits, every batch fits. Seconds of GPU time.
3. **Batch size** (open decision 5 below). Batch 16 uses the GPU well and is
   cheaper per epoch, but gives 4× fewer optimizer steps per epoch than Runs
   1–2 (batch 4) and changes the training dynamics. Keep lr 5e-4 (or raise it
   modestly) with gradient clipping at 10.
4. **Allocator:** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to
   limit fragmentation with variable batch sizes. That is the CUDA counterpart
   of the over-reservation measured on MPS.
5. **Precision: float32.** Not bf16/fp16: the second backward pass for forces
   is sensitive, and the `../mace` notes record low-precision failures. TF32
   matmuls on Ampere+ GPUs are fine.
6. **Container:** modest host RAM (~8 GB) and 4 CPUs, with 2–4 DataLoader
   workers (`num_workers`) so the GPU is not waiting on neighbor lists. The
   data is only a few MB of XYZ plus a small ASE db.
7. **Neighbor-list cache on local container disk** (`/tmp`), not the Volume
   (Volume writes are slower). It rebuilds in the first epoch.
8. **Checkpoint every epoch.** SchNetPack's `ModelCheckpoint` only refreshes
   `last.ckpt` when validation improves (why the Run-1 resume restarted from
   epoch 188). Add a plain Lightning checkpoint that writes `last.ckpt` to the
   Volume **every epoch** (~7 MB), so a timeout loses at most one epoch.

**Locally:** only the GFN2 labelling runs on the Mac (~10–15 min, ~1 GB, the
existing 4-worker setup), plus a few-MB upload to Modal.

## Evaluation (same as Runs 1–2, per system)

- Held-out fit per molecule (energy meV/atom, forces meV/Å); per-formula
  metrics are already in `metrics.json`.
- **In-distribution E_int** on each system's validation frames against GFN2
  (as in Run 2: MAE, bias, correlation).
- `code/rotaxane_bench.py`: stacking (CCSD(T)/GFN2) and whole structures
  (UMA). Main questions: does the whole-structure **ranking** recover, and do
  the stacking fragments move at all?
- `code/ood_spk.py` (mix1 layer, pool = all training frames): are the
  benchmark components closer to the training distribution than in Run 2?
- Compare against Run 2 and the fine-tuned MACE models in the README.

## Decisions (open)

1. **rot3: hold out (recommended) or train on it?** It is the benchmark's
   whole-structure system. Holding it out keeps an honest transfer test. The
   dethread1/3/4 wheel (DB24C8) is the same as rot3's, so the new data should
   help transfer without leaking the answer. Alternative: train on rot3
   umbrella/MetaD frames and test only on the ROT3QM whole structures.
2. **dethread2–4 without trajectories:** if trajectories are not available,
   generate about 100 short GFN2-MD frames each from the start structures with
   `code/gfn2_data.py` (about 30–60 min of local GFN2).
3. **GPU:** A10G (used for the MACE runs) or L4 (cheaper per hour, probably
   slower per epoch). The timing probe can decide.
4. **Small fragments:** add GFN2-labelled capped fragments or S66-like dimers
   if the stacking benchmark matters. They are far outside everything else.
5. **Batch size on the GPU:** 16 (recommended: cheaper, better GPU use; note
   the change in the README) or 4 (directly comparable with Runs 1–2 but
   underuses the GPU and costs more per epoch).

## Checklist when resuming

- [ ] Trajectories + COLVARs in `data/traj/<system>/`
- [ ] Decide rot3 / dethread2–4 / GPU (above)
- [ ] `build_dataset.py` → unlabelled frames + monomers + provenance
- [ ] Topology check vs SMILES for every system
- [ ] GFN2 labels (local) → force-outlier screen → block splits
- [ ] `train_spk.py`: CUDA memory logging, per-epoch `last.ckpt`, DataLoader workers
- [ ] `modal_train_spk.py` → worst-case memory probe → 2-epoch timing probe → size the run to ≤ $3
- [ ] Full run (timeout cap, per-epoch checkpoints)
- [ ] Evaluation (fit, E_int per system, benchmark, OOD) → README → push
