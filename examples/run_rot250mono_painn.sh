#!/bin/zsh
# rot250 + its rod and wheel monomers: GFN2-label the monomers, train PaiNN
# from scratch on rotaxane + rod + wheel, then run the benchmark and the OOD
# check (mix1 layer; the reference pool includes the monomers).
#
# Monomers come from code/make_monomers.py (rotaxane frames split into their
# two covalent components, geometries as in the rotaxane).
#
# Launch detached (see run_rot250_painn.sh for the pattern):
#   ~/miniforge3/bin/python -c "import subprocess; subprocess.Popen(['zsh','/Users/cafierom/python_mac/schnet/examples/run_rot250mono_painn.sh'], stdout=open('/Users/cafierom/python_mac/schnet/output/rot250mono_painn.log','ab'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)"
set -e
REPO=/Users/cafierom/python_mac/schnet
PY=$REPO/.venv/bin/python
D=$REPO/data
NAME=${NAME:-rot250mono_painn}
EPOCHS=${EPOCHS:-300}

caffeinate -i -w $$ &
export KMP_DUPLICATE_LIB_OK=TRUE
cd $REPO/code
echo "== start $(date)  name=$NAME epochs=$EPOCHS"

# 0. old rotaxane-only model, OOD rescored at the new default layer (mix1)
if [[ ! -f $REPO/output/rot250_painn/ood_mix1.txt ]]; then
  $PY ood_spk.py $REPO/output/rot250_painn/best_model_ep200 > $REPO/output/rot250_painn/ood_mix1.txt
  cp $REPO/output/rot250_painn/ood_scores.csv $REPO/output/rot250_painn/ood_scores_mix1.csv
fi

# 1. GFN2 labels for the monomers (skipped if present)
for part in rod wheel; do
  for split in train valid; do
    out=$D/rot250_${part}_gfn2_${split}.xyz
    [[ -f $out ]] || $PY gfn2_label.py $D/rot250_gfn2_${split}_${part}.xyz $out --workers 4
  done
done
echo "== labels done $(date)"

# 2. train (skipped if finished)
if [[ ! -f $REPO/output/$NAME/metrics.json ]]; then
  $PY train_spk.py --name $NAME --model painn --device cpu --epochs $EPOCHS --grad-clip 10 \
      --train $D/rot250_gfn2_train.xyz $D/rot250_rod_gfn2_train.xyz $D/rot250_wheel_gfn2_train.xyz \
      --valid $D/rot250_gfn2_valid.xyz $D/rot250_rod_gfn2_valid.xyz $D/rot250_wheel_gfn2_valid.xyz
fi

# 3. benchmark + OOD
$PY rotaxane_bench.py --spk-model $NAME=$REPO/output/$NAME/best_model | tee $REPO/output/$NAME/bench.txt
$PY ood_spk.py $REPO/output/$NAME/best_model \
    --train $D/rot250_gfn2_train.xyz $D/rot250_rod_gfn2_train.xyz $D/rot250_wheel_gfn2_train.xyz \
    | tee $REPO/output/$NAME/ood.txt
echo "== done $(date)"

# Grace period (as in the Rotaxanes runners): keep caffeinate alive past
# completion so the machine does not idle-sleep the moment the work ends.
sleep 1800
