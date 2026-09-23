#!/bin/zsh
# Train PaiNN on the rot250 GFN2-xTB set (225 train / 25 valid frames, 144-atom
# rotaxane) and score it on the rotaxane interaction-energy benchmark.
#
# Launch fully detached (survives the Claude session; see global CLAUDE.md):
#   ~/miniforge3/bin/python -c "import subprocess; subprocess.Popen(['zsh','/Users/cafierom/python_mac/schnet/examples/run_rot250_painn.sh'], stdout=open('/Users/cafierom/python_mac/schnet/output/rot250_painn.log','ab'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)"
# Monitor: tail -f output/rot250_painn.log   (per-epoch metrics: output/rot250_painn/lightning/metrics.csv)
set -e
REPO=/Users/cafierom/python_mac/schnet
PY=$REPO/.venv/bin/python
NAME=${NAME:-rot250_painn}
DEVICE=${DEVICE:-cpu}
EPOCHS=${EPOCHS:-500}
RESUME=${RESUME:-0}            # RESUME=1: continue this run from last.ckpt up to EPOCHS (total)

caffeinate -i -w $$ &          # hold off idle sleep for the whole run
export KMP_DUPLICATE_LIB_OK=TRUE
cd $REPO/code
echo "== start $(date)  name=$NAME device=$DEVICE epochs=$EPOCHS resume=$RESUME"

# training is skipped if this run already finished (metrics.json present), unless resuming
if [[ $RESUME == 1 ]]; then
  $PY train_spk.py --train ../data/rot250_gfn2_train.xyz --valid ../data/rot250_gfn2_valid.xyz \
      --name $NAME --model painn --device $DEVICE --epochs $EPOCHS --resume
elif [[ ! -f $REPO/output/$NAME/metrics.json ]]; then
  $PY train_spk.py --train ../data/rot250_gfn2_train.xyz --valid ../data/rot250_gfn2_valid.xyz \
      --name $NAME --model painn --device $DEVICE --epochs $EPOCHS
fi

$PY rotaxane_bench.py --spk-model $NAME=$REPO/output/$NAME/best_model \
    | tee $REPO/output/$NAME/bench.txt
$PY ood_spk.py $REPO/output/$NAME/best_model | tee $REPO/output/$NAME/ood.txt
echo "== done $(date)"

# Grace period (as in the Rotaxanes runners): keep caffeinate alive past
# completion so the machine does not idle-sleep the moment the work ends.
sleep 1800
