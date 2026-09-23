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

caffeinate -i -w $$ &          # hold off idle sleep for the whole run
export KMP_DUPLICATE_LIB_OK=TRUE
cd $REPO/code
echo "== start $(date)  name=$NAME device=$DEVICE epochs=$EPOCHS"

# training is skipped if this run already finished (metrics.json present)
if [[ ! -f $REPO/output/$NAME/metrics.json ]]; then
  $PY train_spk.py --train ../data/rot250_gfn2_train.xyz --valid ../data/rot250_gfn2_valid.xyz \
      --name $NAME --model painn --device $DEVICE --epochs $EPOCHS
fi

$PY rotaxane_bench.py --spk-model $NAME=$REPO/output/$NAME/best_model \
    | tee $REPO/output/$NAME/bench.txt
echo "== done $(date)"
