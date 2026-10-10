#!/bin/bash
# Run lab jobs "variant:lab" with at most $MAXJ on the GPU at once (scores only;
# timings from a shared GPU are meaningless). A run whose predict() fell back
# (e.g. CUDA OOM) is deleted and reported, so it can never be compared.
#   MAXJ=4 tools/par.sh v8:gss_wb v8e2:gss_wb ...
cd "$(dirname "$0")/.."
MAXJ=${MAXJ:-4}
mkdir -p results/par
run() {
  v=${1%%:*}; L=${1#*:}; log=results/par/${v}_$L.log
  V8_BUDGET=100000 .venv/bin/python tools/lab.py sub/$v --data data/proxy/$L --reps 1 --tag $v > $log 2>&1
  if grep -q '"fallback"\|retry_after\|Traceback' $log; then
    rm -f results/lab/$L/${v}_rep0.npz; echo "FAILED $v $L (fallback/error), result removed"
  else
    echo "ok $v $L: $(grep -o 'skill [0-9.]*' $log | tail -1)"
  fi
}
for job in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 5; done
  run "$job" &
done
wait
echo PAR DONE
