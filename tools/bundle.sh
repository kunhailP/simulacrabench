#!/bin/bash
# usage: tools/bundle.sh sub/v2 [instruments]   -> headroom table over all scenario worlds
SUB=$1; INST=${2:-unicef,world_bank,unhcr}
cd "$(dirname "$0")/.."
for sc in base weakx strongx cells cleangate ordinal ordweak; do
  .venv/bin/python tools/evaluate.py $SUB --data data/worlds/$sc --instruments $INST 2>&1 \
   | grep -E 'ORACLE|skill' | awk -v sc=$sc '{print sc"\t"$0}' | cut -c1-110
done
