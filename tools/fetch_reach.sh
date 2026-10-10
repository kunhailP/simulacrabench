#!/bin/sh
# Download the REACH Iraq CCCM IDP Camp Profiling VII household dataset
# (Dec 2016 - Jan 2017, HDX, CC BY-IGO) the reach_irq lab is built from and
# check it against the hash of the file the lab was produced with. HDX can
# replace the resource; a hash mismatch means the lab is not the same.
#   tools/fetch_reach.sh [dest]        (default data/raw/reach)
set -e
DEST=${1:-data/raw/reach}
URL=https://data.humdata.org/dataset/43690e22-b6c6-402a-86ca-e565b733bbbf/resource/bbb5892e-4a62-4b0e-90a6-26ff9fd3139a/download/reach_iraq_cccm_camp_profiling_vii_dataset_feb2017.xlsx
F=REACH_IRAQ_CCCM_Camp_Profiling_VII_Dataset_Feb2017.xlsx
mkdir -p "$DEST"
cd "$DEST"
curl -sSL -o "$F" "$URL"
sha256sum -c - <<SUMS
f05f5e0f585d3e1d7fa4b34839185bc3c8fb0331f67aa36682f7a5b685d89738  $F
SUMS
