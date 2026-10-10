#!/bin/sh
# Download the public Afrobarometer Round 8 and Round 9 merged SPSS releases
# the afro_r8 / afro_r9 labs are built from, and check them against the hashes
# of the files the labs were produced with. Afrobarometer can replace these
# files; a hash mismatch means the lab is not the same.
# The downloads are untrusted data: read them only with pyreadstat
# (tools/proxy_afro.py, run with python -I).
#   tools/fetch_afro.sh [dest]        (default data/raw/afro)
set -e
DEST=${1:-data/raw/afro}
BASE=https://www.afrobarometer.org/wp-content/uploads
mkdir -p "$DEST"
cd "$DEST"
curl -sSL -o afro_r8.sav "$BASE/2023/03/afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav"
curl -sSL -o afro_r9.sav "$BASE/2025/06/R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav"
sha256sum afro_r8.sav afro_r9.sav
sha256sum -c - <<SUMS
e5ae40e9f7e8ead6b998bd77a2e98272ce3dc65b54481ca085c438f463ac664c  afro_r8.sav
3afcecd08d2f4b531a926d455a943084d9f71c1276b4f3a58e77a9162e6879d8  afro_r9.sav
SUMS
