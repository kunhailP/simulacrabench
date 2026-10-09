#!/bin/sh
# Download the public GSS Stata releases the real-data lab is built from and
# check them against the hashes of the files the lab results were produced with.
# NORC can replace these files; a hash mismatch means the lab is not the same.
#   tools/fetch_gss.sh [dest]        (default data/raw/gss)
set -e
DEST=${1:-data/raw/gss}
BASE=https://gss.norc.org/content/dam/gss/get-the-data/documents/stata
mkdir -p "$DEST"
cd "$DEST"
for y in 2016 2018 2022; do
  curl -sSL -o gss$y.zip "$BASE/${y}_stata.zip"
done
sha256sum -c - <<SUMS
57224db478423b2fed063d8d525bd38931f710106cd69a20e3b21c4f885b918d  gss2016.zip
541199514cafbcedb9a4bab3e84fc81e61ebc48e6fac31274f653c893eaa9ad8  gss2018.zip
ba0bc3255ce59d5351b8e5382c81d9d29ad6275d61cf24f6f027b69ef2e97fd0  gss2022.zip
SUMS
for y in 2016 2018 2022; do mkdir -p gss$y && unzip -o -q gss$y.zip -d gss$y; done
