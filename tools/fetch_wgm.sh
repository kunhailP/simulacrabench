#!/bin/sh
# Download the public Wellcome Global Monitor 2018 workbook (crosstabs, full
# respondent-level dataset, data dictionary) the WGM lab is built from, and
# check it against the hash of the file the lab results were produced with.
# Wellcome can replace this file; a hash mismatch means the lab is not the same.
# The workbook is untrusted data: it is only ever read with pandas/openpyxl.
#   tools/fetch_wgm.sh [dest]        (default data/raw/wgm)
set -e
DEST=${1:-data/raw/wgm}
URL=https://wellcome.org/sites/default/files/wgm2018-dataset-crosstabs-all-countries.xlsx
mkdir -p "$DEST"
cd "$DEST"
curl -sSL -o wgm2018.xlsx "$URL"
sha256sum -c - <<SUMS
8bcbacd403a4ee531a526913364c163005be6b497ee429214ec2dd6d7a500c90  wgm2018.xlsx
SUMS
