#!/bin/sh
# Download the Syrian Refugee Life Study (S-RLS, Jordan; Harvard Dataverse,
# CC0) household files the real-data lab is built from, and check them against
# the hashes of the files the lab was produced with. Dataverse can version
# these files; a hash mismatch means the lab is not the same.
#   2021 phone round  doi:10.7910/DVN/OURCBF  PS_deidentified_public_clean_label (Stata original)
#   2022 F2F round    doi:10.7910/DVN/DRGBZ6  panel_F2F_deidentified_clean_label.dta
# The 2024 F2F round (datafile 13179969) was considered and not used: it
# shares too few items with the 2021 round (see tools/proxy_srls.py).
#   tools/fetch_srls.sh [dest]        (default data/raw/srls)
set -e
DEST=${1:-data/raw/srls}
API=https://dataverse.harvard.edu/api/access/datafile
mkdir -p "$DEST"
cd "$DEST"
curl -sSL -o ps2021_label.dta "$API/10282490?format=original"
curl -sSL -o f2f2022_label.dta "$API/11041978"
sha256sum -c - <<SUMS
1962b452e88ab2c8bc4cc686ed5efeb7175a042bdc7594b0ee4dcfacb2f2ea63  ps2021_label.dta
8cb4c2cf370a3249a39ea258bbc4012673a0f96d8960ec53ab8f01fbba269bd3  f2f2022_label.dta
SUMS
