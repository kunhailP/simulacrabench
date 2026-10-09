#!/bin/sh
# Fetch the TabICLv2 classifier checkpoint bundled in sub/v8 (BSD-3-Clause),
# pinned to a Hugging Face commit and checked by SHA-256.
#   jingang/TabICL @ 4dcd344ece2c00be9e831fdd35bed57b5ad83e19
#   tabicl-classifier-v2-20260212.ckpt
set -e
DEST=${1:-sub/v8/weights}
REV=4dcd344ece2c00be9e831fdd35bed57b5ad83e19
FILE=tabicl-classifier-v2-20260212.ckpt
SHA=bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0
mkdir -p "$DEST"
curl -sSL -o "$DEST/$FILE" "https://huggingface.co/jingang/TabICL/resolve/$REV/$FILE"
echo "$SHA  $DEST/$FILE" | sha256sum -c -
