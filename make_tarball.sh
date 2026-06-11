#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

checkpoint="whisperdrz-large-v3.safetensors"
out="dist/whisperdrz-public.tar"

[ -f "$checkpoint" ] || { echo "missing $checkpoint" >&2; exit 1; }

mkdir -p dist
rm -f "$out"

{ git ls-files; echo "$checkpoint"; } | tar -cf "$out" -T -

tar -tf "$out"
du -h "$out"
