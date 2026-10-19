#!/bin/bash
# Build gs.wasm with plain clang (no Emscripten needed).
set -e
cd "$(dirname "$0")"
clang --target=wasm32 -O3 -nostdlib -fvisibility=hidden -ffast-math \
  -Wl,--no-entry -Wl,--export-dynamic -Wl,--initial-memory=33554432 \
  -o public/gs.wasm gs.c
echo "public/gs.wasm: $(wc -c < public/gs.wasm) bytes"
