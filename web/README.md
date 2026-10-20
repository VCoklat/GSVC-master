# GSVC Web

A video compression demo with 2D Gaussian Splatting that runs entirely in the browser (WebAssembly, no server or GPU), inspired by GSVC (arXiv:2501.12060).

- `gs.c`: freestanding C kernel (render, analytic gradients, Adam, prune/add splats) compiled to `public/gs.wasm` (~7 KB).
- `public/app.js`: frame capture (synthetic sample video or upload), temporal prediction + keyframe detection, 12/8-bit quantization + dead-zone deltas + deflate, PSNR after dequantization, JPEG q=0.7 baseline.

Build the kernel (plain clang, no Emscripten): `./build.sh`
Run locally: `cd public && python3 -m http.server 8000`
Deploy on Vercel: import the repo (branch `web-demo`), Root Directory = `web`, Framework Preset = Other.

This is a lightweight demo implementation, not a port of the official code; at small resolutions on a CPU it currently performs below JPEG.
