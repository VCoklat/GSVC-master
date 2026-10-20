# GSVC Web

Demo kompresi video 2D Gaussian Splatting yang berjalan sepenuhnya di peramban (WebAssembly, tanpa server/GPU), terinspirasi GSVC (arXiv:2501.12060).

- `gs.c`: kernel C freestanding (render, gradien analitik, Adam, pangkas/tambah splat), dikompilasi ke `public/gs.wasm` (±7 KB).
- `public/app.js`: ambil frame (video contoh sintetis atau unggahan), prediksi temporal + deteksi keyframe, kuantisasi 12/8 bit + delta dead-zone + deflate, PSNR setelah dekuantisasi, pembanding JPEG q=0.7.

Build kernel (cukup clang, tidak perlu Emscripten): `./build.sh`
Jalankan lokal: `cd public && python3 -m http.server 8000`
Deploy Vercel: import repo (branch `web-demo`). Bisa dengan Root Directory default (pakai `vercel.json` di root, output `web/public`) atau Root Directory = `web`. Framework preset: Other.

Ini implementasi ringan untuk demo, bukan port kode resmi; pada resolusi kecil di CPU hasilnya di bawah JPEG.
