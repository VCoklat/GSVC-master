// GSVC Web: 2D Gaussian Splatting video compression in the browser.
// Kernel (render + analytic gradients + Adam) runs in gs.wasm; this file handles frames,
// GSVC-style temporal prediction / keyframes, quantization, entropy coding and metrics.

const $ = (id) => document.getElementById(id);
const sleep = (ms = 0) => new Promise((r) => setTimeout(r, ms));
let K = null; // wasm exports
let stop = false;
let results = [];

async function loadWasm() {
  const { instance } = await WebAssembly.instantiateStreaming(fetch('gs.wasm'));
  K = instance.exports;
  $('status').textContent = `WebAssembly siap (maks ${K.max_gaussians()} splat, ${K.max_pixels()} piksel).`;
}

// ---------- frame sources ----------
function sampleFrames(W, H, n) {
  const cv = new OffscreenCanvas(W, H);
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  const frames = [];
  for (let f = 0; f < n; f++) {
    const t = f / Math.max(1, n - 1);
    const g = ctx.createLinearGradient(0, 0, W, H);
    g.addColorStop(0, `hsl(${210 + 20 * t}, 55%, 35%)`);
    g.addColorStop(1, `hsl(${30 + 10 * t}, 60%, 60%)`);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#f2f2ea';
    ctx.fillRect(W * (0.05 + 0.5 * t), H * 0.62, W * 0.3, H * 0.22); // sliding block
    const balls = [['#e63946', 0.2, 0.3, 0.12], ['#2a9d8f', 0.7, 0.35, 0.09], ['#ffb703', 0.5, 0.7, 0.07]];
    balls.forEach(([c, x, y, r], i) => {
      ctx.fillStyle = c;
      ctx.beginPath();
      ctx.arc(W * (x + 0.15 * Math.sin(t * 6.28 + i)), H * (y + 0.1 * Math.cos(t * 6.28 + i * 2)), W * r, 0, 6.283);
      ctx.fill();
    });
    if (f === Math.floor(n * 0.6)) { // scene cut-ish: new object appears
      ctx.fillStyle = '#7b2cbf';
      ctx.fillRect(W * 0.75, H * 0.05, W * 0.2, H * 0.3);
    }
    frames.push(ctx.getImageData(0, 0, W, H));
  }
  return frames;
}

function fileFrames(file, W, n) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    if (file.type.startsWith('image/')) {
      const im = new Image();
      im.onload = () => {
        const H = Math.round((W * im.naturalHeight) / im.naturalWidth);
        const cv = new OffscreenCanvas(W, H);
        const ctx = cv.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(im, 0, 0, W, H);
        resolve([ctx.getImageData(0, 0, W, H)]);
      };
      im.onerror = reject;
      im.src = url;
      return;
    }
    const v = document.createElement('video');
    v.muted = true; v.playsInline = true; v.src = url;
    v.onloadedmetadata = async () => {
      const H = Math.round((W * v.videoHeight) / v.videoWidth);
      const cv = new OffscreenCanvas(W, H);
      const ctx = cv.getContext('2d', { willReadFrequently: true });
      const span = Math.min(v.duration, n / 6); // ~6 fps sampling
      const frames = [];
      for (let i = 0; i < n; i++) {
        v.currentTime = Math.min(v.duration - 0.01, (span * i) / Math.max(1, n - 1));
        await new Promise((r) => (v.onseeked = r));
        ctx.drawImage(v, 0, 0, W, H);
        frames.push(ctx.getImageData(0, 0, W, H));
      }
      resolve(frames);
    };
    v.onerror = reject;
  });
}

// ---------- helpers ----------
function setTarget(img) {
  const n = img.width * img.height;
  const t = new Float32Array(K.memory.buffer, K.target_ptr(), n * 3);
  for (let i = 0; i < n; i++) {
    t[i * 3] = img.data[i * 4] / 255;
    t[i * 3 + 1] = img.data[i * 4 + 1] / 255;
    t[i * 3 + 2] = img.data[i * 4 + 2] / 255;
  }
}

function outImage(W, H) {
  const o = new Float32Array(K.memory.buffer, K.out_ptr(), W * H * 3);
  const img = new ImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    for (let c = 0; c < 3; c++) img.data[i * 4 + c] = Math.max(0, Math.min(255, Math.round(o[i * 3 + c] * 255)));
    img.data[i * 4 + 3] = 255;
  }
  return img;
}

function psnr(a, b) {
  let s = 0;
  const n = a.width * a.height;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) { const d = (a.data[i * 4 + c] - b.data[i * 4 + c]) / 255; s += d * d; }
  const mse = s / (n * 3);
  return mse <= 1e-12 ? 99 : 10 * Math.log10(1 / mse);
}

async function deflate(bytes) {
  const cs = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(cs).arrayBuffer());
}

// ---------- quantization (GSVC-style: per-channel uniform quantizer, ranges fixed at keyframe) ----------
// bits per parameter: x, y, m11, m21, m22, r, g, b
const NP = 8;
const BITS = [12, 12, 8, 8, 8, 8, 8, 8];
const QMAX = BITS.map((b) => (1 << b) - 1);
function quantRanges(P, N) {
  const lo = new Float32Array(NP).fill(Infinity), hi = new Float32Array(NP).fill(-Infinity);
  for (let g = 0; g < N; g++) for (let j = 0; j < NP; j++) { const v = P[g * NP + j]; if (v < lo[j]) lo[j] = v; if (v > hi[j]) hi[j] = v; }
  for (let j = 0; j < NP; j++) { const pad = (hi[j] - lo[j]) * 0.25 + 1e-3; lo[j] -= pad; hi[j] += pad; } // headroom for later frames
  return { lo, hi };
}
function quantize(P, N, R) {
  const q = new Int32Array(N * NP);
  for (let g = 0; g < N; g++) for (let j = 0; j < NP; j++) {
    const v = (P[g * NP + j] - R.lo[j]) / (R.hi[j] - R.lo[j]);
    q[g * NP + j] = Math.max(0, Math.min(QMAX[j], Math.round(v * QMAX[j])));
  }
  return q;
}
function dequantizeInto(P, q, N, R) {
  for (let g = 0; g < N; g++) for (let j = 0; j < NP; j++) P[g * NP + j] = R.lo[j] + (q[g * NP + j] / QMAX[j]) * (R.hi[j] - R.lo[j]);
}
// Channel-major byte packing (helps the entropy coder).
function packAbs(q, from, to) {
  const n = to - from, chunks = [];
  for (let j = 0; j < NP; j++) {
    const arr = BITS[j] > 8 ? new Uint16Array(n) : new Uint8Array(n);
    for (let g = 0; g < n; g++) arr[g] = q[(from + g) * NP + j];
    chunks.push(new Uint8Array(arr.buffer));
  }
  return chunks;
}
// Delta for existing splats with a dead-zone: tiny changes are not sent (decoder keeps old value).
// Modifies q in place so encoder and decoder stay in sync (closed loop).
function packDelta(q, prev, nPrev) {
  const chunks = [];
  for (let j = 0; j < NP; j++) {
    const big = BITS[j] > 8;
    const arr = big ? new Int16Array(nPrev) : new Int8Array(nPrev);
    const lim = big ? 32767 : 127;
    for (let g = 0; g < nPrev; g++) {
      const k = g * NP + j;
      let d = q[k] - prev[k];
      if (Math.abs(d) <= 1) d = 0;
      d = Math.max(-lim, Math.min(lim, d));
      arr[g] = d;
      q[k] = prev[k] + d;
    }
    chunks.push(new Uint8Array(arr.buffer));
  }
  return chunks;
}
function concat(parts) {
  const len = parts.reduce((s, p) => s + p.length, 0), out = new Uint8Array(len);
  let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

// ---------- drawing ----------
function show(id, img) {
  const cv = $(id);
  if (cv.width !== img.width || cv.height !== img.height) { cv.width = img.width; cv.height = img.height; }
  cv.getContext('2d').putImageData(img, 0, 0);
}
function drawSplats(id) {
  if (!$('showSplats').checked) return;
  const ctx = $(id).getContext('2d');
  const P = new Float32Array(K.memory.buffer, K.params_ptr(), K.count() * NP);
  ctx.save();
  ctx.lineWidth = 0.5;
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  for (let g = 0; g < K.count(); g++) {
    const [x, y, m11, m21, m22] = P.subarray(g * NP, g * NP + 5);
    const a11 = m11 * m11, a12 = m11 * m21, a22 = m21 * m21 + m22 * m22, det = a11 * a22 - a12 * a12;
    const s11 = a22 / det, s22 = a11 / det, s12 = -a12 / det;
    const tr = s11 + s22, d = Math.sqrt(Math.max(0, (tr * tr) / 4 - (s11 * s22 - s12 * s12)));
    const l1 = tr / 2 + d, l2 = Math.max(1e-6, tr / 2 - d);
    const ang = 0.5 * Math.atan2(2 * s12, s11 - s22);
    ctx.beginPath();
    ctx.ellipse(x, y, Math.min(60, Math.sqrt(l1)), Math.min(60, Math.sqrt(l2)), ang, 0, 6.283);
    ctx.stroke();
  }
  ctx.restore();
}

async function jpegBaseline(img) {
  const cv = new OffscreenCanvas(img.width, img.height);
  cv.getContext('2d').putImageData(img, 0, 0);
  const blob = await cv.convertToBlob({ type: 'image/jpeg', quality: 0.7 });
  const bmp = await createImageBitmap(blob);
  const cv2 = new OffscreenCanvas(img.width, img.height);
  const c2 = cv2.getContext('2d', { willReadFrequently: true });
  c2.drawImage(bmp, 0, 0);
  const dec = c2.getImageData(0, 0, img.width, img.height);
  return { bytes: blob.size, img: dec, psnr: psnr(img, dec) };
}

function chart() {
  const svg = $('chart');
  const css = getComputedStyle(document.documentElement);
  const A = css.getPropertyValue('--chart-a'), B = css.getPropertyValue('--chart-b'), M = css.getPropertyValue('--muted');
  if (!results.length) { svg.innerHTML = ''; return; }
  const ys = results.flatMap((r) => [r.psnr, r.jpgPsnr]);
  const lo = Math.floor(Math.min(...ys) - 1), hi = Math.ceil(Math.max(...ys) + 1);
  const X = (i) => 40 + (results.length === 1 ? 260 : (i * 540) / (results.length - 1));
  const Y = (v) => 160 - ((v - lo) / (hi - lo)) * 140;
  const line = (key, col) => `<polyline fill="none" stroke="${col}" stroke-width="2" points="${results.map((r, i) => `${X(i)},${Y(r[key])}`).join(' ')}"/>` +
    results.map((r, i) => `<circle cx="${X(i)}" cy="${Y(r[key])}" r="${r.type === 'I' && key === 'psnr' ? 4 : 2.5}" fill="${col}"><title>Frame ${i}: ${r[key].toFixed(2)} dB</title></circle>`).join('');
  svg.innerHTML = `
    <text x="4" y="${Y(hi) + 4}" font-size="10" fill="${M}">${hi} dB</text>
    <text x="4" y="${Y(lo) + 4}" font-size="10" fill="${M}">${lo} dB</text>
    <line x1="40" x2="580" y1="160" y2="160" stroke="${M}" stroke-opacity="0.3"/>
    ${line('jpgPsnr', B)}${line('psnr', A)}
    <text x="420" y="14" font-size="11" fill="${A}">● splat (besar = keyframe)</text>
    <text x="420" y="28" font-size="11" fill="${B}">● JPEG q=0.7</text>`;
}

function updateStats() {
  if (!results.length) return;
  const avg = (k) => results.reduce((s, r) => s + r[k], 0) / results.length;
  const sum = (k) => results.reduce((s, r) => s + r[k], 0);
  const px = results.length * results[0].W * results[0].H;
  $('sPsnr').textContent = `${avg('psnr').toFixed(2)} dB`;
  $('sJpgPsnr').textContent = `${avg('jpgPsnr').toFixed(2)} dB`;
  $('sSize').textContent = `${(sum('bytes') / 1024).toFixed(1)} KB`;
  $('sJpgSize').textContent = `${(sum('jpgBytes') / 1024).toFixed(1)} KB`;
  $('sBpp').textContent = `${((sum('bytes') * 8) / px).toFixed(3)}`;
  $('sN').textContent = `${results[results.length - 1].n}`;
  $('rows').innerHTML = results.map((r, i) => `<tr><td>${i}</td><td>${r.type === 'I' ? 'Keyframe' : 'Prediksi'}</td><td>${r.iters}</td><td>${r.n}</td><td>${r.bytes}</td><td>${r.psnr.toFixed(2)}</td><td>${r.jpgBytes}</td><td>${r.jpgPsnr.toFixed(2)}</td><td>${r.ms.toFixed(0)}</td></tr>`).join('');
  chart();
}

// ---------- training ----------
async function optimize(iters, lrs, W, H, label) {
  for (let i = 0; i < iters; i++) {
    K.step(...lrs);
    if (i % 10 === 9) {
      if (stop) return false;
      K.render();
      show('cvRec', outImage(W, H));
      drawSplats('cvRec');
      $('status').textContent = `${label}: iterasi ${i + 1}/${iters}`;
      await sleep();
    }
  }
  return true;
}

async function run() {
  stop = false;
  results = [];
  $('run').disabled = true;
  $('play').disabled = true;
  updateStats(); chart(); $('rows').innerHTML = '';
  const W = parseInt($('width').value, 10);
  const n = Math.min(K.max_gaussians(), parseInt($('ngs').value, 10));
  const nf = parseInt($('nframes').value, 10);
  const itk = parseInt($('itk').value, 10), itp = parseInt($('itp').value, 10);
  let frames;
  try {
    if ($('source').value === 'file') {
      const f = $('file').files[0];
      if (!f) { $('status').textContent = 'Pilih file dulu.'; $('run').disabled = false; return; }
      $('status').textContent = 'Mengambil frame…';
      frames = await fileFrames(f, W, nf);
    } else {
      frames = sampleFrames(W, Math.round(W * 0.75), nf);
    }
  } catch (e) {
    $('status').textContent = 'Gagal membaca file: ' + e;
    $('run').disabled = false;
    return;
  }
  const H = frames[0].height;
  if (!K.setup(W, H)) { $('status').textContent = 'Resolusi terlalu besar.'; $('run').disabled = false; return; }

  const KEY_LR = [0.25, 0.01, 0.008], P_LR = [0.12, 0.005, 0.004];
  let R = null, prevQ = null, keyLoss = 0;
  for (let f = 0; f < frames.length && !stop; f++) {
    const t0 = performance.now();
    show('cvOrig', frames[f]);
    setTarget(frames[f]);
    let type = 'P', iters = 0;
    if (f > 0) {
      K.reset_adam();
      K.add_gaussians(Math.max(4, Math.round(n * 0.03)), Math.sqrt((W * H) / n), 1000 + f);
      if (!(await optimize(itp, P_LR, W, H, `Frame ${f} (prediksi)`))) break;
      iters = itp;
      const loss = K.render();
      if (loss > keyLoss * 2.5) type = 'I'; // GSVC: keyframe saat galat melonjak
    } else type = 'I';

    if (type === 'I') {
      K.init_gaussians(n, 17 + f);
      if (!(await optimize(Math.round(itk / 2), KEY_LR, W, H, `Frame ${f} (keyframe)`))) break;
      K.prune(0.3); // buang splat berkontribusi kecil
      if (!(await optimize(itk - Math.round(itk / 2), KEY_LR, W, H, `Frame ${f} (keyframe)`))) break;
      iters = itk;
    }

    // encode: quantize -> (delta) -> deflate ; decoder sees the dequantized params
    const N = K.count();
    const P = new Float32Array(K.memory.buffer, K.params_ptr(), N * NP);
    let payload;
    if (type === 'I') {
      R = quantRanges(P, N);
      const q = quantize(P, N, R);
      const header = new Float32Array([N, ...R.lo, ...R.hi]);
      payload = concat([new Uint8Array(header.buffer), ...packAbs(q, 0, N)]);
      dequantizeInto(P, q, N, R);
      prevQ = q;
    } else {
      const q = quantize(P, N, R);
      const prevN = prevQ.length / NP;
      const head = new Uint8Array(4); new DataView(head.buffer).setUint32(0, N, true);
      payload = concat([head, ...packDelta(q, prevQ, prevN), ...packAbs(q, prevN, N)]);
      dequantizeInto(P, q, N, R);
      prevQ = q;
    }
    const bytes = (await deflate(payload)).length;
    const loss = K.render();
    if (type === 'I') keyLoss = loss;
    const rec = outImage(W, H);
    show('cvRec', rec);
    drawSplats('cvRec');
    const jpg = await jpegBaseline(frames[f]);
    show('cvJpg', jpg.img);
    results.push({ type, iters, n: N, bytes, psnr: psnr(frames[f], rec), jpgBytes: jpg.bytes, jpgPsnr: jpg.psnr, ms: performance.now() - t0, W, H, rec, orig: frames[f], jpg: jpg.img });
    updateStats();
  }
  $('status').textContent = stop ? 'Dihentikan.' : `Selesai: ${results.length} frame.`;
  $('run').disabled = false;
  $('play').disabled = results.length === 0;
}

async function play() {
  for (const r of results) {
    show('cvOrig', r.orig); show('cvRec', r.rec); show('cvJpg', r.jpg);
    await sleep(220);
  }
}

$('source').onchange = () => { if ($('source').value === 'file') $('file').click(); };
$('file').onchange = () => { const f = $('file').files[0]; if (f) $('status').textContent = `File: ${f.name}`; };
$('run').onclick = () => run();
$('stop').onclick = () => { stop = true; };
$('play').onclick = () => play();
loadWasm().catch((e) => { $('status').textContent = 'Gagal memuat WebAssembly: ' + e; });
