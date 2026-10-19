// 2D Gaussian Splatting image fitting kernel (freestanding C -> WebAssembly, no libc).
// Representation follows GaussianImage/GSVC: each splat = mean (x,y), Cholesky factor of
// the inverse covariance (m11, m21, m22) and an RGB colour; a pixel is the sum of weighted
// colours (accumulated blending). Gradients are analytic; optimizer is Adam.
#define MAXG 6000
#define MAXPIX (320 * 240)
#define NP 8
#define EXPORT __attribute__((visibility("default")))

static float target[MAXPIX * 3];
static float out[MAXPIX * 3];
static float err[MAXPIX];
static float P[MAXG * NP], G[MAXG * NP], AM[MAXG * NP], AV[MAXG * NP];
static float imp[MAXG];
static int W = 0, H = 0, N = 0, T = 0;
static unsigned rng = 1;

static unsigned rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float frand(void) { return (rnd() & 0xFFFFFF) / 16777216.0f; }
static float fabsf_(float x) { return x < 0 ? -x : x; }
static float sqrtf_(float x) { return __builtin_sqrtf(x); }

// exp(x) for x <= 0 via 2^(x*log2e), polynomial on the fractional part.
static float expneg(float x) {
  if (x < -30.0f) return 0.0f;
  float t = x * 1.44269504f;
  int i = (int)t; if ((float)i > t) i--;
  float f = t - (float)i;
  float p = 1.0f + f * (0.6931472f + f * (0.2402265f + f * (0.0555041f + f * (0.0096181f + f * 0.0013334f))));
  union { float f; int i; } u; u.i = (i + 127) << 23;
  return p * u.f;
}

// clang lowers zeroing loops to memset; provide it since there is no libc.
void *memset(void *d, int c, unsigned long n) {
  unsigned char *p = (unsigned char *)d;
  while (n--) *p++ = (unsigned char)c;
  return d;
}
void *memcpy(void *d, const void *s, unsigned long n) {
  unsigned char *a = (unsigned char *)d; const unsigned char *b = (const unsigned char *)s;
  while (n--) *a++ = *b++;
  return d;
}

EXPORT float *target_ptr(void) { return target; }
EXPORT float *out_ptr(void) { return out; }
EXPORT float *params_ptr(void) { return P; }
EXPORT int max_gaussians(void) { return MAXG; }
EXPORT int max_pixels(void) { return MAXPIX; }
EXPORT int count(void) { return N; }
EXPORT void set_count(int n) { N = n < 0 ? 0 : (n > MAXG ? MAXG : n); }

EXPORT int setup(int w, int h) {
  if (w * h > MAXPIX) return 0;
  W = w; H = h; return 1;
}

EXPORT void reset_adam(void) {
  for (int i = 0; i < MAXG * NP; i++) { AM[i] = 0; AV[i] = 0; }
  T = 0;
}

static void place(int g, float x, float y, float sigma) {
  float *p = P + g * NP;
  p[0] = x; p[1] = y;
  p[2] = 1.0f / sigma; p[3] = 0.0f; p[4] = 1.0f / sigma;
  int ix = (int)x, iy = (int)y;
  if (ix < 0) ix = 0; if (ix >= W) ix = W - 1; if (iy < 0) iy = 0; if (iy >= H) iy = H - 1;
  const float *t = target + (iy * W + ix) * 3;
  // initial colour scaled down because splats overlap additively
  p[5] = t[0] * 0.5f; p[6] = t[1] * 0.5f; p[7] = t[2] * 0.5f;
}

EXPORT void init_gaussians(int n, unsigned seed) {
  rng = seed ? seed : 1;
  set_count(n);
  float sigma = sqrtf_((float)(W * H) / (float)(N > 0 ? N : 1)) * 0.9f;
  for (int g = 0; g < N; g++) place(g, frand() * W, frand() * H, sigma);
  reset_adam();
}

static float radius_of(const float *p) {
  float a11 = p[2] * p[2], a12 = p[2] * p[3], a22 = p[3] * p[3] + p[4] * p[4];
  float det = a11 * a22 - a12 * a12;
  if (det < 1e-12f) return 64.0f;
  float s11 = a22 / det, s22 = a11 / det, s12 = -a12 / det;
  float tr = s11 + s22, dt = s11 * s22 - s12 * s12;
  float disc = tr * tr * 0.25f - dt; if (disc < 0) disc = 0;
  float r = 3.2f * sqrtf_(tr * 0.5f + sqrtf_(disc));
  return r > 64.0f ? 64.0f : r;
}

// Forward render; if with_grad, also back-propagate the MSE loss into G.
static float pass(int with_grad) {
  int npx = W * H;
  for (int i = 0; i < npx * 3; i++) out[i] = 0.0f;
  for (int g = 0; g < N; g++) {
    const float *p = P + g * NP;
    float r = radius_of(p);
    int x0 = (int)(p[0] - r), x1 = (int)(p[0] + r) + 1, y0 = (int)(p[1] - r), y1 = (int)(p[1] + r) + 1;
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0; if (x1 > W) x1 = W; if (y1 > H) y1 = H;
    for (int y = y0; y < y1; y++) {
      float dy = (float)y + 0.5f - p[1];
      for (int x = x0; x < x1; x++) {
        float dx = (float)x + 0.5f - p[0];
        float u = p[2] * dx + p[3] * dy, v = p[4] * dy;
        float q = u * u + v * v;
        if (q > 14.0f) continue;
        float w = expneg(-0.5f * q);
        float *o = out + (y * W + x) * 3;
        o[0] += w * p[5]; o[1] += w * p[6]; o[2] += w * p[7];
      }
    }
  }
  float loss = 0.0f;
  for (int i = 0; i < npx; i++) {
    float e = 0.0f;
    for (int c = 0; c < 3; c++) { float d = out[i * 3 + c] - target[i * 3 + c]; e += d * d; }
    err[i] = e; loss += e;
  }
  loss /= (float)(npx * 3);
  if (!with_grad) return loss;

  float scale = 2.0f / (float)(npx * 3);
  for (int i = 0; i < N * NP; i++) G[i] = 0.0f;
  for (int g = 0; g < N; g++) {
    const float *p = P + g * NP;
    float *gr = G + g * NP;
    float r = radius_of(p);
    int x0 = (int)(p[0] - r), x1 = (int)(p[0] + r) + 1, y0 = (int)(p[1] - r), y1 = (int)(p[1] + r) + 1;
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0; if (x1 > W) x1 = W; if (y1 > H) y1 = H;
    float acc = 0.0f;
    for (int y = y0; y < y1; y++) {
      float dy = (float)y + 0.5f - p[1];
      for (int x = x0; x < x1; x++) {
        float dx = (float)x + 0.5f - p[0];
        float u = p[2] * dx + p[3] * dy, v = p[4] * dy;
        float q = u * u + v * v;
        if (q > 14.0f) continue;
        float w = expneg(-0.5f * q);
        acc += w;
        int k = (y * W + x) * 3;
        float e0 = (out[k] - target[k]) * scale, e1 = (out[k + 1] - target[k + 1]) * scale, e2 = (out[k + 2] - target[k + 2]) * scale;
        gr[5] += e0 * w; gr[6] += e1 * w; gr[7] += e2 * w;
        float dLdw = e0 * p[5] + e1 * p[6] + e2 * p[7];
        float dLdq = dLdw * (-0.5f * w);
        gr[0] += dLdq * (-2.0f * u * p[2]);
        gr[1] += dLdq * (-2.0f * u * p[3] - 2.0f * v * p[4]);
        gr[2] += dLdq * (2.0f * u * dx);
        gr[3] += dLdq * (2.0f * u * dy);
        gr[4] += dLdq * (2.0f * v * dy);
      }
    }
    imp[g] = acc;
  }
  return loss;
}

EXPORT float render(void) { return pass(0); }

// One Adam step. lr_pos in pixels, lr_cov for Cholesky terms, lr_col for colours.
EXPORT float step(float lr_pos, float lr_cov, float lr_col) {
  float loss = pass(1);
  T++;
  const float b1 = 0.9f, b2 = 0.999f;
  float c1 = 1.0f, c2 = 1.0f;
  for (int i = 0; i < T && i < 2000; i++) { c1 *= b1; c2 *= b2; }
  c1 = 1.0f - c1; c2 = 1.0f - c2;
  for (int g = 0; g < N; g++) {
    for (int j = 0; j < NP; j++) {
      int k = g * NP + j;
      float gr = G[k];
      AM[k] = b1 * AM[k] + (1 - b1) * gr;
      AV[k] = b2 * AV[k] + (1 - b2) * gr * gr;
      float lr = j < 2 ? lr_pos : (j < 5 ? lr_cov : lr_col);
      P[k] -= lr * (AM[k] / c1) / (sqrtf_(AV[k] / c2) + 1e-8f);
    }
    float *p = P + g * NP;
    if (p[2] < 0.02f) p[2] = 0.02f;
    if (p[4] < 0.02f) p[4] = 0.02f;
    if (p[0] < 0) p[0] = 0; if (p[0] > W) p[0] = (float)W;
    if (p[1] < 0) p[1] = 0; if (p[1] > H) p[1] = (float)H;
  }
  return loss;
}

// GSVC-style pruning: drop splats whose total weight (contribution) is below thr.
EXPORT int prune(float thr) {
  int k = 0;
  for (int g = 0; g < N; g++) {
    if (imp[g] >= thr) {
      if (k != g) for (int j = 0; j < NP; j++) { P[k * NP + j] = P[g * NP + j]; AM[k * NP + j] = AM[g * NP + j]; AV[k * NP + j] = AV[g * NP + j]; }
      imp[k] = imp[g];
      k++;
    }
  }
  N = k;
  return N;
}

// GSVC-style augmentation: add splats where the current error is high (new content / motion).
EXPORT int add_gaussians(int k, float sigma, unsigned seed) {
  rng = seed ? seed : 7;
  pass(0);
  int npx = W * H;
  float mean = 0.0f;
  for (int i = 0; i < npx; i++) mean += err[i];
  mean /= (float)npx;
  int added = 0;
  for (int tries = 0; tries < k * 50 && added < k && N < MAXG; tries++) {
    int i = (int)(frand() * npx);
    if (err[i] > mean * 2.0f) {
      int g = N++;
      place(g, (float)(i % W) + 0.5f, (float)(i / W) + 0.5f, sigma);
      float *p = P + g * NP;
      const float *t = target + i * 3, *o = out + i * 3;
      p[5] = t[0] - o[0]; p[6] = t[1] - o[1]; p[7] = t[2] - o[2];  // fit the residual
      for (int j = 0; j < NP; j++) { AM[g * NP + j] = 0; AV[g * NP + j] = 0; }
      added++;
    }
  }
  return added;
}
