// Small seeded RNG utilities (mulberry32 + Box-Muller). Deterministic across runs.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeRng(seed) {
  const u = mulberry32(seed);
  let spare = null;
  const rng = {
    uniform: u,
    normal(mean = 0, sd = 1) {
      if (spare !== null) { const s = spare; spare = null; return mean + sd * s; }
      let x, y, r;
      do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0);
      const f = Math.sqrt(-2 * Math.log(r) / r);
      spare = y * f;
      return mean + sd * x * f;
    },
    bernoulli(p) { return u() < p; },
    // Beta via two gammas (Marsaglia-Tsang), shape >= 1 branch plus boost for < 1
    gamma(k) {
      if (k < 1) { const g = rng.gamma(k + 1); return g * Math.pow(u(), 1 / k); }
      const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
      for (;;) {
        let x, v;
        do { x = rng.normal(); v = 1 + c * x; } while (v <= 0);
        v = v * v * v;
        const uu = u();
        if (uu < 1 - 0.0331 * x * x * x * x) return d * v;
        if (Math.log(uu) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
      }
    },
    beta(a, b) { const x = rng.gamma(a), y = rng.gamma(b); return x / (x + y); },
    pick(arr, weights) {
      const tot = weights.reduce((s, w) => s + w, 0);
      let r = u() * tot;
      for (let i = 0; i < arr.length; i++) { r -= weights[i]; if (r <= 0) return arr[i]; }
      return arr[arr.length - 1];
    },
  };
  return rng;
}

export const logistic = (z) => 1 / (1 + Math.exp(-z));
