"""
XC-AdaGuiDE extended experiments (P1-P8).

Requires the engine (xc_engine.py, or the engine cell of the notebook). Each module
writes its own folder and CSV and does not read earlier outputs.

  P1  Search and consistency under additive Gaussian noise
  P2  Local importance: forward / central / second difference x eta_h
  P3  Eq. (12) gate: base vs XC vs XC-gated, axis-aligned and rotated
  P4  Run time per module, D = 10 ... 100
  P5  Block-rotated (partially coupled) suite, block size 1 / 5 / 10 / 30
  P6  Quadrotor attitude, cascaded PID, 12 gains
  P7  Parameter sweep on a separate tuning set (functions outside the test suite)
  P8  Bound revision on/off at 10000*D evaluations
"""
import os, time, json
import numpy as np
from scipy import stats
from scipy.stats import qmc

try:                                   # notebook: engine cell already executed
    Benchmark
except NameError:                      # script: import the engine module
    from xc_engine import *            # noqa: F401,F403
    from xc_engine import _bootstrap_mean_ci, _log10f, _bounds, _ortho, _wcsv, ZSTAR, _zstar_zero

EXT_CFG = dict(
    ext_seeds=15, ext_perm_iters=2000, ext_outroot="extended",
    noise_funcs=["Ellipsoid", "Zakharov", "Discus", "DixonPrice", "BentCigar", "Rosenbrock", "Rastrigin"],
    noise_sigmas=[0.0, 1e-3, 1e-2, 5e-2], noise_dim=30,
    diff_etas=[1e-4, 1e-3, 1e-2], diff_schemes=["forward", "central", "second"],
    gate_alpha=0.05,
    timing_dims=[10, 30, 50, 100], timing_reps=3,
    block_sizes=[1, 5, 10, 30], block_dim=30,
    quad_seeds=20, quad_NP=40, quad_fe=6000, quad_posthoc_N=512,
    tune_funcs=["StyblinskiTang", "Alpine1", "Salomon", "Schwefel222", "Weierstrass"],
    tune_grid_fixed=dict(kappa=[0.0, 0.1, 0.2, 0.3, 0.5], decay_frac=[0.25, 0.5, 0.75, 1.0], sens_every=[1, 5, 10]),
    tune_grid_cec=dict(gamma=[0.0, 0.15, 0.3, 0.6], beta0=[0.1, 0.2, 0.3, 0.5], G_trigger=[10, 20, 40]),
    tune_run_cec=False,
    onoff_funcs=["Rastrigin", "Ackley", "Griewank", "Levy", "Rosenbrock"], onoff_seeds=15,
)

def _dir(cfg, name):
    d = os.path.join(cfg["ext_outroot"], name); os.makedirs(d, exist_ok=True); return d

def _fn(name):
    return {n: (f, a, b) for n, f, a, b in SUITE + EXTRA_SUITE}[name]

def _per_seed_r(STs, rhos):
    out = []
    for a, b in zip(STs, rhos):
        out.append(float(np.corrcoef(a, b)[0, 1]) if (np.std(a) > 0 and np.std(b) > 0) else np.nan)
    return out

def _consistency_block(STs, rhos, cfg, key):
    """mean per-seed r, bootstrap CI, permutation p (same statistic as E4) and the null draws."""
    STs = np.asarray(STs); rhos = np.asarray(rhos)
    ok = (STs.std(1) > 0) & (rhos.std(1) > 0)
    if ok.sum() < 2: return dict(r_mean=np.nan, ci_low=np.nan, ci_high=np.nan, perm_p=np.nan), None
    obs, p, null = perm_mean_r(STs[ok], rhos[ok], cfg["ext_perm_iters"], stable_seed("p2perm", key))
    m, lo, hi = _bootstrap_mean_ci(_per_seed_r(STs[ok], rhos[ok]), cfg["boot_iters"], stable_seed("p2boot", key))
    return dict(r_mean=obs, ci_low=lo, ci_high=hi, perm_p=p), null

def _holm_verdicts(rows, a="base_median", b="xc_median", pkey="wilcoxon_p"):
    adj = holm([r[pkey] for r in rows])
    for r, pa in zip(rows, adj):
        r["holm_p"] = float(pa)
        r["verdict"] = ("win" if r[b] < r[a] else "loss") if pa < 0.05 else "tie"
    return {v: sum(r["verdict"] == v for r in rows) for v in ("win", "loss", "tie")}

# ---------------------------------------------------------------------------------------------
# Extra functions for P7 (outside the 11-function test suite). f(x*) = 0 at a feasible x*.
# ---------------------------------------------------------------------------------------------
_ST_Z = -2.903534027771178
def _st_raw(x): return 0.5*np.sum(x**4 - 16*x**2 + 5*x, axis=-1)
def f_styblinski(x): return _st_raw(x) - x.shape[-1]*0.5*(_ST_Z**4 - 16*_ST_Z**2 + 5*_ST_Z)
def f_alpine1(x):    return np.sum(np.abs(x*np.sin(x) + 0.1*x), axis=-1)
def f_salomon(x):
    r = np.sqrt(np.sum(x**2, axis=-1)); return 1 - np.cos(2*np.pi*r) + 0.1*r
def f_schwefel222(x): return np.sum(np.abs(x), axis=-1) + np.prod(np.abs(x), axis=-1)
_WK = np.arange(21)
def f_weierstrass(x):
    a, b = 0.5, 3.0; ak = a**_WK; bk = b**_WK; D = x.shape[-1]
    s = np.sum(ak*np.cos(2*np.pi*bk*(x[..., None] + 0.5)), axis=-1)
    return np.sum(s, axis=-1) - D*np.sum(ak*np.cos(np.pi*bk))
EXTRA_SUITE = [("StyblinskiTang", f_styblinski, -5, 5), ("Alpine1", f_alpine1, -10, 10),
               ("Salomon", f_salomon, -100, 100), ("Schwefel222", f_schwefel222, -10, 10),
               ("Weierstrass", f_weierstrass, -0.5, 0.5)]
ZSTAR["StyblinskiTang"] = lambda D: np.full(D, _ST_Z)

# ---------------------------------------------------------------------------------------------
# Benchmark wrappers
# ---------------------------------------------------------------------------------------------
def _block_ortho(D, b, seed):
    Q = np.zeros((D, D)); k = 0; i = 0
    while i < D:
        m = min(b, D - i)
        Q[i:i+m, i:i+m] = _ortho(m, seed + 31*k) if m > 1 else np.eye(1)
        i += m; k += 1
    return Q

class BlockBenchmark(Benchmark):
    """f(x) = fn(Q (x - o)), Q block-diagonal with orthogonal blocks of size b. b=1: axis-aligned; b=D: full."""
    def __init__(self, name, fn, lo, hi, D, block, rotate_seed):
        super().__init__(name, fn, lo, hi, D)
        self.Q = _block_ortho(D, block, rotate_seed) if block > 1 else np.eye(D)
        self.o = self.x_star - self.Q.T @ ZSTAR.get(name, _zstar_zero)(D)
        self.name = f"{name}-b{block}"; self.rotated = block > 1

class NoisyBenchmark:
    """Additive Gaussian noise: f(x) + sigma * spread * N(0,1). spread = std of f over the box."""
    def __init__(self, bench, sigma, seed):
        self.clean, self.sigma = bench, sigma
        self.lo, self.hi, self.D, self.name = bench.lo, bench.hi, bench.D, bench.name
        P = qmc.Sobol(d=bench.D, scramble=True, seed=7).random_base2(12)
        self.spread = float(np.std(bench(bench.lo + (bench.hi - bench.lo)*P)))
        self.rng = np.random.default_rng(seed)
    def __call__(self, X):
        y = self.clean(X)
        return y + self.sigma*self.spread*self.rng.standard_normal(y.shape) if self.sigma > 0 else y

def local_rho(bench, xb, lo, hi, D, frac, scheme):
    """Local importance at x*: forward |df|/h, central |f+ - f-|/(h+ + h-), or second difference
    |f'' estimate| on a possibly asymmetric stencil. Steps are clipped to the box."""
    lo = np.broadcast_to(np.asarray(lo, float), (D,)); hi = np.broadcast_to(np.asarray(hi, float), (D,))
    h = frac*(hi - lo); f0 = bench(xb[None, :])[0]
    Xp = np.repeat(xb[None, :], D, 0); Xm = Xp.copy()
    idx = np.arange(D)
    Xp[idx, idx] = np.minimum(xb + h, hi); Xm[idx, idx] = np.maximum(xb - h, lo)
    hp = Xp[idx, idx] - xb; hm = xb - Xm[idx, idx]
    if scheme == "forward":
        return local_perturb(bench, xb, lo, hi, D, frac)
    fp = bench(Xp); fm = bench(Xm)
    if scheme == "central":
        rho = np.abs(fp - fm)/np.maximum(hp + hm, 1e-300)
    else:
        den = hp*hm*(hp + hm)
        rho = np.where(den > 0, np.abs(2*(fp*hm + fm*hp - f0*(hp + hm)))/np.maximum(den, 1e-300), 0.0)
    s = rho.sum()
    return rho/s if s > 0 else np.full(D, 1.0/D)

# ---------------------------------------------------------------------------------------------
# P1 Noise robustness
# ---------------------------------------------------------------------------------------------
def p1_noise(cfg):
    out = _dir(cfg, "P1_noise"); D = cfg["noise_dim"]; rows = []; nulls = {}
    for sig in cfg["noise_sigmas"]:
        for name in cfg["noise_funcs"]:
            fn, lo, hi = _fn(name); clean = Benchmark(name, fn, lo, hi, D)
            b, x, STs, rhos = [], [], [], []
            for s in range(cfg["ext_seeds"]):
                seed = cfg["seed_base"] + 5100 + 7*s
                nb = NoisyBenchmark(clean, sig, stable_seed("noise-b", name, sig, s))
                nx = NoisyBenchmark(clean, sig, stable_seed("noise-x", name, sig, s))
                rb = adaguide(nb, lo, hi, D, cfg, seed)
                rx = xc_adaguide(nx, lo, hi, D, cfg, seed, diagnostics=True)
                b.append(float(clean(rb["best_x"][None, :])[0])); x.append(float(clean(rx["best_x"][None, :])[0]))
                STs.append(rx["ST"]); rhos.append(rx["rho"])
            wr = wilcoxon_report(np.array(b), np.array(x))
            cons, null = _consistency_block(STs, rhos, cfg, f"noise|{name}|{sig}")
            if null is not None: nulls.setdefault(sig, []).append(null)
            rows.append(dict(function=name, sigma=sig, base_median=float(np.median(b)), xc_median=float(np.median(x)),
                             wilcoxon_p=wr["p"], paired_dz=paired_dz(np.array(b), np.array(x)), **cons))
            print(f"   P1 sigma={sig:<6} {name:<11} r={cons['r_mean']:.3f} p={cons['perm_p']:.3g} search p={wr['p']:.3g}")
    for sig in cfg["noise_sigmas"]:
        sub = [r for r in rows if r["sigma"] == sig]; _holm_verdicts(sub)
        rm = float(np.percentile(np.concatenate(nulls[sig]), 95)) if sig in nulls else np.nan
        for r in sub:
            r["r_min"] = rm
            r["supported"] = bool(np.isfinite(r["ci_low"]) and r["ci_low"] > rm and r["perm_p"] < 0.05)
    _wcsv(os.path.join(out, "P1_noise.csv"), rows, ["function", "sigma", "base_median", "xc_median", "wilcoxon_p",
          "holm_p", "verdict", "paired_dz", "r_mean", "ci_low", "ci_high", "perm_p", "r_min", "supported"])
    summ = {str(sig): dict(supported=sum(r["supported"] for r in rows if r["sigma"] == sig),
                           mean_r=float(np.nanmean([r["r_mean"] for r in rows if r["sigma"] == sig])),
                           search={v: sum(r["verdict"] == v for r in rows if r["sigma"] == sig) for v in ("win", "loss", "tie")})
            for sig in cfg["noise_sigmas"]}
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    for name in cfg["noise_funcs"]:
        rr = [r for r in rows if r["function"] == name]
        plt.plot([max(r["sigma"], 1e-4) for r in rr], [r["r_mean"] for r in rr], marker="o", label=name, color="k",
                 ls=["-", "--", ":", "-."][cfg["noise_funcs"].index(name) % 4], alpha=0.4 + 0.6*(cfg["noise_funcs"].index(name) % 2))
    plt.xscale("log"); plt.xlabel("noise sigma / objective spread (0 plotted at 1e-4)")
    plt.ylabel("mean per-seed r(ST, rho)"); plt.title(f"Consistency under noise ({D}D)"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(out, "figP1_noise.png"), dpi=150); plt.close()
    return summ

# ---------------------------------------------------------------------------------------------
# P2 Forward / central / second difference x eta_h
# ---------------------------------------------------------------------------------------------
def p2_differences(cfg):
    out = _dir(cfg, "P2_differences"); rows = []
    for D in cfg["dims"]:
        for (name, fn, lo, hi) in SUITE:
            bench = Benchmark(name, fn, lo, hi, D); STs = []; X = []
            for s in range(cfg["ext_seeds"]):
                seed = cfg["seed_base"] + 1000*D + 7*s            # same seeds as E1 -> same optima
                rx = xc_adaguide(bench, lo, hi, D, cfg, seed, diagnostics=True)
                STs.append(rx["ST"]); X.append(rx["best_x"])
            for sch in cfg["diff_schemes"]:
                for eta in cfg["diff_etas"]:
                    rhos = [local_rho(bench, xb, lo, hi, D, eta, sch) for xb in X]
                    per = _per_seed_r(STs, rhos)
                    m, lo_, hi_ = _bootstrap_mean_ci(per, cfg["boot_iters"], stable_seed("p2d", name, D, sch, eta))
                    rows.append(dict(function=name, dim=D, scheme=sch, eta_h=eta, r_mean=m, ci_low=lo_, ci_high=hi_,
                                     undefined_seeds=int(np.sum(np.isnan(per)))))
        print(f"   P2 {D}D done")
    _wcsv(os.path.join(out, "P2_differences.csv"), rows,
          ["function", "dim", "scheme", "eta_h", "r_mean", "ci_low", "ci_high", "undefined_seeds"])
    summ = {}
    for sch in cfg["diff_schemes"]:
        for eta in cfg["diff_etas"]:
            v = [r["r_mean"] for r in rows if r["scheme"] == sch and r["eta_h"] == eta]
            summ[f"{sch}|{eta}"] = dict(mean_r=float(np.nanmean(v)), cells_r_above_0p5=int(np.sum(np.array(v) > 0.5)))
    return summ

# ---------------------------------------------------------------------------------------------
# P3 Eq. (12) stability gate
# ---------------------------------------------------------------------------------------------
def p3_gate(cfg):
    out = _dir(cfg, "P3_gate"); rows = []
    gcfg = dict(cfg, cr_gate="null", gate_alpha=cfg["gate_alpha"])
    for suite_name in ("axis", "rotated"):
        for D in cfg["dims"]:
            for k, (name, fn, lo, hi) in enumerate(SUITE):
                if suite_name == "axis":
                    bench = Benchmark(name, fn, lo, hi, D); base_seed = cfg["seed_base"] + 1000*D
                else:
                    bench = Benchmark(name, fn, lo, hi, D, rotate_seed=cfg["seed_base"] + 97*k + D)
                    base_seed = cfg["seed_base"] + 314 + 1000*D
                b, x, g, gof = [], [], [], []
                for s in range(cfg["ext_seeds"]):
                    seed = base_seed + 7*s
                    b.append(adaguide(bench, lo, hi, D, cfg, seed)["best"])
                    x.append(xc_adaguide(bench, lo, hi, D, cfg, seed)["best"])
                    rg = xc_adaguide(bench, lo, hi, D, gcfg, seed); g.append(rg["best"]); gof.append(rg["gate_open_frac"])
                b, x, g = map(np.array, (b, x, g))
                rows.append(dict(suite=suite_name, function=bench.name, dim=D,
                                 base_median=float(np.median(b)), xc_median=float(np.median(x)),
                                 gated_median=float(np.median(g)),
                                 p_xc_vs_base=wilcoxon_report(b, x)["p"], p_gated_vs_base=wilcoxon_report(b, g)["p"],
                                 p_gated_vs_xc=wilcoxon_report(x, g)["p"], gate_open_frac=float(np.mean(gof))))
            print(f"   P3 {suite_name} {D}D done")
    summ = {}
    for sn in ("axis", "rotated"):
        sub = [r for r in rows if r["suite"] == sn]
        for pk, a, bk in (("p_xc_vs_base", "base_median", "xc_median"), ("p_gated_vs_base", "base_median", "gated_median"),
                          ("p_gated_vs_xc", "xc_median", "gated_median")):
            adj = holm([r[pk] for r in sub]); cnt = dict(win=0, loss=0, tie=0)
            for r, pa in zip(sub, adj):
                r[pk.replace("p_", "holm_")] = float(pa)
                v = ("win" if r[bk] < r[a] else "loss") if pa < 0.05 else "tie"; r[pk.replace("p_", "verdict_")] = v; cnt[v] += 1
            summ[f"{sn}|{pk}"] = cnt
        summ[f"{sn}|mean_gate_open"] = float(np.mean([r["gate_open_frac"] for r in sub]))
    f = ["suite", "function", "dim", "base_median", "xc_median", "gated_median", "gate_open_frac"]
    for pk in ("xc_vs_base", "gated_vs_base", "gated_vs_xc"): f += [f"p_{pk}", f"holm_{pk}", f"verdict_{pk}"]
    _wcsv(os.path.join(out, "P3_gate.csv"), rows, f)
    return summ

# ---------------------------------------------------------------------------------------------
# P4 Complexity timing
# ---------------------------------------------------------------------------------------------
def p4_timing(cfg):
    out = _dir(cfg, "P4_timing"); rows = []
    for D in cfg["timing_dims"]:
        bench = Benchmark("Sphere", f_sphere, -100, 100, D); tm = None; gens = 0
        for s in range(cfg["timing_reps"]):
            r = xc_adaguide(bench, -100, 100, D, cfg, 900 + s)
            tm = {k: (tm[k] if tm else 0) + v for k, v in r["timing"].items()}; gens += r["generations"]
        for k, v in tm.items():
            rows.append(dict(D=D, module=f"in-loop:{k}", ms=1000*v/gens, unit="ms per generation"))
        t = time.perf_counter(); S1, ST = sobol_posthoc(bench, -100, 100, D, cfg["posthoc_N"], 1)
        rows.append(dict(D=D, module="post-hoc:Sobol", ms=1000*(time.perf_counter()-t), unit=f"ms per call, N={cfg['posthoc_N']}, {cfg['posthoc_N']*(D+2)} evals"))
        xb = bench.x_star + 1e-3
        t = time.perf_counter(); rho = local_perturb(bench, xb, -100, 100, D, 0.01)
        rows.append(dict(D=D, module="post-hoc:perturbation", ms=1000*(time.perf_counter()-t), unit=f"ms per call, {D+1} evals"))
        rng = np.random.default_rng(0); A = rng.random((cfg["seeds"], D)); B = rng.random((cfg["seeds"], D))
        t = time.perf_counter(); perm_mean_r(A, B, cfg["perm_iters"], 1)
        rows.append(dict(D=D, module="post-hoc:permutation test", ms=1000*(time.perf_counter()-t), unit=f"ms per call, B={cfg['perm_iters']}, {cfg['seeds']} seeds"))
        print(f"   P4 D={D} done")
    t = time.perf_counter(); maglev_cost(np.array([[800., 2000., 20., 100.]]))
    rows.append(dict(D=4, module="reference: one maglev objective evaluation", ms=1000*(time.perf_counter()-t), unit="ms per eval"))
    _wcsv(os.path.join(out, "P4_timing.csv"), rows, ["D", "module", "ms", "unit"])
    theory = {"in-loop:sensitivity": "O(NP*D) every G_s generations", "in-loop:variation": "O(NP*D) per generation",
              "in-loop:evaluation": "NP objective calls per generation", "in-loop:selection": "O(NP*D) per generation",
              "in-loop:revision": "O(D^2) per revision (pairwise check); O(D) without the check",
              "post-hoc:Sobol": "N*(D+2) objective calls", "post-hoc:perturbation": "D+1 (forward) or 2D+1 calls",
              "post-hoc:permutation test": "O(B*S*D)"}
    with open(os.path.join(out, "P4_complexity_theory.json"), "w") as f: json.dump(theory, f, indent=2)
    return rows

# ---------------------------------------------------------------------------------------------
# P5 Block-rotated suite
# ---------------------------------------------------------------------------------------------
def p5_block(cfg):
    out = _dir(cfg, "P5_block"); D = cfg["block_dim"]; rows = []; nulls = {}
    for bs in cfg["block_sizes"]:
        for (name, fn, lo, hi) in SUITE:
            bench = BlockBenchmark(name, fn, lo, hi, D, bs, stable_seed("block", name, bs) % 100000)
            b, x, STs, rhos = [], [], [], []
            for s in range(cfg["ext_seeds"]):
                seed = cfg["seed_base"] + 4242 + 7*s
                b.append(adaguide(bench, lo, hi, D, cfg, seed)["best"])
                rx = xc_adaguide(bench, lo, hi, D, cfg, seed, diagnostics=True)
                x.append(rx["best"]); STs.append(rx["ST"]); rhos.append(rx["rho"])
            wr = wilcoxon_report(np.array(b), np.array(x))
            cons, null = _consistency_block(STs, rhos, cfg, f"block|{name}|{bs}")
            if null is not None: nulls.setdefault(bs, []).append(null)
            rows.append(dict(block=bs, function=name, base_median=float(np.median(b)), xc_median=float(np.median(x)),
                             improvement_pct=float(100*(np.median(b)-np.median(x))/(abs(np.median(b))+1e-300)),
                             wilcoxon_p=wr["p"], **cons))
        print(f"   P5 block={bs} done")
    summ = {}
    for bs in cfg["block_sizes"]:
        sub = [r for r in rows if r["block"] == bs]; cnt = _holm_verdicts(sub)
        rm = float(np.percentile(np.concatenate(nulls[bs]), 95)) if bs in nulls else np.nan
        for r in sub:
            r["r_min"] = rm; r["supported"] = bool(np.isfinite(r["ci_low"]) and r["ci_low"] > rm and r["perm_p"] < 0.05)
        summ[str(bs)] = dict(search=cnt, supported=sum(r["supported"] for r in sub),
                             mean_r=float(np.nanmean([r["r_mean"] for r in sub])),
                             xc_better_median=sum(r["xc_median"] < r["base_median"] for r in sub))
    _wcsv(os.path.join(out, "P5_block.csv"), rows, ["block", "function", "base_median", "xc_median", "improvement_pct",
          "wilcoxon_p", "holm_p", "verdict", "r_mean", "ci_low", "ci_high", "perm_p", "r_min", "supported"])
    return summ

# ---------------------------------------------------------------------------------------------
# P6 Quadrotor attitude: rigid body + Euler kinematics, cascaded angle-P / rate-PID per axis
# ---------------------------------------------------------------------------------------------
QUAD = dict(I=np.array([0.0082, 0.0082, 0.0149]), tau_max=np.array([0.8, 0.8, 0.2]), tau_m=0.02,
            rate_max=np.array([4.0, 4.0, 2.0]), Nf=100.0, T=2.5, dt=0.002,
            ref_deg=np.array([20.0, -15.0, 30.0]), dist=np.array([0.15, -0.15, 0.03]), t_dist=(1.5, 1.6),
            w_axis=np.array([1.0, 1.0, 0.5]), w_os=0.5, w_u=20.0, fail_cost=1e3, fail_deg=80.0)
QUAD_NAMES = [f"{g}_{ax}" for ax in ("roll", "pitch", "yaw") for g in ("Ka", "Kp", "Ki", "Kd")]
QUAD_BOUNDS = (np.tile([0.5, 0.0, 0.0, 0.0], 3),
               np.array([15, 0.30, 0.50, 0.010, 15, 0.30, 0.50, 0.010, 10, 0.50, 0.50, 0.010], float))

def quad_cost(G, P=QUAD, return_parts=False, return_trace=False):
    """J = sum_axis w_i*ITAE_i[deg s^2] + w_os*overshoot[deg] + w_u*effort[(N m)^2 s]; J = 1e3 if |roll| or |pitch| > 80 deg.
    Semi-implicit Euler, dt = 2 ms. Derivative on measurement with first-order filter (N = 100 rad/s)."""
    G = np.atleast_2d(np.asarray(G, float)); n = G.shape[0]
    Ka = G[:, [0, 4, 8]]; Kp = G[:, [1, 5, 9]]; Ki = G[:, [2, 6, 10]]; Kd = G[:, [3, 7, 11]]
    I = P["I"]; dt = P["dt"]; steps = int(P["T"]/dt); ref = np.deg2rad(P["ref_deg"])
    ang = np.zeros((n, 3)); w = np.zeros((n, 3)); tau = np.zeros((n, 3)); integ = np.zeros((n, 3)); xf = np.zeros((n, 3))
    itae = np.zeros((n, 3)); effort = np.zeros(n); os_ = np.zeros((n, 3)); alive = np.ones(n, bool)
    tr = np.zeros((steps, 3)) if return_trace else None
    sgn = np.sign(ref)
    with np.errstate(all="ignore"):
        for k in range(steps):
            t = k*dt
            wcmd = np.clip(Ka*(ref - ang), -P["rate_max"], P["rate_max"])
            er = wcmd - w
            integ = np.clip(integ + er*dt, -0.5, 0.5)
            dmeas = P["Nf"]*(w - xf); xf = xf + dt*dmeas
            u = np.clip(Kp*er + Ki*integ - Kd*dmeas, -P["tau_max"], P["tau_max"])
            tau = tau + dt*(u - tau)/P["tau_m"]
            td = P["dist"] if P["t_dist"][0] <= t < P["t_dist"][1] else 0.0
            Iw = w*I; gyro = np.cross(w, Iw)
            w = w + dt*(tau + td - gyro)/I
            ph, th = ang[:, 0], ang[:, 1]; p, q, r = w[:, 0], w[:, 1], w[:, 2]
            ct = np.cos(th); ct = np.where(np.abs(ct) < 1e-3, 1e-3, ct)
            dang = np.stack([p + (q*np.sin(ph) + r*np.cos(ph))*np.tan(th), q*np.cos(ph) - r*np.sin(ph),
                             (q*np.sin(ph) + r*np.cos(ph))/ct], 1)
            ang = ang + dt*dang
            bad = (~np.all(np.isfinite(ang), 1)) | (np.abs(np.rad2deg(ang[:, :2])).max(1) > P["fail_deg"])
            alive &= ~bad
            e_deg = np.where(alive[:, None], np.rad2deg(np.abs(ref - ang)), 0.0)
            itae += t*e_deg*dt; effort += np.where(alive, np.sum(u**2, 1)*dt, 0.0)
            os_ = np.maximum(os_, np.where(alive[:, None], np.rad2deg(sgn*(ang - ref)), 0.0))
            if return_trace: tr[k] = np.rad2deg(ang[0])
    J = np.where(alive, itae @ P["w_axis"] + P["w_os"]*os_.sum(1) + P["w_u"]*effort, P["fail_cost"])
    J = np.where(np.isfinite(J), J, P["fail_cost"])
    if return_trace: return J, tr, np.arange(steps)*dt
    if return_parts: return J, dict(itae=itae.sum(1), overshoot=os_.sum(1), effort=effort, alive=alive)
    return J

class QuadPlant:
    D = 12; name = "Quadrotor-attitude"; lo, hi = QUAD_BOUNDS
    def __call__(self, X): return quad_cost(X)

def p6_quadrotor(cfg):
    out = _dir(cfg, "P6_quadrotor"); plant = QuadPlant(); D = plant.D; lo, hi = plant.lo, plant.hi
    qc = dict(cfg, NP=cfg["quad_NP"], posthoc_N=cfg["quad_posthoc_N"])
    b, x, xb_b, xb_x, STs, S1s, rhos = [], [], [], [], [], [], []
    for s in range(cfg["quad_seeds"]):
        seed = cfg["seed_base"] + 12000 + 13*s
        rb = adaguide(plant, lo, hi, D, qc, seed, maxfe=cfg["quad_fe"])
        rx = xc_adaguide(plant, lo, hi, D, qc, seed, diagnostics=True, maxfe=cfg["quad_fe"])
        b.append(rb["best"]); x.append(rx["best"]); xb_b.append(rb["best_x"]); xb_x.append(rx["best_x"])
        STs.append(rx["ST"]); S1s.append(rx["S1"]); rhos.append(rx["rho"])
        print(f"   P6 seed {s}: base J {rb['best']:.4g}  xc J {rx['best']:.4g}")
    b, x = np.array(b), np.array(x); wr = wilcoxon_report(b, x)
    cons, null = _consistency_block(STs, rhos, cfg, "quad")
    r_min = float(np.percentile(null, 95)) if null is not None else np.nan
    ST = np.mean(STs, 0); S1 = np.mean(S1s, 0); rho = np.mean(rhos, 0)
    Pq = qmc.Sobol(d=D, scramble=True, seed=3).random_base2(10); _, parts = quad_cost(lo + (hi-lo)*Pq, return_parts=True)
    res = dict(D=D, names=QUAD_NAMES, seeds=cfg["quad_seeds"], fe=cfg["quad_fe"], NP=cfg["quad_NP"],
               objective="J = sum w_i ITAE_i + 0.5*overshoot + 20*effort (fail -> 1e3)",
               base_median=float(np.median(b)), xc_median=float(np.median(x)), wilcoxon_p=wr["p"],
               wilcoxon_method=wr["method"], paired_dz=paired_dz(b, x), cliffs_delta=cliffs_delta(b, x),
               S1=S1.tolist(), ST=ST.tolist(), rho=rho.tolist(), consistency=cons, r_min_D12=r_min,
               supported=bool(np.isfinite(cons["ci_low"]) and cons["ci_low"] > r_min and cons["perm_p"] < 0.05),
               corr_perseed=_per_seed_r(STs, rhos), sobol_design_fail_fraction=float(1 - parts["alive"].mean()),
               base_best=b.tolist(), xc_best=x.tolist(), xc_gains_seed0=np.asarray(xb_x[0]).tolist())
    with open(os.path.join(out, "P6_quadrotor.json"), "w") as f: json.dump(res, f, indent=2, default=float)
    import matplotlib.pyplot as plt
    J, tr, t = quad_cost(np.asarray(xb_x[0])[None, :], return_trace=True)
    plt.figure(figsize=(8, 4.2))
    for i, (ax, ls) in enumerate(zip(("roll", "pitch", "yaw"), ("-", "--", ":"))):
        plt.plot(t, tr[:, i], color="k", ls=ls, label=ax); plt.axhline(QUAD["ref_deg"][i], color="0.6", lw=0.8)
    plt.axvspan(*QUAD["t_dist"], color="0.9"); plt.xlabel("time (s)"); plt.ylabel("angle (deg)")
    plt.title(f"Quadrotor attitude, XC-AdaGuiDE seed 0 (J {J[0]:.3g}); shaded = torque disturbance")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out, "figQ1_response.png"), dpi=150); plt.close()
    plt.figure(figsize=(6, 5)); plt.scatter(ST, rho, color="k")
    for j, nm in enumerate(QUAD_NAMES): plt.annotate(nm, (ST[j], rho[j]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    plt.xlabel("Sobol total-order ST"); plt.ylabel("perturbation importance rho")
    plt.title(f"12-gain quadrotor: mean r = {cons['r_mean']:.3f} [{cons['ci_low']:.2f}, {cons['ci_high']:.2f}]")
    plt.tight_layout(); plt.savefig(os.path.join(out, "figQ2_consistency.png"), dpi=150); plt.close()
    return {k: res[k] for k in ("base_median", "xc_median", "wilcoxon_p", "consistency", "r_min_D12", "supported",
                                "sobol_design_fail_fraction")}

# ---------------------------------------------------------------------------------------------
# P7 Separate tuning set
# ---------------------------------------------------------------------------------------------
def p7_tuning(cfg):
    out = _dir(cfg, "P7_tuning")
    for D in (10, 30):
        for (n, f, a, b) in EXTRA_SUITE:
            ok, fx, inside = Benchmark(n, f, a, b, D).check(tol=1e-6)
            assert ok, f"tuning-set optimum check failed: {n} {D}D f={fx} inside={inside}"
    c = dict(cfg, dims=[30], sweep_seeds=cfg["ext_seeds"], sweep_funcs=cfg["tune_funcs"], extra_suite=EXTRA_SUITE,
             sweep_grid=cfg["tune_grid_fixed"])
    rows = run_param_sweep(c, SUITE, out)
    if cfg["tune_run_cec"]:
        oc = _dir(cfg, "P7_tuning_cec")
        rows += run_param_sweep(dict(c, budget_mode="cec", sweep_grid=cfg["tune_grid_cec"]), SUITE, oc)
    return rows

# ---------------------------------------------------------------------------------------------
# P8 Bound revision on / off at 10000*D
# ---------------------------------------------------------------------------------------------
def p8_onoff(cfg):
    out = _dir(cfg, "P8_onoff"); D = 30; c = dict(cfg, budget_mode="cec"); rows = []; raw = []
    for name in cfg["onoff_funcs"]:
        fn, lo, hi = _fn(name); bench = Benchmark(name, fn, lo, hi, D); on, off, rev = [], [], []
        for s in range(cfg["onoff_seeds"]):
            seed = cfg["seed_base"] + 8800 + 7*s
            ron = guided_de(bench, lo, hi, D, c, seed, sens_cross=True, sens_bound=True, boundary=True)
            roff = guided_de(bench, lo, hi, D, c, seed, sens_cross=True, sens_bound=False, boundary=False)
            on.append(ron["best"]); off.append(roff["best"]); rev.append(ron["property1"]["revisions"])
            raw.append(dict(function=name, seed=s, with_revision=ron["best"], without_revision=roff["best"],
                            revisions=ron["property1"]["revisions"]))
        on, off = np.array(on), np.array(off); wr = wilcoxon_report(off, on)
        rows.append(dict(function=name, base_median=float(np.median(off)), xc_median=float(np.median(on)),
                         wilcoxon_p=wr["p"], paired_dz=paired_dz(off, on), cliffs_delta=cliffs_delta(off, on),
                         revisions_per_run=float(np.mean(rev)), tied_seeds=int(np.sum(on == off))))
        print(f"   P8 {name}: without {np.median(off):.4g}  with {np.median(on):.4g}  p={wr['p']:.3g}")
    cnt = _holm_verdicts(rows)
    allon = np.array([r["with_revision"] for r in raw]); alloff = np.array([r["without_revision"] for r in raw])
    pooled = wilcoxon_report(_log10f(alloff), _log10f(allon))
    for r in rows: r["without_revision_median"] = r.pop("base_median"); r["with_revision_median"] = r.pop("xc_median")
    _wcsv(os.path.join(out, "P8_onoff.csv"), rows, ["function", "without_revision_median", "with_revision_median",
          "wilcoxon_p", "holm_p", "verdict", "paired_dz", "cliffs_delta", "revisions_per_run", "tied_seeds"])
    _wcsv(os.path.join(out, "P8_onoff_raw.csv"), raw, ["function", "seed", "without_revision", "with_revision", "revisions"])
    return dict(per_function=cnt, pooled_wilcoxon_p=pooled["p"], pooled_n=pooled["n"])

# ---------------------------------------------------------------------------------------------
def ext_selftest(cfg):
    for D in (10, 30):
        for (n, f, a, b) in EXTRA_SUITE:
            ok, fx, _ = Benchmark(n, f, a, b, D).check(tol=1e-6); assert ok, (n, D, fx)
        for bs in (5, 10):
            Q = _block_ortho(D, bs, 1); assert np.allclose(Q @ Q.T, np.eye(D))
            for (n, f, a, b) in SUITE:
                ok, fx, _ = BlockBenchmark(n, f, a, b, D, bs, 11).check(); assert ok, (n, D, bs, fx)
    ref = np.array([[6, 0.08, 0.2, 0.002]*2 + [4, 0.2, 0.2, 0.002]], float)
    J, parts = quad_cost(ref, return_parts=True)
    assert parts["alive"][0] and J[0] < 100, f"reference quadrotor gains unstable: J={J[0]}"
    return dict(tuning_set_ok=True, block_rotation_ok=True, quad_reference_J=float(J[0]))

def zip_extended(cfg, zipname="xc_adaguide_extended_results.zip"):
    import zipfile
    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(cfg["ext_outroot"]):
            for fn in files: z.write(os.path.join(root, fn))
    return zipname
