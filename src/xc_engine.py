"""
XC-AdaGuiDE engine.

Benchmark suite (axis-aligned and rotated, feasible shifted optimum), five DE variants
(classic DE, JADE, L-SHADE, AdaGuiDE, XC-AdaGuiDE), post-run Sobol indices (centred
Saltelli S1, Jansen ST), forward-difference local importance, the global-local
consistency test, Friedman / Nemenyi / Wilcoxon / Holm statistics, the maglev PID case
study and the figures.

Optional switches (off by default, results unchanged when off):
  cfg["cr_gate"] = "null"   apply the CR shift only when max |corr(x_j, f)| exceeds a
                            Bonferroni null threshold (Eq. 12 gate)
  out["timing"]             wall-clock time per module
  run_param_sweep           one-at-a-time sweep of kappa, gamma, beta0, delta, G, G_s

See CHANGELOG.md for the version history.
"""

import os, json, time, zipfile, platform, zlib
import numpy as np
import scipy
from scipy import stats
from scipy.stats import qmc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 0. Configuration
# --------------------------------------------------------------------------
CFG = dict(
    seeds=15, abl_seeds=30, dims=[10, 30], NP=50,
    budget_mode="fixed",      # "fixed" -> max_fe ; "cec" -> fe_per_dim * D
    max_fe=12000, fe_per_dim=10000,
    p_best=0.11, F_min=0.05, p_min=0.05,
    gamma=0.30, beta0=0.30, kappa=0.20, decay_frac=0.5, eta=0.02,
    G_trigger=20, sens_every=5, eta_h=0.01,
    posthoc_N=1024,
    r_min_fallback=0.30, perm_iters=10000, boot_iters=5000, cv_gate=0.10,
    run_plant=True, plant_seeds=30, plant_NP=30, plant_fe=4000, plant_posthoc_N=512,
    run_rotated=True, rot_seeds=15,
    run_sobol_sweep=True, sweep_fn="Ackley", sweep_N=[256, 512, 1024, 2048, 4096], sweep_reps=20,
    run_param_sweep=False, sweep_funcs=["Rosenbrock", "Ellipsoid", "Discus", "Griewank", "Levy"],
    sweep_seeds=15, sweep_grid=dict(kappa=[0.0, 0.1, 0.2, 0.3, 0.5], gamma=[0.0, 0.15, 0.3, 0.6],
                                    beta0=[0.1, 0.2, 0.3, 0.5], decay_frac=[0.25, 0.5, 0.75, 1.0],
                                    G_trigger=[10, 20, 40], sens_every=[1, 5, 10]),
    seed_base=20260608, outdir="results",
)

def budget(cfg, D):
    return int(cfg["fe_per_dim"]*D) if cfg.get("budget_mode") == "cec" else int(cfg["max_fe"])

def stable_seed(*parts):
    """Process-independent integer seed (Python hash() is salted per process)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2**31)

# --------------------------------------------------------------------------
# 1. Benchmark suite with a feasible shifted optimum
# --------------------------------------------------------------------------
def f_sphere(x):     return np.sum(x**2, axis=-1)
def f_rastrigin(x):  return 10*x.shape[-1] + np.sum(x**2 - 10*np.cos(2*np.pi*x), axis=-1)
def f_rosenbrock(x): return np.sum(100*(x[..., 1:]-x[..., :-1]**2)**2 + (1-x[..., :-1])**2, axis=-1)
def f_ackley(x):
    D = x.shape[-1]
    return (-20*np.exp(-0.2*np.sqrt(np.sum(x**2, axis=-1)/D))
            - np.exp(np.sum(np.cos(2*np.pi*x), axis=-1)/D) + 20 + np.e)
def f_griewank(x):
    i = np.arange(1, x.shape[-1]+1)
    return np.sum(x**2, axis=-1)/4000 - np.prod(np.cos(x/np.sqrt(i)), axis=-1) + 1
def f_levy(x):
    w = 1 + (x-1)/4
    t1 = np.sin(np.pi*w[..., 0])**2
    t3 = (w[..., -1]-1)**2*(1+np.sin(2*np.pi*w[..., -1])**2)
    mid = np.sum((w[..., :-1]-1)**2*(1+10*np.sin(np.pi*w[..., :-1]+1)**2), axis=-1)
    return t1 + mid + t3
def f_zakharov(x):
    i = np.arange(1, x.shape[-1]+1); s2 = np.sum(0.5*i*x, axis=-1)
    return np.sum(x**2, axis=-1) + s2**2 + s2**4
def f_dixonprice(x):
    i = np.arange(2, x.shape[-1]+1)
    return (x[..., 0]-1)**2 + np.sum(i*(2*x[..., 1:]**2 - x[..., :-1])**2, axis=-1)
def f_ellipsoid(x):
    D = x.shape[-1]; w = 10.0**(6*np.arange(D)/max(D-1, 1))
    return np.sum(w*x**2, axis=-1)
def f_bentcigar(x): return x[..., 0]**2 + 1e6*np.sum(x[..., 1:]**2, axis=-1)
def f_discus(x):    return 1e6*x[..., 0]**2 + np.sum(x[..., 1:]**2, axis=-1)

def _zstar_zero(D): return np.zeros(D)
def _zstar_one(D):  return np.ones(D)
def _zstar_dixon(D):
    i = np.arange(1, D+1); return 2.0**(-(2.0**i - 2)/2.0**i)

SUITE = [
    ("Sphere",     f_sphere,     -100, 100),
    ("Rastrigin",  f_rastrigin,  -5.12, 5.12),
    ("Rosenbrock", f_rosenbrock, -30, 30),
    ("Ackley",     f_ackley,     -32, 32),
    ("Griewank",   f_griewank,   -600, 600),
    ("Levy",       f_levy,       -10, 10),
    ("Zakharov",   f_zakharov,   -10, 10),
    ("DixonPrice", f_dixonprice, -10, 10),
    ("Ellipsoid",  f_ellipsoid,  -100, 100),
    ("BentCigar",  f_bentcigar,  -100, 100),
    ("Discus",     f_discus,     -100, 100),
]
ZSTAR = {"Rosenbrock": _zstar_one, "Levy": _zstar_one, "DixonPrice": _zstar_dixon}

def _ortho(D, seed):
    A = np.random.default_rng(seed).standard_normal((D, D))
    Q, R = np.linalg.qr(A); Q *= np.sign(np.diag(R)); return Q

class Benchmark:
    """f(x) = fn(Q (x - o)). Q = I for the axis-aligned suite.
    The optimum x* is drawn inside 80% of the box; o is set so that Q (x* - o) = z*."""
    def __init__(self, name, fn, lo, hi, D, rotate_seed=None, shift_seed=12345):
        self.base_name, self.fn, self.lo, self.hi, self.D = name, fn, lo, hi, D
        self.rotated = rotate_seed is not None
        self.name = name + ("-rot" if self.rotated else "")
        self.Q = _ortho(D, rotate_seed) if self.rotated else np.eye(D)
        rng = np.random.default_rng(stable_seed("shift", name, D, shift_seed))
        self.x_star = rng.uniform(0.8*lo, 0.8*hi, size=D)
        z_star = ZSTAR.get(name, _zstar_zero)(D)
        self.o = self.x_star - self.Q.T @ z_star
    def __call__(self, X):
        z = (X - self.o) @ self.Q.T
        return self.fn(z)
    def check(self, tol=1e-8):
        """Feasible, global optimum value ~ 0 (Rosenbrock/Levy/Dixon-Price included)."""
        fx = float(self(self.x_star[None, :])[0])
        inside = bool(np.all((self.x_star >= self.lo) & (self.x_star <= self.hi)))
        return inside and abs(fx) < tol, fx, inside

# --------------------------------------------------------------------------
# 2. Sensitivity: zero-cost in-loop proxy, post-hoc Sobol, local perturbation
# --------------------------------------------------------------------------
def pop_sensitivity(pop, fit):
    """Zero-cost per-dimension weight from |Pearson corr(x_j, f)| over the live population."""
    f = fit - fit.mean(); fs = f.std() + 1e-12
    xj = pop - pop.mean(0); sj = xj.std(0) + 1e-12
    w = np.abs((xj*f[:, None]).mean(0)/(sj*fs))
    D = pop.shape[1]
    return np.full(D, 1.0/D) if w.sum() < 1e-12 else w/w.sum()

def pop_abs_corr(pop, fit):
    """Raw |Pearson corr(x_j, f)| per dimension (used by the optional Eq. 12 gate)."""
    f = fit - fit.mean(); fs = f.std() + 1e-12
    xj = pop - pop.mean(0); sj = xj.std(0) + 1e-12
    return np.abs((xj*f[:, None]).mean(0)/(sj*fs))

def sobol_posthoc(bench, lo, hi, D, N, seed, return_se=False):
    """Sobol S1 (Saltelli 2010) and ST (Jansen 1999) on a scrambled Sobol' design.
    Outputs are centred before estimation. S1 is returned unclipped."""
    lo = np.broadcast_to(np.asarray(lo, float), (D,)); hi = np.broadcast_to(np.asarray(hi, float), (D,))
    m = int(np.ceil(np.log2(max(N, 2))))
    P = qmc.Sobol(d=2*D, scramble=True, seed=seed).random_base2(m=m)
    A = lo + (hi-lo)*P[:, :D]; B = lo + (hi-lo)*P[:, D:]
    yA = bench(A); yB = bench(B)
    mu = np.mean(np.concatenate([yA, yB])); yA = yA - mu; yB = yB - mu
    varY = np.var(np.concatenate([yA, yB]))
    if varY < 1e-300:
        z = np.full(D, 1.0/D); return (z, z, np.zeros(D)) if return_se else (z, z)
    S1 = np.zeros(D); ST = np.zeros(D); STse = np.zeros(D)
    for j in range(D):
        AB = A.copy(); AB[:, j] = B[:, j]; yAB = bench(AB) - mu
        S1[j] = np.mean(yB*(yAB - yA))/varY
        t = 0.5*(yA - yAB)**2/varY
        ST[j] = t.mean(); STse[j] = t.std(ddof=1)/np.sqrt(len(t))
    return (S1, ST, STse) if return_se else (S1, ST)

def local_perturb(bench, xb, lo, hi, D, frac=0.01):
    """rho_j = |f(x* + h e_j) - f(x*)| / |h|, forward difference. At an upper bound the
    step is taken inward, and the actual displacement is the divisor."""
    lo = np.broadcast_to(np.asarray(lo, float), (D,)); hi = np.broadcast_to(np.asarray(hi, float), (D,))
    step = frac*(hi - lo); base = bench(xb[None, :])[0]; rho = np.zeros(D)
    X = np.repeat(xb[None, :], D, axis=0); dx = np.zeros(D)
    for j in range(D):
        xn = xb[j] + step[j]
        if xn > hi[j]: xn = xb[j] - step[j]
        X[j, j] = np.clip(xn, lo[j], hi[j]); dx[j] = abs(X[j, j] - xb[j])
    fX = bench(X)
    rho = np.where(dx > 0, np.abs(fX - base)/np.maximum(dx, 1e-300), 0.0)
    s = rho.sum()
    return rho/s if s > 0 else np.full(D, 1.0/D)

# --------------------------------------------------------------------------
# 3. Optimizers. Same seed -> same initial population for the NP=50 methods.
# --------------------------------------------------------------------------
def _rand_idx(NP, k, rng):
    """k distinct indices per row, each excluding the row's own index."""
    keys = rng.random((NP, NP)); keys[np.arange(NP), np.arange(NP)] = -1.0
    return np.argsort(keys, axis=1)[:, 1:k+1]

def _archive_idx(NP, n_pa, r1, rng):
    """Index into pop U archive, distinct from i and r1."""
    idx = rng.integers(n_pa, size=NP); i = np.arange(NP)
    bad = (idx == i) | (idx == r1)
    while bad.any():
        idx[bad] = rng.integers(n_pa, size=int(bad.sum()))
        bad = (idx == i) | (idx == r1)
    return idx

def _crossover(pop, V, CRmat, rng):
    NP, D = pop.shape
    mask = rng.random((NP, D)) < CRmat
    mask[np.arange(NP), rng.integers(D, size=NP)] = True
    return np.where(mask, V, pop)

def _bounds(lo, hi, D):
    return (np.broadcast_to(np.asarray(lo, float), (D,)).copy(),
            np.broadcast_to(np.asarray(hi, float), (D,)).copy())

def de_classic(bench, lo, hi, D, cfg, seed):
    """Classic DE/rand/1/bin, F = 0.5, CR = 0.9."""
    rng = np.random.default_rng(seed); NP, maxfe, F, CR = cfg["NP"], budget(cfg, D), 0.5, 0.9
    L, U = _bounds(lo, hi, D)
    pop = rng.uniform(L, U, (NP, D)); fit = bench(pop); fe = NP; hist = [fit.min()]
    while fe < maxfe:
        r = _rand_idx(NP, 3, rng)
        V = np.clip(pop[r[:, 0]] + F*(pop[r[:, 1]] - pop[r[:, 2]]), L, U)
        Um = _crossover(pop, V, np.full((NP, D), CR), rng); fu = bench(Um); fe += NP
        imp = fu < fit; pop[imp] = Um[imp]; fit[imp] = fu[imp]; hist.append(fit.min())
    return dict(best=float(fit.min()), history=np.array(hist), best_x=pop[np.argmin(fit)], fe=fe)

def jade(bench, lo, hi, D, cfg, seed):
    """JADE: current-to-pbest/1 + archive, Cauchy F (Lehmer mean), Gaussian CR."""
    rng = np.random.default_rng(seed); NP, maxfe, p, c = cfg["NP"], budget(cfg, D), cfg["p_best"], 0.1
    L, U = _bounds(lo, hi, D)
    pop = rng.uniform(L, U, (NP, D)); fit = bench(pop); fe = NP; hist = [fit.min()]
    muF, muCR = 0.5, 0.5; archive = []
    while fe < maxfe:
        F = np.clip(stats.cauchy.rvs(muF, 0.1, size=NP, random_state=rng), 1e-3, 1.0)
        CR = np.clip(rng.normal(muCR, 0.1, NP), 0.0, 1.0)
        pc = max(2, int(p*NP)); pbest = np.argsort(fit)[:pc][rng.integers(pc, size=NP)]
        PA = np.vstack([pop] + archive) if archive else pop
        r1 = _rand_idx(NP, 1, rng)[:, 0]; r2 = _archive_idx(NP, PA.shape[0], r1, rng)
        V = np.clip(pop + F[:, None]*(pop[pbest]-pop) + F[:, None]*(pop[r1]-PA[r2]), L, U)
        Um = _crossover(pop, V, np.repeat(CR[:, None], D, 1), rng); fu = bench(Um); fe += NP
        imp = fu < fit
        if imp.any():
            archive.extend(pop[idx].copy() for idx in np.where(imp)[0])
            sF, sCR = F[imp], CR[imp]
            muF = (1-c)*muF + c*(np.sum(sF**2)/(np.sum(sF)+1e-12))
            muCR = (1-c)*muCR + c*np.mean(sCR)
            pop[imp] = Um[imp]; fit[imp] = fu[imp]
        while len(archive) > NP: archive.pop(rng.integers(len(archive)))
        hist.append(fit.min())
    return dict(best=float(fit.min()), history=np.array(hist), best_x=pop[np.argmin(fit)], fe=fe)

def lshade(bench, lo, hi, D, cfg, seed):
    """L-SHADE: SHADE memory (H=6) + linear population-size reduction + archive (rate 2.6).
    Initial population = min(18 D, 400); it is NOT shared with the NP=50 methods."""
    rng = np.random.default_rng(seed); maxfe, p = budget(cfg, D), cfg["p_best"]
    H = 6; Ninit = min(18*D, 400); Nmin = 4; arc_rate = 2.6
    MF = np.full(H, 0.5); MCR = np.full(H, 0.5); k = 0
    L, U = _bounds(lo, hi, D)
    NP = Ninit; pop = rng.uniform(L, U, (NP, D)); fit = bench(pop); fe = NP; hist = [fit.min()]; archive = []
    while fe < maxfe:
        ri = rng.integers(H, size=NP)
        F = np.clip(stats.cauchy.rvs(MF[ri], 0.1, size=NP, random_state=rng), 1e-3, 1.0)
        CR = np.clip(rng.normal(MCR[ri], 0.1, NP), 0.0, 1.0)
        pc = max(2, int(p*NP)); pbest = np.argsort(fit)[:pc][rng.integers(pc, size=NP)]
        PA = np.vstack([pop] + archive) if archive else pop
        r1 = _rand_idx(NP, 1, rng)[:, 0]; r2 = _archive_idx(NP, PA.shape[0], r1, rng)
        V = np.clip(pop + F[:, None]*(pop[pbest]-pop) + F[:, None]*(pop[r1]-PA[r2]), L, U)
        Um = _crossover(pop, V, np.repeat(CR[:, None], D, 1), rng); fu = bench(Um); fe += NP
        imp = fu < fit
        if imp.any():
            wg = np.abs(fit[imp]-fu[imp]); wg = wg/(wg.sum()+1e-12); sF, sCR = F[imp], CR[imp]
            MF[k] = np.sum(wg*sF**2)/(np.sum(wg*sF)+1e-12)
            MCR[k] = np.sum(wg*sCR**2)/(np.sum(wg*sCR)+1e-12) if np.sum(wg*sCR) > 0 else 0.0
            k = (k+1) % H
            archive.extend(pop[idx].copy() for idx in np.where(imp)[0])
            pop[imp] = Um[imp]; fit[imp] = fu[imp]
        Nn = max(Nmin, int(round(Ninit + (Nmin-Ninit)*fe/maxfe)))
        if Nn < NP: keep = np.argsort(fit)[:Nn]; pop = pop[keep]; fit = fit[keep]; NP = Nn
        amax = int(round(arc_rate*NP))
        while len(archive) > amax: archive.pop(rng.integers(len(archive)))
        hist.append(fit.min())
    return dict(best=float(fit.min()), history=np.array(hist), best_x=pop[np.argmin(fit)], fe=fe)

def _property1_check(L, U, L0, U0, xb, width):
    """Property 1 on the actual clipped interval.
    A dimension is interior when L0 + Delta <= x* <= U0 - Delta (paper condition).
    For every interior pair (a, b) with s_a >= s_b, phi_a >= phi_b must hold."""
    phi = (U - L)/(U0 - L0)
    interior = (xb - width >= L0 - 1e-12) & (xb + width <= U0 + 1e-12)
    return phi, interior

def guided_de(bench, lo, hi, D, cfg, seed, sens_cross=False, sens_bound=False,
              boundary=True, with_reward=False, diagnostics=False, maxfe=None):
    """Shared core. AdaGuiDE-style base: sens_cross=sens_bound=False, boundary=True.
    XC-AdaGuiDE: sens_cross=sens_bound=True, boundary=True."""
    rng = np.random.default_rng(seed)
    NP, p = cfg["NP"], cfg["p_best"]; maxfe = budget(cfg, D) if maxfe is None else maxfe
    L, U = _bounds(lo, hi, D); L0, U0 = L.copy(), U.copy()
    pop = rng.uniform(L, U, (NP, D)); fit = bench(pop); fe = NP; hist = [fit.min()]
    muF, muCR = 0.5, 0.9; archive = []; CS = np.ones(3); pmin = cfg.get("p_min", 0.0)
    stagn = 0; best_prev = fit.min(); w = np.full(D, 1.0/D); gen = 0
    p1 = dict(revisions=0, pairs_checked=0, pairs_ok=0, dims_interior=0, dims_total=0)
    # optional Eq. (12) gate (off by default)
    gate = cfg.get("cr_gate", None); gate_open = True; n_ref = 0; n_open = 0; n_flat = 0
    if gate == "null":
        gate_crit = stats.norm.ppf(1 - cfg.get("gate_alpha", 0.05)/(2*D))/np.sqrt(max(NP - 1, 2))
    tm = dict(sensitivity=0.0, variation=0.0, evaluation=0.0, selection=0.0, revision=0.0)
    _pc = time.perf_counter
    while fe < maxfe:
        gen += 1
        decay = max(0.0, 1.0 - fe/(cfg["decay_frac"]*maxfe))
        t0 = _pc()
        if (sens_cross or sens_bound) and (gen-1) % cfg["sens_every"] == 0:
            w = pop_sensitivity(pop, fit)
            if gate == "null":   # open only if some |corr(x_j, f)| beats a Bonferroni null threshold
                gate_open = bool(np.max(pop_abs_corr(pop, fit)) > gate_crit)
                n_ref += 1; n_open += int(gate_open)
        spread_w = w.max() - w.min()                       # Eq. (12); w_hat = 0.5 if all weights coincide
        wn = (w - w.min())/spread_w if spread_w > 1e-12 else np.full(D, 0.5)
        if spread_w <= 1e-12: n_flat += 1
        if gate == "null" and not gate_open:
            wn = np.full(D, 0.5)
        tm["sensitivity"] += _pc() - t0; t0 = _pc()
        F = np.clip(stats.cauchy.rvs(muF, 0.1, size=NP, random_state=rng), cfg.get("F_min", 0.05), 1.0)
        CR = np.clip(rng.normal(muCR, 0.1, NP), 0.0, 1.0)
        SP = pmin + (1 - 3*pmin)*CS/CS.sum()
        strat = rng.choice(3, size=NP, p=SP/SP.sum())
        pc = max(2, int(p*NP)); pbest = np.argsort(fit)[:pc][rng.integers(pc, size=NP)]
        PA = np.vstack([pop] + archive) if archive else pop
        r = _rand_idx(NP, 3, rng); ra = _archive_idx(NP, PA.shape[0], r[:, 0], rng)
        V = np.empty_like(pop); m0 = strat == 0; m1 = strat == 1; m2 = strat == 2
        V[m0] = pop[m0] + F[m0, None]*(pop[pbest][m0]-pop[m0]) + F[m0, None]*(pop[r[:, 0]][m0]-PA[ra][m0])
        V[m1] = pop[r[:, 0]][m1] + F[m1, None]*(pop[r[:, 1]][m1]-pop[r[:, 2]][m1])
        V[m2] = pop[m2] + F[m2, None]*(pop[r[:, 0]][m2]-pop[m2]) + F[m2, None]*(pop[r[:, 1]][m2]-pop[r[:, 2]][m2])
        V = np.clip(V, L, U)
        CRmat = np.repeat(CR[:, None], D, 1)
        if sens_cross:
            CRmat = np.clip(CRmat + cfg["kappa"]*decay*(wn - 0.5)[None, :], 0.05, 1.0)
        Um = _crossover(pop, V, CRmat, rng)
        tm["variation"] += _pc() - t0; t0 = _pc()
        fu = bench(Um); fe += NP
        tm["evaluation"] += _pc() - t0; t0 = _pc()
        imp = fu < fit
        if imp.any():
            archive.extend(pop[idx].copy() for idx in np.where(imp)[0])
            sF, sCR = F[imp], CR[imp]
            muF = 0.9*muF + 0.1*(np.sum(sF**2)/(np.sum(sF)+1e-12))   # JADE Lehmer rule, c = 0.1
            muCR = 0.9*muCR + 0.1*np.mean(sCR)
            rew = (fit[imp]-fu[imp])/(np.abs(fit[imp])+1e-12)
            if with_reward: rew = rew + cfg.get("eta", 0.02)*decay*float(np.sum(w*wn))
            for si in np.unique(strat[imp]):
                CS[si] += max(float(np.sum(rew[strat[imp] == si])), 0.0)
            pop[imp] = Um[imp]; fit[imp] = fu[imp]
        while len(archive) > NP: archive.pop(rng.integers(len(archive)))
        if fit.min() < best_prev - 1e-12: stagn = 0; best_prev = fit.min()
        else: stagn += 1
        tm["selection"] += _pc() - t0; t0 = _pc()
        if boundary and stagn >= cfg["G_trigger"]:
            xb = pop[np.argmin(fit)].copy()
            sj = w*D if sens_bound else np.ones(D)
            beta = np.clip(cfg["beta0"]*(1 + cfg["gamma"]*sj), 0, 1.0)
            width = beta*(U0 - L0)
            L = np.maximum(L0, xb - width); U = np.minimum(U0, xb + width)
            phi, inter = _property1_check(L, U, L0, U0, xb, width)
            p1["revisions"] += 1; p1["dims_interior"] += int(inter.sum()); p1["dims_total"] += D
            idx = np.where(inter)[0]
            if len(idx) >= 2:
                a, b = np.meshgrid(idx, idx, indexing="ij"); msk = sj[a] > sj[b] + 1e-15
                p1["pairs_checked"] += int(msk.sum())
                p1["pairs_ok"] += int(np.sum(phi[a][msk] >= phi[b][msk] - 1e-12))
            stagn = 0
        tm["revision"] += _pc() - t0
        hist.append(fit.min())
    best_x = pop[np.argmin(fit)].copy()
    out = dict(best=float(fit.min()), history=np.array(hist), best_x=best_x, fe=fe, property1=p1,
               generations=gen, timing=tm, gate_open_frac=(n_open/n_ref if n_ref else None),
               flat_weight_generations=n_flat)
    if diagnostics:
        S1, ST, STse = sobol_posthoc(bench, lo, hi, D, cfg["posthoc_N"], seed, return_se=True)
        rho = local_perturb(bench, best_x, lo, hi, D, cfg.get("eta_h", 0.01))
        out.update(S1=S1, ST=ST, STse=STse, rho=rho, cv=float(ST.std()/(abs(ST.mean())+1e-12)))
    return out

def adaguide(bench, lo, hi, D, cfg, seed, diagnostics=False, **kw):
    return guided_de(bench, lo, hi, D, cfg, seed, False, False, True, diagnostics=diagnostics, **kw)

def xc_adaguide(bench, lo, hi, D, cfg, seed, diagnostics=False, **kw):
    return guided_de(bench, lo, hi, D, cfg, seed, True, True, True, diagnostics=diagnostics, **kw)

PANEL = [("ClassicDE", de_classic), ("JADE", jade), ("L-SHADE", lshade),
         ("AdaGuiDE", adaguide), ("XC-AdaGuiDE", xc_adaguide)]

# --------------------------------------------------------------------------
# 4. Statistics
# --------------------------------------------------------------------------
def _log10f(x): return np.log10(np.maximum(np.asarray(x, float), 0) + 1e-300)

def paired_dz(base, xc):
    """Paired effect size on log10 fitness. Positive = XC lower (better)."""
    d = _log10f(base) - _log10f(xc); sd = d.std(ddof=1)
    return float(d.mean()/sd) if sd > 0 else 0.0

def cliffs_delta(base, xc):
    """P(base > xc) - P(base < xc). Positive = XC better."""
    b = np.asarray(base, float)[:, None]; x = np.asarray(xc, float)[None, :]
    return float(np.mean(b > x) - np.mean(b < x))

def wilcoxon_report(base, xc):
    """Two-sided Wilcoxon signed-rank; zeros dropped; exact when n <= 50 and no ties."""
    base = np.asarray(base, float); xc = np.asarray(xc, float)
    d = base - xc; dnz = d[d != 0]; nz = len(dnz)
    if nz == 0: return dict(W=np.nan, p=1.0, method="all-zero", n=0)
    ties = len(np.unique(np.abs(dnz))) < nz
    method = "exact" if (nz <= 50 and not ties) else "approx"
    res = stats.wilcoxon(dnz, alternative="two-sided", method=method)
    return dict(W=float(res.statistic), p=float(res.pvalue), method=method, n=int(nz))

def holm(pvals):
    p = np.asarray(pvals, float); m = np.sum(~np.isnan(p)); adj = np.full(len(p), np.nan)
    run = 0.0
    for rank, idx in enumerate([i for i in np.argsort(p) if not np.isnan(p[i])]):
        run = max(run, (m-rank)*p[idx]); adj[idx] = min(run, 1.0)
    return adj

_Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}
def friedman_nemenyi(matrix):
    M = np.asarray(matrix, float)
    all_tied = bool(np.all(np.ptp(M, axis=0) == 0))   # every block identical across settings
    if all_tied:                                     # Friedman undefined (tie correction = 0): no effect
        stat, p = 0.0, 1.0
    elif M.shape[0] < 3:                             # Friedman needs k >= 3; paired Wilcoxon for k = 2
        wr = wilcoxon_report(M[0], M[1]); stat, p = wr["W"], wr["p"]
    else:
        stat, p = stats.friedmanchisquare(*M)
    ranks = np.apply_along_axis(stats.rankdata, 0, M); k, n = M.shape
    return float(stat), float(p), ranks.mean(1), float(_Q_NEMENYI[k]*np.sqrt(k*(k+1)/(6.0*n)))

def _bootstrap_mean_ci(vals, iters, seed):
    vals = np.asarray(vals, float); vals = vals[~np.isnan(vals)]
    if len(vals) < 2: return (float(np.mean(vals)) if len(vals) else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = vals[rng.integers(len(vals), size=(iters, len(vals)))].mean(1)
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def _zrows(M):
    M = np.asarray(M, float); Z = M - M.mean(1, keepdims=True)
    sd = Z.std(1, keepdims=True); sd[sd == 0] = np.nan
    return Z/sd

def perm_mean_r(STs, rhos, iters, seed, chunk=1000):
    """Permutation null for the mean per-seed Pearson r.
    The same label shuffle is applied to every seed of one function.
    Returns observed mean r, one-sided p, and the null draws."""
    ZS = _zrows(STs); ZR = _zrows(rhos); S, D = ZS.shape
    obs = float(np.nanmean(np.nanmean(ZS*ZR, 1)))
    rng = np.random.default_rng(seed); null = np.empty(iters)
    for a in range(0, iters, chunk):
        b = min(iters, a+chunk)
        perms = np.argsort(rng.random((b-a, D)), axis=1)          # (c, D)
        R = ZR[:, perms]                                            # (S, c, D)
        null[a:b] = np.nanmean(np.nanmean(ZS[:, None, :]*R, 2), 0)
    p = float((np.sum(null >= obs - 1e-12) + 1)/(iters + 1))
    return obs, p, null

# --------------------------------------------------------------------------
# 5. Maglev case study (same guided_de core as the benchmark)
#    Plant: m y'' = m g - C (i/y)^2, open-loop unstable. Tune [Kp, Ki, Kd, N].
# --------------------------------------------------------------------------
MAGLEV = dict(m=0.05, g=9.81, y0=0.01, i0=1.0, i_max=3.0, T=1.0, dt=0.004, y_init=0.013, crash=1e-4,
              w_os=4.0, w_u=0.5, fail_cost=1e3)
MAGLEV["C"] = MAGLEV["m"]*MAGLEV["g"]*MAGLEV["y0"]**2/MAGLEV["i0"]**2
MAGLEV_BOUNDS = (np.array([0., 0., 0., 5.]), np.array([2000., 8000., 80., 300.]))
MAGLEV_NAMES = ["Kp", "Ki", "Kd", "N"]

def maglev_cost(gains, P=MAGLEV, return_trace=False, return_parts=False):
    """J = ITAE[mm s^2] + w_os * overshoot[mm] + w_u * effort; J = fail_cost if the gap crashes.
    Integral state: RK4 with stage errors, then anti-windup clip to +-0.05 (m3)."""
    G = np.atleast_2d(np.asarray(gains, float)); n = G.shape[0]
    Kp, Ki, Kd, N = G[:, 0], G[:, 1], G[:, 2], G[:, 3]
    g, C, m, y0, i0, imax = P["g"], P["C"], P["m"], P["y0"], P["i0"], P["i_max"]
    dt, steps = P["dt"], int(P["T"]/P["dt"])
    y = np.full(n, P["y_init"]); v = np.zeros(n); I = np.zeros(n); df = np.zeros(n)
    itae = np.zeros(n); effort = np.zeros(n); ymin = np.full(n, P["y_init"]); alive = np.ones(n, bool)
    ytr = np.zeros((steps, n)); itr = np.zeros((steps, n))
    def deriv(y, v, I, df):
        e = y - y0; i = np.clip(i0 + Kp*e + Ki*I + Kd*df, 0., imax); yy = np.maximum(y, P["crash"])
        return v, g - (C/m)*(i/yy)**2, e, N*(v - df), i
    for k in range(steps):
        t = k*dt
        v1, a1, e1, d1, ii = deriv(y, v, I, df)
        v2, a2, e2, d2, _ = deriv(y+0.5*dt*v1, v+0.5*dt*a1, I+0.5*dt*e1, df+0.5*dt*d1)
        v3, a3, e3, d3, _ = deriv(y+0.5*dt*v2, v+0.5*dt*a2, I+0.5*dt*e2, df+0.5*dt*d2)
        v4, a4, e4, d4, _ = deriv(y+dt*v3, v+dt*a3, I+dt*e3, df+dt*d3)
        with np.errstate(all="ignore"):
            y = y + dt/6*(v1+2*v2+2*v3+v4); v = v + dt/6*(a1+2*a2+2*a3+a4)
            I = np.clip(I + dt/6*(e1+2*e2+2*e3+e4), -0.05, 0.05); df = df + dt/6*(d1+2*d2+2*d3+d4)
        bad = (~np.isfinite(y)) | (y <= P["crash"]) | (y > 5*y0); alive &= ~bad
        e_mm = np.where(alive, (y - y0)*1000.0, 0.0)
        itae += t*np.abs(e_mm)*dt; effort += np.where(alive, (ii - i0)**2*dt, 0.0)
        ymin = np.minimum(ymin, np.where(alive, y, ymin)); ytr[k] = np.where(alive, y, np.nan); itr[k] = ii
    overshoot = np.maximum(0.0, y0 - ymin)*1000.0
    cost = np.where(alive, itae + P["w_os"]*overshoot + P["w_u"]*effort, P["fail_cost"])
    if return_parts: return cost, dict(itae=itae, overshoot=overshoot, effort=effort, alive=alive)
    return (cost, ytr, itr, np.arange(steps)*dt) if return_trace else cost

class MaglevPlant:
    D = 4; name = "Maglev-PID"; lo, hi = MAGLEV_BOUNDS
    def __call__(self, X): return maglev_cost(X)

def run_plant_case_study(cfg, outdir):
    plant = MaglevPlant(); D = plant.D; lo, hi = plant.lo, plant.hi
    pc = dict(cfg, NP=cfg["plant_NP"], posthoc_N=cfg["plant_posthoc_N"])
    ns = cfg["plant_seeds"]; B = dict(best=[], x=[], hist=[]); X = dict(best=[], x=[], hist=[])
    S1a, STa, rhoa, rseed, fail_frac = [], [], [], [], []
    for s in range(ns):
        seed = cfg["seed_base"] + 9000 + 13*s
        rb = adaguide(plant, lo, hi, D, pc, seed, maxfe=cfg["plant_fe"])
        rx = xc_adaguide(plant, lo, hi, D, pc, seed, diagnostics=True, maxfe=cfg["plant_fe"])
        for R, r in ((B, rb), (X, rx)): R["best"].append(r["best"]); R["x"].append(r["best_x"]); R["hist"].append(r["history"])
        S1a.append(rx["S1"]); STa.append(rx["ST"]); rhoa.append(rx["rho"])
        rseed.append(float(np.corrcoef(rx["ST"], rx["rho"])[0, 1]) if rx["rho"].std() > 0 else np.nan)
    bb = np.array(B["best"]); xb = np.array(X["best"])
    wr = wilcoxon_report(bb, xb)
    S1 = np.mean(S1a, 0); ST = np.mean(STa, 0); rho = np.mean(rhoa, 0)
    r_mean, r_lo, r_hi = _bootstrap_mean_ci(rseed, cfg["boot_iters"], stable_seed("plant-boot"))
    # share of the Sobol design that crashes: explains what S_T ranks on this plant
    Pq = qmc.Sobol(d=D, scramble=True, seed=1).random_base2(10); Xs = lo + (hi-lo)*Pq
    _, parts = maglev_cost(Xs, return_parts=True)
    _, parts_b = maglev_cost(np.array(B["x"]), return_parts=True)
    _, parts_x = maglev_cost(np.array(X["x"]), return_parts=True)
    res = dict(D=D, names=MAGLEV_NAMES, role="interpretability_validation", seeds=ns, fe=cfg["plant_fe"],
               objective="J = ITAE + 4*overshoot + 0.5*effort (fail -> 1e3)",
               base_median=float(np.median(bb)), xc_median=float(np.median(xb)),
               improvement_pct=float(100*(np.median(bb)-np.median(xb))/(abs(np.median(bb))+1e-12)),
               wilcoxon_W=wr["W"], wilcoxon_p=wr["p"], wilcoxon_method=wr["method"],
               paired_dz=paired_dz(bb, xb), cliffs_delta=cliffs_delta(bb, xb),
               S1=S1.tolist(), ST=ST.tolist(), rho=rho.tolist(),
               corr_ST_rho_pooled=float(np.corrcoef(ST, rho)[0, 1]),
               corr_ST_rho_perseed_mean=r_mean, corr_ST_rho_perseed_ci=[r_lo, r_hi],
               corr_ST_rho_perseed=rseed,
               sobol_design_fail_fraction=float(1 - parts["alive"].mean()),
               median_itae_only=dict(base=float(np.median(parts_b["itae"])), xc=float(np.median(parts_x["itae"]))),
               base_best=bb.tolist(), xc_best=xb.tolist())
    _plant_figures(outdir, plant, B["hist"], X["hist"], S1, ST, rho, bb, xb, res, B["x"][0], X["x"][0])
    with open(os.path.join(outdir, "maglev_case_study.json"), "w") as f: json.dump(res, f, indent=2, default=float)
    np.savetxt(os.path.join(outdir, "data", "maglev_best_J.csv"), np.column_stack([bb, xb]),
               delimiter=",", header="base,xc", comments="")
    return res

# --------------------------------------------------------------------------
# 6. Experiment drivers
# --------------------------------------------------------------------------
def selftest(cfg, suite):
    """Feasibility of every shifted optimum, axis-aligned and rotated; Sobol on Ishigami."""
    rows = []
    for D in cfg["dims"]:
        for k, (name, fn, lo, hi) in enumerate(suite):
            for rot in (None, cfg["seed_base"] + 97*k + D):
                ok, fx, inside = Benchmark(name, fn, lo, hi, D, rotate_seed=rot).check()
                rows.append(dict(function=name + ("-rot" if rot else ""), dim=D, f_at_xstar=fx, inside=inside, ok=ok))
    bad = [r for r in rows if not r["ok"]]
    def ishigami(X): return np.sin(X[:, 0]) + 7*np.sin(X[:, 1])**2 + 0.1*X[:, 2]**4*np.sin(X[:, 0])
    S1, ST = sobol_posthoc(ishigami, -np.pi, np.pi, 3, 8192, 1)
    ref_S1, ref_ST = np.array([0.3139, 0.4424, 0.0]), np.array([0.5576, 0.4424, 0.2437])
    ish = dict(S1=S1.round(4).tolist(), ST=ST.round(4).tolist(), ref_S1=ref_S1.tolist(), ref_ST=ref_ST.tolist(),
               max_abs_err=float(max(np.abs(S1-ref_S1).max(), np.abs(ST-ref_ST).max())))
    return dict(optimum_checks=len(rows), failures=bad, ishigami=ish,
                passed=(not bad) and ish["max_abs_err"] < 0.02)

def run_panel(cfg, suite, out):
    names = [n for n, _ in PANEL]; cells = []; medians = {n: [] for n in names}
    diag = []; rows = []; histories = {}; p1 = []
    for D in cfg["dims"]:
        for (name, fn, lo, hi) in suite:
            bench = Benchmark(name, fn, lo, hi, D)
            best = {n: [] for n in names}; hh = {n: [] for n in names}
            for s in range(cfg["seeds"]):
                seed = cfg["seed_base"] + 1000*D + 7*s
                for mn, mf in PANEL:
                    r = mf(bench, lo, hi, D, cfg, seed, diagnostics=True) if mn == "XC-AdaGuiDE" \
                        else mf(bench, lo, hi, D, cfg, seed)
                    best[mn].append(r["best"]); hh[mn].append(r["history"])
                    if mn == "XC-AdaGuiDE":
                        diag.append(dict(fn=f"{name}({D})", D=D, ST=r["ST"], STse=r["STse"], rho=r["rho"], cv=r["cv"]))
                        p1.append(r["property1"])
            cells.append((name, D))
            for mn in names: medians[mn].append(float(np.median(best[mn])))
            ml = min(len(h) for h in hh["AdaGuiDE"] + hh["XC-AdaGuiDE"])
            histories[(name, D)] = (np.median(np.vstack([h[:ml] for h in hh["AdaGuiDE"]]), 0),
                                    np.median(np.vstack([h[:ml] for h in hh["XC-AdaGuiDE"]]), 0))
            b = np.array(best["AdaGuiDE"]); x = np.array(best["XC-AdaGuiDE"]); wr = wilcoxon_report(b, x)
            mb, mx = np.median(b), np.median(x)
            rows.append(dict(function=name, dim=D, adaguide_median=mb, xc_median=mx,
                             improvement_pct=100*(mb-mx)/(abs(mb)+1e-300), W=wr["W"], wilcoxon_p=wr["p"],
                             wil_method=wr["method"], paired_dz=paired_dz(b, x), cliffs_delta=cliffs_delta(b, x)))
            print(f"   {name:<11} {D:>2}D  base {mb:.3g}  xc {mx:.3g}  p={wr['p']:.3g}")
    padj = holm([r["wilcoxon_p"] for r in rows]); wl = dict(win=0, loss=0, tie=0)
    for r, pa in zip(rows, padj):
        r["holm_p"] = float(pa)
        r["verdict"] = ("win" if r["xc_median"] < r["adaguide_median"] else "loss") if pa < 0.05 else "tie"
        wl[r["verdict"]] += 1
    M = np.array([medians[n] for n in names]); fs, fp, mr, CD = friedman_nemenyi(M)
    tot = {k: sum(d[k] for d in p1) for k in p1[0]} if p1 else {}
    panel = dict(methods=names, n_cells=len(cells), friedman_stat=fs, friedman_p=fp, nemenyi_CD=CD,
                 mean_ranks=dict(zip(names, map(float, mr))), holm_vs_adaguide=wl,
                 budget_mode=cfg["budget_mode"],
                 property1=dict(tot, pair_rate=(tot["pairs_ok"]/tot["pairs_checked"]) if tot.get("pairs_checked") else None,
                                interior_dim_share=(tot["dims_interior"]/tot["dims_total"]) if tot.get("dims_total") else None))
    _wcsv(os.path.join(out, "E1_xc_vs_adaguide.csv"), rows,
          ["function", "dim", "adaguide_median", "xc_median", "improvement_pct", "W", "wilcoxon_p", "holm_p",
           "wil_method", "paired_dz", "cliffs_delta", "verdict"])
    _wcsv(os.path.join(out, "E1_panel_medians.csv"),
          [dict(function=c[0], dim=c[1], **{n: medians[n][i] for n in names}) for i, c in enumerate(cells)],
          ["function", "dim"] + names)
    return dict(panel=panel, rows=rows, medians=medians, cells=cells, diag=diag, histories=histories)

def run_ablation_perseed(cfg, suite, out):
    D = max(cfg["dims"]); ns = cfg["abl_seeds"]
    variants = {"baseline": dict(sens_cross=False, sens_bound=False, boundary=True),
                "no_crossover": dict(sens_cross=False, sens_bound=True, boundary=True),
                "no_boundary": dict(sens_cross=True, sens_bound=False, boundary=False),
                "with_reward": dict(sens_cross=True, sens_bound=True, boundary=True, with_reward=True),
                "full_xc": dict(sens_cross=True, sens_bound=True, boundary=True)}
    vn = list(variants); blocks = {v: [] for v in vn}
    for (name, fn, lo, hi) in suite:
        bench = Benchmark(name, fn, lo, hi, D)
        for s in range(ns):
            seed = cfg["seed_base"] + 555 + 7*s
            for v, sw in variants.items():
                blocks[v].append(guided_de(bench, lo, hi, D, cfg, seed, **sw)["best"])
    Mat = np.array([blocks[v] for v in vn]); fs, fp, mr, CD = friedman_nemenyi(Mat)
    wr = wilcoxon_report(np.array(blocks["no_boundary"]), np.array(blocks["full_xc"]))
    rows = [dict(variant=v, mean_rank=float(m), median=float(np.median(blocks[v])), mean=float(np.mean(blocks[v])))
            for v, m in zip(vn, mr)]
    _wcsv(os.path.join(out, "E2_ablation_perseed.csv"), rows, ["variant", "mean_rank", "median", "mean"])
    return dict(dim=D, seeds=ns, n_blocks=Mat.shape[1], variants=vn, friedman_stat=fs, friedman_p=fp,
                nemenyi_CD=CD, mean_ranks=dict(zip(vn, map(float, mr))),
                full_vs_no_boundary=dict(W=wr["W"], p=wr["p"], method=wr["method"], n=wr["n"]))

def run_rotated(cfg, suite, out):
    rows = []
    for D in cfg["dims"]:
        for k, (name, fn, lo, hi) in enumerate(suite):
            rb = Benchmark(name, fn, lo, hi, D, rotate_seed=cfg["seed_base"] + 97*k + D)
            b, x = [], []
            for s in range(cfg["rot_seeds"]):
                seed = cfg["seed_base"] + 314 + 1000*D + 7*s
                b.append(adaguide(rb, lo, hi, D, cfg, seed)["best"]); x.append(xc_adaguide(rb, lo, hi, D, cfg, seed)["best"])
            b, x = np.array(b), np.array(x); wr = wilcoxon_report(b, x); mb, mx = np.median(b), np.median(x)
            raw = ("win" if mx < mb else "loss") if wr["p"] < 0.05 else "tie"
            rows.append(dict(function=rb.name, dim=D, adaguide_median=mb, xc_median=mx,
                             improvement_pct=100*(mb-mx)/(abs(mb)+1e-300), W=wr["W"], wilcoxon_p=wr["p"],
                             paired_dz=paired_dz(b, x), verdict_raw=raw))
    padj = holm([r["wilcoxon_p"] for r in rows])
    for r, pa in zip(rows, padj):
        r["holm_p"] = float(pa)
        r["verdict_holm"] = ("win" if r["xc_median"] < r["adaguide_median"] else "loss") if pa < 0.05 else "tie"
    _wcsv(os.path.join(out, "E3_rotated.csv"), rows, ["function", "dim", "adaguide_median", "xc_median",
          "improvement_pct", "W", "wilcoxon_p", "holm_p", "paired_dz", "verdict_raw", "verdict_holm"])
    cnt = lambda key: {v: sum(r[key] == v for r in rows) for v in ("win", "loss", "tie")}
    return dict(n_cells=len(rows), raw=cnt("verdict_raw"), holm=cnt("verdict_holm"))

def run_consistency(cfg, diag, out):
    """Per-function mean per-seed r; bootstrap CI and permutation p on the same statistic;
    r_min calibrated per dimension; Eq. (29): supported iff ci_low > r_min(D) and p < 0.05."""
    by_fn = {}
    for d in diag: by_fn.setdefault(d["fn"], []).append(d)
    rows, nulls = [], {}
    for fn, lst in by_fn.items():
        D = lst[0]["D"]; STs = np.array([d["ST"] for d in lst]); rhos = np.array([d["rho"] for d in lst])
        cv = float(np.median([d["cv"] for d in lst]))
        # heterogeneity gate: CV above cv_gate AND spread of ST larger than its own Monte-Carlo noise
        STm = STs.mean(0); se = np.sqrt(np.mean(np.array([d["STse"] for d in lst])**2, 0)/len(lst))
        het = bool(cv > cfg["cv_gate"] and (STm.max() - STm.min()) > 4*np.max(se))
        row = dict(function=fn, dim=D, cv_ST=cv, gated=het)
        if het and np.all(rhos.std(1) > 0):
            obs, p, null = perm_mean_r(STs, rhos, cfg["perm_iters"], stable_seed("perm", fn))
            nulls.setdefault(D, []).append(null)
            per = [np.corrcoef(a, b)[0, 1] for a, b in zip(STs, rhos)]
            m, lo_, hi_ = _bootstrap_mean_ci(per, cfg["boot_iters"], stable_seed("boot", fn))
            sp = np.nanmean([stats.spearmanr(a, b)[0] for a, b in zip(STs, rhos)])
            kt = np.nanmean([stats.kendalltau(a, b)[0] for a, b in zip(STs, rhos)])
            row.update(r_mean=obs, ci_low=lo_, ci_high=hi_, perm_p=p, spearman=float(sp), kendall=float(kt),
                       p_floor=float(1.0/(cfg["perm_iters"]+1)))
        rows.append(row)
    r_min = {D: float(np.percentile(np.concatenate(v), 95)) for D, v in nulls.items()}
    for r in rows:
        if r["gated"] and "r_mean" in r:
            rm = r_min.get(r["dim"], cfg["r_min_fallback"]); r["r_min_D"] = rm
            r["supported"] = bool(r["ci_low"] > rm and r["perm_p"] < 0.05)
        else: r["supported"] = False
    rows.sort(key=lambda r: -r.get("r_mean", -9))
    _wcsv(os.path.join(out, "E4_consistency.csv"), rows,
          ["function", "dim", "cv_ST", "gated", "r_mean", "ci_low", "ci_high", "perm_p", "spearman", "kendall",
           "r_min_D", "supported"])
    return dict(r_min_by_dim=r_min, cv_gate=cfg["cv_gate"], n_functions=len(rows),
                n_gated=sum(r["gated"] for r in rows), n_supported=sum(r["supported"] for r in rows),
                per_function=rows)

def run_sobol_sweep(cfg, out):
    """E5: Sobol convergence with the centred estimator."""
    D = max(cfg["dims"]); name = cfg["sweep_fn"]
    fn, lo, hi = {n: (f, a, b) for n, f, a, b in SUITE}[name]
    bench = Benchmark(name, fn, lo, hi, D); rows = []
    for N in cfg["sweep_N"]:
        s1 = np.array([sobol_posthoc(bench, lo, hi, D, N, cfg["seed_base"] + rep)[0].mean()
                       for rep in range(cfg["sweep_reps"])])
        rows.append(dict(N=int(N), S1_mean=float(s1.mean()), S1_se=float(s1.std(ddof=1)/np.sqrt(len(s1)))))
    _wcsv(os.path.join(out, "E5_sobol_sweep.csv"), rows, ["N", "S1_mean", "S1_se"])
    return dict(function=name, dim=D, reps=cfg["sweep_reps"], reference_S1=1.0/D, sweep=rows)

def run_param_sweep(cfg, suite, out):
    """E6: one-at-a-time sweep of XC parameters at 30D. Median final fitness
    and mean rank of each setting across (function x seed) blocks."""
    D = max(cfg["dims"]); rows = []; raw_all = []
    pool = suite + [t for t in cfg.get("extra_suite", []) if t not in suite]
    fns = [s for s in pool if s[0] in cfg["sweep_funcs"]]
    for par, grid in cfg["sweep_grid"].items():
        res = {v: [] for v in grid}; rev = {v: 0 for v in grid}; raw = []
        for (name, fn, lo, hi) in fns:
            bench = Benchmark(name, fn, lo, hi, D)
            for s in range(cfg["sweep_seeds"]):
                seed = cfg["seed_base"] + 777 + 7*s
                for v in grid:
                    r = xc_adaguide(bench, lo, hi, D, dict(cfg, **{par: v}), seed)
                    res[v].append(r["best"]); rev[v] += r["property1"]["revisions"]
                    raw.append(dict(param=par, value=v, function=name, seed=s, best=r["best"],
                                    revisions=r["property1"]["revisions"]))
        M = np.array([res[v] for v in grid]); fs, fp, mr, CD = friedman_nemenyi(M)
        tied = bool(np.all(np.ptp(M, axis=0) == 0)); nrun = M.shape[1]
        for v, m in zip(grid, mr):
            rows.append(dict(param=par, value=v, mean_rank=float(m), friedman_p=fp, CD=CD,
                             median_log10=float(np.median(_log10f(res[v]))),
                             revisions_per_run=rev[v]/nrun, all_blocks_tied=tied))
        nt = int(np.sum(np.ptp(M, axis=0) == 0)); raw_all.extend(raw)
        for rr in rows[-len(grid):]: rr["tied_blocks"] = f"{nt}/{nrun}"
        note = "  [all blocks tied: parameter inactive at this budget]" if tied else ""
        print(f"   sweep {par}: Friedman p = {fp:.3g}, bound revisions/run = {np.mean(list(rev.values()))/nrun:.2f}{note}")
    _wcsv(os.path.join(out, "E6_param_sweep.csv"), rows, ["param", "value", "mean_rank", "friedman_p", "CD",
          "median_log10", "revisions_per_run", "all_blocks_tied", "tied_blocks"])
    _wcsv(os.path.join(out, "E6_param_sweep_raw.csv"), raw_all, ["param", "value", "function", "seed", "best", "revisions"])
    return rows

def _wcsv(path, rows, fields):
    with open(path, "w") as f:
        f.write(",".join(fields) + "\n")
        for r in rows: f.write(",".join(str(r.get(k, "")) for k in fields) + "\n")

# --------------------------------------------------------------------------
# 7. Figures
# --------------------------------------------------------------------------
def _plant_figures(out, plant, base_hist, xc_hist, S1, ST, rho, bb, xb, res, g_base, g_xc):
    fg = os.path.join(out, "figures"); D = plant.D
    _, yb, _, t = maglev_cost(g_base[None, :], return_trace=True); _, yx, _, _ = maglev_cost(g_xc[None, :], return_trace=True)
    plt.figure(figsize=(8, 4.2)); plt.axhline(MAGLEV["y0"]*1000, color="k", ls=":", lw=1, label="set-point")
    plt.plot(t, yb[:, 0]*1000, label=f"AdaGuiDE (J {plant(g_base[None, :])[0]:.3g})", lw=1.6)
    plt.plot(t, yx[:, 0]*1000, label=f"XC-AdaGuiDE (J {plant(g_xc[None, :])[0]:.3g})", lw=1.6, ls="--")
    plt.xlabel("time (s)"); plt.ylabel("air gap (mm)"); plt.title("Maglev regulation response (seed 0)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(fg, "figP1_response.png"), dpi=150); plt.close()
    idx = np.arange(D)
    plt.figure(figsize=(6.5, 4)); plt.bar(idx-0.2, S1, 0.4, label="S1", color="#A6A6A6", edgecolor="k")
    plt.bar(idx+0.2, ST, 0.4, label="ST", color="#404040", edgecolor="k")
    plt.xticks(idx, MAGLEV_NAMES); plt.ylabel("Sobol index"); plt.title("Controller-gain sensitivity of J")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(fg, "figP2_sensitivity.png"), dpi=150); plt.close()
    plt.figure(figsize=(5, 5)); plt.scatter(ST, rho, s=80, color="k")
    for j, nm in enumerate(MAGLEV_NAMES): plt.annotate(nm, (ST[j], rho[j]), textcoords="offset points", xytext=(6, 4))
    plt.xlabel("Sobol total-order ST"); plt.ylabel("perturbation importance rho")
    ci = res["corr_ST_rho_perseed_ci"]
    plt.title(f"per-seed mean r = {res['corr_ST_rho_perseed_mean']:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]")
    plt.tight_layout(); plt.savefig(os.path.join(fg, "figP3_consistency.png"), dpi=150); plt.close()
    fig, ax = plt.subplots(figsize=(5.5, 4)); ax.boxplot([bb, xb]); ax.set_xticks([1, 2])
    ax.set_xticklabels(["AdaGuiDE", "XC-AdaGuiDE"]); ax.set_yscale("log"); ax.set_ylabel("final J (per seed)")
    ax.set_title(f"n = {len(bb)} seeds, Wilcoxon p = {res['wilcoxon_p']:.3g}")
    fig.tight_layout(); fig.savefig(os.path.join(fg, "figP4_boxplot.png"), dpi=150); plt.close(fig)

def make_figures(out, P, E2, E3, E4, E5):
    fg = os.path.join(out, "figures")
    mr = P["panel"]["mean_ranks"]; nm = list(mr); v = [mr[n] for n in nm]; o = np.argsort(v)
    plt.figure(figsize=(7, 4)); plt.barh([nm[i] for i in o], [v[i] for i in o], color="#595959")
    cd = P["panel"]["nemenyi_CD"]; plt.axvline(min(v)+cd, color="k", ls="--", lw=1, label=f"CD = {cd:.2f} from best")
    plt.xlabel("mean Friedman rank (lower better)"); plt.title(f"Five-method panel ({P['panel']['n_cells']} cells)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(fg, "fig1_panel_ranks.png"), dpi=150); plt.close()
    r30 = [r for r in P["rows"] if r["dim"] == max(r["dim"] for r in P["rows"])]
    col = {"win": "#262626", "loss": "#8C8C8C", "tie": "#D9D9D9"}
    plt.figure(figsize=(9, 4)); plt.bar([r["function"] for r in r30], [r["improvement_pct"] for r in r30],
                                        color=[col[r["verdict"]] for r in r30], edgecolor="k")
    plt.ylabel("median improvement over base (%)"); plt.title("XC-AdaGuiDE vs AdaGuiDE base (Holm verdict)")
    plt.xticks(rotation=30, ha="right"); plt.axhline(0, color="k", lw=0.8); plt.tight_layout()
    plt.savefig(os.path.join(fg, "fig2_xc_vs_adaguide.png"), dpi=150); plt.close()
    Dm = max(c[1] for c in P["cells"]); plt.figure(figsize=(10, 7))
    for k, fn in enumerate(["Sphere", "Rastrigin", "Rosenbrock", "Ackley"], 1):
        if (fn, Dm) not in P["histories"]: continue
        b, x = P["histories"][(fn, Dm)]; ax = plt.subplot(2, 2, k)
        ax.plot(np.maximum(b, 1e-300), label="AdaGuiDE", lw=1.5, color="k")
        ax.plot(np.maximum(x, 1e-300), label="XC-AdaGuiDE", lw=1.5, color="k", ls="--")
        ax.set_yscale("log"); ax.set_title(f"{fn} ({Dm}D)"); ax.set_xlabel("generation"); ax.set_ylabel("best fitness")
        if k == 1: ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(fg, "fig3_convergence.png"), dpi=150); plt.close()
    mr2 = E2["mean_ranks"]; n2 = list(mr2); v2 = [mr2[n] for n in n2]; o2 = np.argsort(v2)
    plt.figure(figsize=(7, 4)); plt.barh([n2[i] for i in o2], [v2[i] for i in o2], color="#7F7F7F")
    plt.axvline(min(v2)+E2["nemenyi_CD"], color="k", ls="--", lw=1, label=f"CD = {E2['nemenyi_CD']:.2f}")
    plt.xlabel(f"mean rank over {E2['n_blocks']} blocks"); plt.title(f"Per-seed ablation ({E2['dim']}D)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(fg, "fig4_ablation_perseed.png"), dpi=150); plt.close()
    if E3:
        r3 = [r for r in _read_csv(os.path.join(out, "E3_rotated.csv")) if int(r["dim"]) == Dm]
        plt.figure(figsize=(9, 4)); plt.bar([r["function"] for r in r3], [float(r["improvement_pct"]) for r in r3],
                                            color="#A6A6A6", edgecolor="k")
        plt.axhline(0, color="k", lw=0.8); plt.ylabel("median improvement over base (%)")
        plt.title(f"Rotated suite ({Dm}D)"); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
        plt.savefig(os.path.join(fg, "fig5_rotated.png"), dpi=150); plt.close()
    pf = [r for r in E4["per_function"] if "r_mean" in r]
    if pf:
        plt.figure(figsize=(8, 4)); x = np.arange(len(pf))
        plt.bar(x, [r["r_mean"] for r in pf], color=["#262626" if r["supported"] else "#BFBFBF" for r in pf], edgecolor="k")
        plt.errorbar(x, [r["r_mean"] for r in pf], yerr=[[r["r_mean"]-r["ci_low"] for r in pf],
                     [r["ci_high"]-r["r_mean"] for r in pf]], fmt="none", ecolor="k", capsize=3)
        for D, rm in E4["r_min_by_dim"].items(): plt.axhline(rm, ls="--", lw=1, color="k")
        plt.xticks(x, [r["function"] for r in pf], rotation=35, ha="right")
        plt.ylabel("mean per-seed r(ST, rho)"); plt.title("Consistency; dashed = r_min per dimension")
        plt.tight_layout(); plt.savefig(os.path.join(fg, "fig6_consistency.png"), dpi=150); plt.close()
    if E5:
        sw = E5["sweep"]; plt.figure(figsize=(6, 4))
        plt.errorbar([r["N"] for r in sw], [r["S1_mean"] for r in sw], yerr=[r["S1_se"] for r in sw],
                     marker="o", capsize=4, color="k")
        plt.axhline(E5["reference_S1"], ls=":", color="k", label="1/D")
        plt.xscale("log", base=2); plt.xlabel("Sobol base samples N"); plt.ylabel(f"mean S1 ({E5['function']} {E5['dim']}D)")
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(fg, "fig7_sobol_sweep.png"), dpi=150); plt.close()

def _read_csv(path):
    import csv
    with open(path) as f: return list(csv.DictReader(f))

# --------------------------------------------------------------------------
# 8. Orchestrator + ZIP
# --------------------------------------------------------------------------
def main(cfg, suite=None):
    suite = suite or SUITE; t0 = time.time(); out = cfg["outdir"]
    for sub in ["", "figures", "data"]: os.makedirs(os.path.join(out, sub), exist_ok=True)
    print(">> Self-test (optimum feasibility, Ishigami Sobol) ..."); ST_ = selftest(cfg, suite)
    print(f"   {ST_['optimum_checks']} optimum checks, {len(ST_['failures'])} failures; "
          f"Ishigami max error {ST_['ishigami']['max_abs_err']:.4f}")
    if not ST_["passed"]: raise RuntimeError(f"Self-test failed: {ST_}")
    print(">> E1 five-method panel ..."); P = run_panel(cfg, suite, out)
    print(">> E2 per-seed ablation ..."); E2 = run_ablation_perseed(cfg, suite, out)
    E3 = None
    if cfg["run_rotated"]: print(">> E3 rotated suite ..."); E3 = run_rotated(cfg, suite, out)
    print(">> E4 consistency ..."); E4 = run_consistency(cfg, P["diag"], out)
    E5 = None
    if cfg["run_sobol_sweep"]: print(">> E5 Sobol convergence ..."); E5 = run_sobol_sweep(cfg, out)
    E6 = None
    if cfg["run_param_sweep"]: print(">> E6 parameter sweep ..."); E6 = run_param_sweep(cfg, suite, out)
    plant = None
    if cfg["run_plant"]: print(">> Maglev case study ..."); plant = run_plant_case_study(cfg, out)
    make_figures(out, P, E2, E3, E4, E5)
    summary = dict(config=cfg, selftest=ST_, E1_panel=P["panel"], E2_ablation_perseed=E2, E3_rotated=E3,
                   E4_consistency=E4, E5_sobol_sweep=E5, E6_param_sweep=E6, maglev_case_study=plant,
                   runtime_sec=round(time.time()-t0, 1),
                   environment=dict(python=platform.python_version(), numpy=np.__version__,
                                    scipy=scipy.__version__, matplotlib=matplotlib.__version__))
    with open(os.path.join(out, "summary.json"), "w") as f: json.dump(summary, f, indent=2, default=float)
    print("\n===== SUMMARY =====")
    print(json.dumps(dict(E1=P["panel"], E2=E2, E3=E3,
                          E4={k: E4[k] for k in ("r_min_by_dim", "n_functions", "n_gated", "n_supported")},
                          maglev={k: plant[k] for k in ("base_median", "xc_median", "wilcoxon_p", "paired_dz",
                                                        "corr_ST_rho_perseed_mean", "corr_ST_rho_perseed_ci",
                                                        "sobol_design_fail_fraction")} if plant else None),
                     indent=2, default=float))
    return summary

def zip_results(out="results", zipname="xc_adaguide_results.zip"):
    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out):
            for fn in files:
                p = os.path.join(root, fn); z.write(p, os.path.relpath(p, os.path.dirname(os.path.abspath(out)) or "."))
    return zipname
