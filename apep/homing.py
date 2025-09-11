# homing.py
# Homing-in for vertical sounding:
# For each frequency, launch a fan of rays, build D(theta) = ground range (signed),
# interpolate D, and solve D(theta)=0 for theta roots (possibly multiple roots).

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional
from loguru import logger
from types import SimpleNamespace
import sys

sys.path.append("apep/")
from iri import IRI, IonosphereModels
from rt2d import RayConfig, RayTracer2D
from rtplots import PlotRays

# Optional: SciPy for a nicer shape-preserving spline and robust brentq
try:
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------- Small helpers ----------
def _sign_or_eps(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Sign function with epsilon deadzone around zero (stabilizes sign-change detection)."""
    s = np.sign(x)
    s[np.abs(x) < eps] = 0.0
    return s

def _piecewise_linear_interpolant(xs: np.ndarray, ys: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Fallback interpolant if SciPy is unavailable."""
    def f(xq):
        return np.interp(np.asarray(xq, float), xs, ys)
    return f

def _bracketed_roots(theta: np.ndarray, D: np.ndarray, tol: float = 1e-4) -> List[tuple]:
    """
    Return index pairs (i,i+1) where D changes sign or touches ~0.
    Handles flat zeros (plateau crossing).
    """
    roots = []
    s = _sign_or_eps(D, eps=tol)
    for i in range(len(theta) - 1):
        if s[i] == 0.0:  # exact/near zero at node
            roots.append((i, i))       # singleton; will return theta[i]
        elif s[i] * s[i+1] < 0.0:      # sign change across interval
            roots.append((i, i+1))
        elif s[i+1] == 0.0:
            roots.append((i+1, i+1))
    # Deduplicate any adjacent singletons
    uniq = []
    for r in roots:
        if not uniq or uniq[-1] != r:
            uniq.append(r)
    return uniq

# ---------- Your tracer “interface” ----------
# We assume a callable to launch ONE ray:
#   launch(f_MHz: float, el_deg: float) -> dict with keys:
#       'reason' in {'ground_hit','cutoff_reached','domain_limit','out_of_bounds',...}
#       'x_km' (np.ndarray), 'y_km' (np.ndarray)
#
# If you’re using the earlier RayTracer2D / RayConfig I provided:
#   make a small closure that constructs RayConfig and calls trace(f, el)

@dataclass
class HomingConfig:
    angles_deg: np.ndarray      # e.g., np.linspace(75, 90, 61) (elev above horizontal)
    ds_km: float = 0.4          # passed to your RayConfig (if used in launcher)
    s_max_km: float = 2200.0    # "
    zero_tol_km: float = 1e-2   # root tolerance in ground range
    min_valid_span_deg: float = 0.05  # discard teeny brackets (deg)
    restrict_to_ground_hit: bool = True  # ignore non-returning rays

def homing_for_frequency(
    f_MHz: float,
    launch_fn: Callable[[float, float], Dict[str, Any]],
    cfg: HomingConfig,
) -> Dict[str, Any]:
    """
    Run homing for one frequency:
      - launch fan, collect ground ranges D(theta)
      - build interpolant
      - find all roots of D(theta)=0 (vertical sounding returns)
    Returns dict with samples and roots.
    """
    thetas = np.asarray(cfg.angles_deg, float)
    D = np.full_like(thetas, np.nan, dtype=float)  # signed ground range

    # --- 1) shoot the ray fan and record landing x
    for i, el in enumerate(thetas):
        out = launch_fn(f_MHz, el)
        hit = (out.get("reason") in ["ground_hit", "evanescent"])
        if cfg.restrict_to_ground_hit and not hit:
            continue
        x = out.get("x_km"); y = out.get("y_km")
        if x is None or y is None or len(x) == 0:
            continue
        # take signed ground range at the FIRST ground hit (tracer returns that)
        D[i] = x[-1]

    # filter to valid samples
    mask = ~np.isnan(D)
    if mask.sum() < 2:
        return dict(f_MHz=f_MHz, theta=thetas, D=D, roots=[], interpolant=None)

    theta_s = thetas[mask]
    D_s = D[mask]

    # Ensure strictly increasing theta (required by interpolators)
    order = np.argsort(theta_s)
    theta_s = theta_s[order]
    D_s = D_s[order]

    # --- 2) interpolant D_hat(theta)
    if _HAVE_SCIPY and len(theta_s) >= 3:
        D_hat = PchipInterpolator(theta_s, D_s, extrapolate=False)
        f_eval = lambda t: np.array(D_hat(t), dtype=float)
    else:
        D_hat = _piecewise_linear_interpolant(theta_s, D_s)
        f_eval = D_hat

    # --- 3) bracketing & root finding
    roots = []
    # evaluate on a refined grid to find brackets robustly
    theta_dense = np.linspace(theta_s[0], theta_s[-1], max(400, 5*len(theta_s)))
    D_dense = f_eval(theta_dense)
    brackets = _bracketed_roots(theta_dense, D_dense, tol=cfg.zero_tol_km)

    for (i0, i1) in brackets:
        if i0 == i1:
            # near-zero at a grid node
            th = float(theta_dense[i0])
            roots.append(th)
            continue
        a, b = float(theta_dense[i0]), float(theta_dense[i1])
        if abs(b - a) < cfg.min_valid_span_deg:
            # too narrow to be robust; take midpoint if close to zero
            mid = 0.5*(a+b)
            if abs(float(f_eval(mid))) < 5*cfg.zero_tol_km:
                roots.append(mid)
            continue
        # robust bracket solve
        if _HAVE_SCIPY:
            try:
                th = float(brentq(lambda t: float(f_eval(t)), a, b, xtol=cfg.zero_tol_km, rtol=1e-6))
                roots.append(th)
            except Exception:
                # fallback: midpoint if function appears small there
                mid = 0.5*(a+b)
                if abs(float(f_eval(mid))) < 5*cfg.zero_tol_km:
                    roots.append(mid)
        else:
            # bisection on interpolant (piecewise linear)
            fa, fb = float(f_eval(a)), float(f_eval(b))
            if fa == 0.0:
                roots.append(a); continue
            if fb == 0.0:
                roots.append(b); continue
            # bisection
            left, right = a, b
            for _ in range(60):
                mid = 0.5*(left+right)
                fm = float(f_eval(mid))
                if abs(fm) < cfg.zero_tol_km:
                    break
                if fa*fm <= 0:
                    right, fb = mid, fm
                else:
                    left, fa = mid, fm
            roots.append(mid)

    # sort & dedup within tolerance
    roots = np.array(sorted(roots), float)
    if roots.size:
        keep = [0]
        for i in range(1, roots.size):
            if abs(roots[i] - roots[keep[-1]]) > 0.02:  # 0.02 deg merge tolerance
                keep.append(i)
        roots = roots[keep]

    return dict(
        f_MHz=f_MHz,
        theta_samples=theta_s,
        D_samples=D_s,
        interpolant=D_hat,
        roots_deg=roots.tolist()
    )

# ---------- Convenience: run across many frequencies ----------
def homing_sweep(
    freqs_MHz: np.ndarray,
    launch_fn: Callable[[float, float], Dict[str, Any]],
    cfg: HomingConfig
) -> List[Dict[str, Any]]:
    results = []
    for f in np.asarray(freqs_MHz, float):
        res = homing_for_frequency(f, launch_fn, cfg)
        results.append(res)
    return results


# Build a demo Ne(x,y) field: Chapman background + Wave front
def ray_trace_2d_ionosphereic_wave_front(
    x: np.ndarray = np.linspace(-500, 500, 2001),  # horizontal distance [km]
    hs: np.ndarray = np.linspace(0, 1000, 2001),      # altitude [km]
    layer_names: np.ndarray = np.asarray(["E", "F1", "F2"]),
    layer_heights: np.ndarray = np.asarray([110., 180.0, 300.]),
    layer_base_ne: np.ndarray = np.asarray([1e11, 4.e11, 11.e11]),
    layer_scales: np.ndarray = np.asarray([10., 25., 50.]),
    Ne_floor:float = 2e10,
    x_params: np.ndarray = np.asarray([-62, 93, 127]),
    d_params: np.ndarray = np.asarray([0.4, 0.15]),
    homing_freq: float = 8.3,
    el_angles: np.ndarray = np.arange(50, 110, 2),
    x0_km=0.0,
    y0_km=0.0,
    s_max_km=3000.0,   # allow enough total path
    ds_km=0.05,         # 0.25–1.0 km is a good starting step
    y_max_km=1100.0,
    x_max_km=4000.0,
    keep_every=1,
    figure_file_name=None,
):

    def launch(f_MHz: float, el_deg: float):
        cg = RayConfig(
            f_MHz=f_MHz,
            el0_deg=el_deg,
            x0_km=x0_km,
            y0_km=y0_km,
            s_max_km=s_max_km,   # allow enough total path
            ds_km=ds_km,         # 0.25–1.0 km is a good starting step
            y_max_km=y_max_km,
            x_max_km=x_max_km,
            keep_every=keep_every
        )
        out = rt.trace(cg)
        return out  # must include 'reason','x_km','y_km'

    X, Z, Ne, alpha_X, Nex = IonosphereModels.cusp_function_alpha(
        x, hs, layer_names, layer_heights, layer_base_ne,
        layer_scales, Ne_floor, x_params, d_params
    )
    rt = RayTracer2D(x, hs, Nex)
    cfg = HomingConfig(angles_deg=el_angles, ds_km=ds_km, s_max_km=s_max_km, zero_tol_km=1e-3)
    res_1050 = homing_for_frequency(homing_freq, launch, cfg)
    print(res_1050)
    return


if __name__ == "__main__":
    ray_trace_2d_ionosphereic_wave_front(
        figure_file_name="figures/rt/wv.png"
    )