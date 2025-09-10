# hf_raytrace_2d.py
# ---------------------------------------------------------------------
# 2-D HF ray tracing in an inhomogeneous ionosphere n(x,y) from Ne(x,y)
# No magnetic field, no collisions: n^2 = 1 - (f_p/f)^2
# Uses Hamiltonian ray equations in 2D with a simple RK4 integrator.
# Units:
#   x, y in km; Ne in m^-3; frequency f in Hz; c in km/s internally.
# ---------------------------------------------------------------------
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# ------------------------- Physical constants -------------------------
EPS0 = 8.8541878128e-12      # F/m
QE   = 1.602176634e-19       # C
ME   = 9.10938356e-31        # kg
TWOPI = 2.0 * np.pi
C_M_S = 299792458.0          # m/s
C_KM_S = C_M_S / 1e3         # km/s


# ======================== Helper: Interpolator ========================
class Bilinear2D:
    """
    Bilinear interpolator for a rectilinear (x,y) grid.
    Grid:
      x [Nx] (km, ascending), y [Ny] (km, ascending)
      F [Ny, Nx] values (e.g., Ne in m^-3) at (x[i], y[j]) => F[j, i]
    """
    def __init__(self, x_km: np.ndarray, y_km: np.ndarray, F: np.ndarray):
        x = np.asarray(x_km, float)
        y = np.asarray(y_km, float)
        F = np.asarray(F, float)
        assert F.shape == (y.size, x.size), "F must be [Ny, Nx]"
        assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0), "x,y must be strictly increasing"
        self.x = x
        self.y = y
        self.F = F
        self.dx = np.diff(x).mean()
        self.dy = np.diff(y).mean()

    def inside(self, x: float, y: float) -> bool:
        return (self.x[0] <= x <= self.x[-1]) and (self.y[0] <= y <= self.y[-1])

    def __call__(self, xq: float, yq: float) -> float:
        """ Interpolate value at (xq,yq). Out of bounds returns 0. """
        if not self.inside(xq, yq):
            return 0.0
        # find i such that x[i] <= xq <= x[i+1]
        i = np.searchsorted(self.x, xq) - 1
        j = np.searchsorted(self.y, yq) - 1
        i = np.clip(i, 0, self.x.size - 2)
        j = np.clip(j, 0, self.y.size - 2)
        x0, x1 = self.x[i], self.x[i+1]
        y0, y1 = self.y[j], self.y[j+1]
        tx = (xq - x0) / (x1 - x0)
        ty = (yq - y0) / (y1 - y0)
        f00 = self.F[j,   i  ]
        f10 = self.F[j,   i+1]
        f01 = self.F[j+1, i  ]
        f11 = self.F[j+1, i+1]
        return ( (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11 )

    def grad_central(self, xq: float, yq: float, h: float | None = None) -> tuple[float, float]:
        """
        Central-difference gradient of F at (xq,yq) using bilinear values.
        h: step (km); default uses grid-spacing average.
        Returns (dF/dx, dF/dy) with F units per km.
        """
        if h is None:
            h = 0.5*(self.dx + self.dy)
        fxp = self(xq + h, yq); fxm = self(xq - h, yq)
        fyp = self(xq, yq + h); fym = self(xq, yq - h)
        dfdx = (fxp - fxm) / (2*h)
        dfdy = (fyp - fym) / (2*h)
        return dfdx, dfdy


# ======================== Plasma / refractive index ====================
def plasma_freq_hz(ne_m3: float | np.ndarray) -> np.ndarray:
    """ f_p [Hz] from electron density ne [m^-3]. """
    omega_p = np.sqrt(ne_m3 * QE*QE / (EPS0 * ME))
    return omega_p / TWOPI

def n2_from_ne(f_hz: float, ne_m3: float | np.ndarray, n2_floor: float = 1e-8) -> np.ndarray:
    """ n^2 = 1 - (f_p/f)^2 with a small positive floor to avoid NaNs. """
    fp = plasma_freq_hz(ne_m3)
    n2 = 1.0 - (fp / f_hz)**2
    return np.maximum(n2, n2_floor)


def n_and_grad(f_hz: float, ne_interp: Bilinear2D, x: float, y: float):
    """Return n(x,y) and ∇n using finite-diff on n^2 to be robust near cutoff."""
    ne = ne_interp(x, y)
    n2 = n2_from_ne(f_hz, ne)
    n = np.sqrt(n2)

    # step for finite-diff (km)
    h = 0.5*(ne_interp.dx + ne_interp.dy)
    n2_xp = n2_from_ne(f_hz, ne_interp(x+h, y)); n2_xm = n2_from_ne(f_hz, ne_interp(x-h, y))
    n2_yp = n2_from_ne(f_hz, ne_interp(x, y+h)); n2_ym = n2_from_ne(f_hz, ne_interp(x, y-h))

    # ∂n/∂x = (1/(2n)) ∂(n^2)/∂x  (and same for y). Guard n>0.
    if n > 0:
        dn_dx = 0.5*(n2_xp - n2_xm)/(2*h)/n
        dn_dy = 0.5*(n2_yp - n2_ym)/(2*h)/n
    else:
        dn_dx = dn_dy = 0.0
    return float(n), float(dn_dx), float(dn_dy)

@dataclass
class RayConfig:
    f_MHz: float
    el0_deg: float
    x0_km: float = 0.0
    y0_km: float = 0.0
    s_max_km: float = 4000.0   # TOTAL path length to integrate (km)
    ds_km: float = 0.5         # step in km along the ray
    y_ground_km: float = 0.0
    y_max_km: float = 1200.0
    x_max_km: float = 6000.0
    n2_floor: float = 1e-6     # treat n^2 below this as evanescent
    keep_every: int = 1

class RayTracer2D:
    def __init__(self, x_km, y_km, Ne_m3):
        self.ne = Bilinear2D(x_km, y_km, Ne_m3)

    @staticmethod
    def _rk4_step_arc(f_hz, ne_interp, r, T, ds):
        """One RK4 step for arc-length system: dr/ds=T; dT/ds=(∇n - (∇n·T)T)/n."""
        def rhs(r, T):
            x, y = r
            n, dn_dx, dn_dy = n_and_grad(f_hz, ne_interp, x, y)
            # stop forcing if near cutoff to avoid singular accel
            if n <= 1e-6:
                a = np.array([0.0, 0.0])
            else:
                grad = np.array([dn_dx, dn_dy])
                a = (grad - (grad @ T)*T)/n  # curvature vector
            return T, a

        # k1
        k1_r, k1_T = rhs(r, T)
        # k2
        k2_r, k2_T = rhs(r + 0.5*ds*k1_r, T + 0.5*ds*k1_T)
        # k3
        k3_r, k3_T = rhs(r + 0.5*ds*k2_r, T + 0.5*ds*k2_T)
        # k4
        k4_r, k4_T = rhs(r + ds*k3_r,     T + ds*k3_T)

        r_new = r + (ds/6.0)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
        T_new = T + (ds/6.0)*(k1_T + 2*k2_T + 2*k3_T + k4_T)
        # renormalize T to unit length (controls drift)
        nrm = np.hypot(T_new[0], T_new[1])
        if nrm > 0:
            T_new /= nrm
        return r_new, T_new

    def trace(self, cfg: RayConfig):
        f_hz = cfg.f_MHz * 1e6

        # initial position & unit tangent from elevation (CW-from-north not needed in 2D)
        r = np.array([cfg.x0_km, cfg.y0_km], float)
        el = np.deg2rad(cfg.el0_deg)
        T = np.array([np.cos(el), np.sin(el)], float)   # (x,y) components; |T|=1

        s_vals, xs, ys, ns = [], [], [], []
        reason = "max_s_reached"
        steps = int(np.ceil(cfg.s_max_km / cfg.ds_km))
        last_above = True

        for i in range(steps):
            if i % cfg.keep_every == 0:
                n_here, *_ = n_and_grad(f_hz, self.ne, r[0], r[1])
                s_vals.append(i*cfg.ds_km); xs.append(r[0]); ys.append(r[1]); ns.append(n_here)

            # termination checks BEFORE step
            if not self.ne.inside(r[0], r[1]):
                reason = "out_of_bounds"; break
            if abs(r[0]) > cfg.x_max_km or r[1] > cfg.y_max_km:
                reason = "domain_limit"; break
            n2_here = n2_from_ne(f_hz, self.ne(r[0], r[1]))
            if n2_here < cfg.n2_floor:
                reason = "evanescent"; break
            if i > 0 and last_above and (r[1] <= cfg.y_ground_km):
                reason = "ground_hit"; break
            last_above = (r[1] > cfg.y_ground_km - 1e-6)

            # advance one arc-length step
            r, T = self._rk4_step_arc(f_hz, self.ne, r, T, cfg.ds_km)

        return {
            "s_km": np.asarray(s_vals),
            "x_km": np.asarray(xs),
            "y_km": np.asarray(ys),
            "n":    np.asarray(ns),
            "f_MHz": cfg.f_MHz,
            "el0_deg": cfg.el0_deg,
            "reason": reason,
        }

# ========================== Example / quick test =======================
if __name__ == "__main__":
    # Build a demo Ne(x,y) field: background + Chapman bump centered at x=1200 km
    x = np.linspace(0, 3000, 601)      # km
    y = np.linspace(0, 1000, 401)      # km
    X, Y = np.meshgrid(x, y)           # [Ny, Nx]

    # background low density + F2-like bump that varies with x
    NmF2 = 1e12 * (1.0 + 0.15*np.exp(-((X-1200)/400)**2))
    hmF2 = 300.0 + 30.0*np.exp(-((X-1200)/600)**2)
    H    = 50.0
    z = (Y - hmF2)/H
    Ne = NmF2 * np.exp(0.5*(1.0 - z - np.exp(-z))) + 2e10   # add floor

    rt = RayTracer2D(x, y, Ne)

    cfg = RayConfig(
        f_MHz=8.0,
        el0_deg=90,
        x0_km=500.0,
        y0_km=0.0,
        s_max_km=3000.0,   # allow enough total path
        ds_km=0.5,         # 0.25–1.0 km is a good starting step
        y_max_km=1100.0,
        x_max_km=4000.0,
        keep_every=1
    )

    out = rt.trace(cfg)
    print("Termination:", out["reason"])
    print("Max height (km):", out["y_km"].max() if out["y_km"].size else None)
    print("Ground range (km):", out["x_km"][-1] if out["x_km"].size else None)

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,4))
        # quick background of NmF2 proxy (not to scale), then ray path
        plt.contourf(x, y, np.log10(Ne), levels=25)
        plt.colorbar(label="log10 Ne (m^-3)")
        plt.plot(out["x_km"], out["y_km"], "w-", lw=2)
        plt.axhline(0, color="k", lw=1)
        plt.ylim(0, cfg.y_max_km)
        plt.xlim(0, x[-1])
        plt.xlabel("Ground range x (km)")
        plt.ylabel("Height y (km)")
        plt.title(f"2-D HF ray, f={cfg.f_MHz} MHz, el0={cfg.el0_deg}°, reason={out['reason']}")
        plt.tight_layout()
        plt.savefig("out.png")
    except Exception as e:
        print("Plot skipped:", e)
