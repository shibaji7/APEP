import numpy as np

# ------------------------- Physical constants -------------------------
EPS0 = 8.8541878128e-12   # vacuum permittivity [F/m]
QE   = 1.602176634e-19    # electron charge [C]
ME   = 9.10938356e-31     # electron mass [kg]
TWO_PI = 2.0*np.pi

# -------------------------- Plasma frequency --------------------------
def plasma_freq_hz(ne_m3: np.ndarray) -> np.ndarray:
    """
    Electron plasma frequency [Hz] from electron density ne [m^-3].
    f_p = (1/2π) * sqrt( (n_e e^2) / (ε0 m_e) )
    """
    omega_p = np.sqrt(ne_m3 * QE*QE / (EPS0 * ME))
    return omega_p / TWO_PI

# ---------------------- Chapman F2 electron density --------------------
def chapman_Ne_m3(h_km: np.ndarray,
                  NmF2: float = 8e12,   # peak electron density [m^-3]
                  hmF2_km: float = 300, # peak height [km]
                  H_km: float = 50      # scale height [km]
                  ) -> np.ndarray:
    """
    Classic Chapman α-layer (F2-like).
    N(h) = NmF2 * exp( 0.5 * (1 - z - exp(-z)) ),  z = (h - hmF2)/H
    """
    z = (np.asarray(h_km) - hmF2_km) / H_km
    return NmF2 * np.exp(0.5 * (1.0 - z - np.exp(-z)))

# --------------------------- Refractive index --------------------------
def refractive_index(f_hz: float, h_km: np.ndarray,
                     NmF2: float, hmF2_km: float, H_km: float) -> np.ndarray:
    """
    No-B, no-collisions Appleton-Hartree: n^2 = 1 - (f_p/f)^2.
    Values where f_p > f are clamped to a tiny positive n^2 -> n≈0 (turning region).
    """
    ne = chapman_Ne_m3(h_km, NmF2, hmF2_km, H_km)
    fp = plasma_freq_hz(ne)
    n2 = 1.0 - (fp / f_hz)**2
    # prevent numerical issues (imaginary refractive index region)
    n2 = np.maximum(n2, 1e-8)
    return np.sqrt(n2)

# ------------------------- Raytrace (2D, spherical) --------------------
def raytrace_2d_spherical(
    f_MHz: float,
    el0_deg: float,
    *,
    Re_km: float = 6371.0,
    h0_km: float = 0.0,
    NmF2: float = 8e12,
    hmF2_km: float = 300.0,
    H_km: float = 50.0,
    dr_km: float = 0.25,
    max_alt_km: float = 1000.0,
):
    """
    2-D great-circle HF ray tracing in a spherically stratified ionosphere.

    Parameters
    ----------
    f_MHz : float
        Wave frequency in MHz.
    el0_deg : float
        Launch elevation angle (deg) above local horizontal at the ground.
    Re_km : float
        Earth radius [km].
    h0_km : float
        Launch altitude [km] (0 for sea level site).
    NmF2, hmF2_km, H_km : Chapman layer parameters.
    dr_km : float
        Radial step [km] for integration (adaptive near turning by internal clamp).
    max_alt_km : float
        Safety altitude to abort if the ray does not turn (e.g., too high frequency).

    Returns
    -------
    result : dict
        {
          "f_MHz": ...,
          "el0_deg": ...,
          "turning_height_km": float or None,
          "one_hop_ground_range_km": float or None,
          "r_km": array of radii for up-leg,
          "phi_rad": array of central angles for up-leg,
          "n": array of refractive index along up-leg
        }
    """
    f_hz = f_MHz * 1e6

    # Starting radius and launch geometry
    r0 = Re_km + h0_km
    el0 = np.deg2rad(el0_deg)
    # Ray invariant I = n0 r0 cos(el0); near ground, n0 ~ 1
    I = r0 * np.cos(el0)

    # Storage
    r_list = [r0]
    phi_list = [0.0]   # central angle from launch site
    n_list = [1.0]

    # Integrate upward until turning point: (n * r) -> I
    r = r0
    phi = 0.0
    turning = False

    # Cap for safety
    r_max = Re_km + max_alt_km

    while r < r_max:
        h = r - Re_km
        n = refractive_index(f_hz, h, NmF2, hmF2_km, H_km)

        # Check turning proximity: I/(n r) -> 1
        q = I / (n * r)
        if q >= 1.0:
            turning = True
            break

        # dphi/dr = (1/r) * (q / sqrt(1-q^2))
        denom = np.sqrt(max(1e-12, 1.0 - q*q))
        dphidr = (q / denom) / r

        # adaptive step near turning
        step = min(dr_km, max(0.05, 0.2 * (1.0 - q)) * dr_km * 5)

        # advance one step
        r_next = r + step
        phi_next = phi + dphidr * step

        r_list.append(r_next)
        phi_list.append(phi_next)
        n_list.append(n)

        r, phi = r_next, phi_next

    if not turning:
        return dict(
            f_MHz=f_MHz,
            el0_deg=el0_deg,
            turning_height_km=None,
            one_hop_ground_range_km=None,
            r_km=np.array(r_list),
            phi_rad=np.array(phi_list),
            n=np.array(n_list),
        )

    # Turning at current r where q ~ 1
    turning_height = r - Re_km
    # One-hop ground range is ~ 2 * phi_turn * Re
    one_hop_ground = 2.0 * phi * Re_km

    return dict(
        f_MHz=f_MHz,
        el0_deg=el0_deg,
        turning_height_km=float(turning_height),
        one_hop_ground_range_km=float(one_hop_ground),
        r_km=np.array(r_list),
        phi_rad=np.array(phi_list),
        n=np.array(n_list),
    )

# ----------------------------- Convenience -----------------------------
def multi_hop_ground_range(
    f_MHz: float,
    el0_deg: float,
    hops: int = 1,
    **kwargs
):
    """
    Repeat the 1-hop solution from raytrace_2d_spherical to estimate multi-hop range.
    (Assumes symmetric hops and identical ionosphere.)
    """
    out = raytrace_2d_spherical(f_MHz, el0_deg, **kwargs)
    if out["one_hop_ground_range_km"] is None:
        return None
    return hops * out["one_hop_ground_range_km"]


if __name__ == "__main__":
    # Example: 7 MHz, 20° launch elevation, default Chapman (NmF2=8e12 m^-3, hmF2=300 km, H=50 km)
    out = raytrace_2d_spherical(
        f_MHz=1.0,
        el0_deg=40.0,
        NmF2=1e12,
        hmF2_km=300.0,
        H_km=55.0,
        dr_km=0.2,
        max_alt_km=1000.0,
    )

    print("Turning height (km):", out["turning_height_km"])
    print("1-hop ground range (km):", out["one_hop_ground_range_km"])

    # 3-hop estimate
    print("3-hop range (km):", multi_hop_ground_range(7.0, 20.0, hops=3, NmF2=8e12, hmF2_km=300.0, H_km=55.0))

    # Optional: plot up-leg ray path in (range, height) coordinates
    import matplotlib.pyplot as plt
    s_km = out["phi_rad"] * 6371.0  # arc length along Earth (approx for small angles)
    h_km = out["r_km"] - 6371.0

    plt.figure(figsize=(6,4))
    plt.plot(s_km, h_km)
    plt.axhline(0, color="k", lw=0.5)
    plt.xlim(-300, 300)
    plt.xlabel("Ground arc (km)")
    plt.ylabel("Altitude (km)")
    plt.title(f"HF Ray Up-leg: f={out['f_MHz']} MHz, el0={out['el0_deg']}°")
    plt.grid(True, ls=":")
    plt.savefig("out.png")
