
import numpy as np
from pynasonde.model.trace.ionosphere import IonosphereModels
from pynasonde.model.trace.rt2d import RayTracer2D, plasma_freq_hz
from pynasonde.model.trace.plottrace import PlotRays

# ========================== Examples =======================
# Build a demo Ne(x,y) field: background + Chapman bump centered at x
def ray_trace_2d_ionosphereic_bump(
    x: np.ndarray = np.linspace(-1500, 1500, 601),  # horizontal distance [km]
    hs: np.ndarray = np.linspace(0, 1000, 501),      # altitude [km]
    NmF2: float = 1e12,                             # peak density [m^-3]
    hmF2: float = 300.0,                            # F2 peak height [km]
    nmf2_funct = lambda dx: (1.0 + 0.15*np.exp(-((dx+30)/100)**2)),
    hmf2_funct = lambda dx: (30.0*np.exp(-((dx+30)/100)**2)),
    H_scale:float = 50.0,  # scale height [km]
    Ne_floor:float = 2e10,
    frequencies: np.ndarray = np.asarray([8]),
    el_angles: np.ndarray = np.arange(50, 130, 5),
    x0_km=0.0,
    y0_km=0.0,
    s_max_km=3000.0,   # allow enough total path
    ds_km=0.05,         # 0.25–1.0 km is a good starting step
    y_max_km=1100.0,
    x_max_km=4000.0,
    keep_every=1,
    figure_file_name=None,
):
    X, Z, Ne = IonosphereModels.create_chapman_ionosphere_bump(
        x, hs, NmF2, hmF2, nmf2_funct, 
        hmf2_funct, H_scale, Ne_floor
    )
    rt = RayTracer2D(x, hs, Ne)
    outputs = rt.run_all_rays(
        frequencies, el_angles,
        x0_km, y0_km, s_max_km, ds_km,
        y_max_km, x_max_km, keep_every,
    )
    rp = rt.plot_fan(X, Z, Ne, figure_file_name=figure_file_name)
    return X, Z, Ne, outputs


# Build a demo Ne(x,y) field: background + Chapman tilt centered
def ray_trace_2d_ionosphereic_tilt(
    x: np.ndarray = np.linspace(-1500, 1500, 601),  # horizontal distance [km]
    hs: np.ndarray = np.linspace(0, 1000, 501),      # altitude [km]
    NmF2: float = 1e12,                             # peak density [m^-3]
    hmF2: float = 300.0,                            # F2 peak height [km]
    H_scale:float = 50.0,  # scale height [km]
    Ne_floor:float = 2e10,
    hmf2_tilt_funct = lambda dx: (-0.2*dx),
    frequencies: np.ndarray = np.asarray([8]),
    el_angles: np.ndarray = np.arange(50, 130, 5),
    x0_km=0.0,
    y0_km=0.0,
    s_max_km=3000.0,   # allow enough total path
    ds_km=0.01,         # 0.25–1.0 km is a good starting step
    y_max_km=1100.0,
    x_max_km=4000.0,
    keep_every=1,
    figure_file_name=None,
    homing_roots: np.ndarray = np.arange(72, 85, 1),
):
    X, Z, Ne = IonosphereModels.chapman_with_tilted_hmf2(
        x, hs, NmF2, hmF2, H_scale, Ne_floor, hmf2_tilt_funct
    )
    rt = RayTracer2D(x, hs, Ne)
    outputs = rt.run_all_rays(
        frequencies, el_angles,
        x0_km, y0_km, s_max_km, ds_km,
        y_max_km, x_max_km, keep_every,
    )
    rp = rt.plot_fan(X, Z, Ne, homing_roots=homing_roots, figure_file_name=figure_file_name)
    return X, Z, Ne, outputs

# Build a demo Ne(x,y) field: background + Chapman tilt centered
def ray_trace_2d_ionosphereic_obs(
    x: np.ndarray = np.linspace(-1500, 1500, 601),  # horizontal distance [km]
    hs: np.ndarray = np.linspace(0, 1000, 501),      # altitude [km]
    NmF2: float = 1e12,                             # peak density [m^-3]
    hmF2: float = 300.0,                            # F2 peak height [km]
    H_scale:float = 50.0,  # scale height [km]
    Ne_floor:float = 2e10,
    hmf2_tilt_funct = lambda dx: (-0.2*dx),
    frequencies: np.ndarray = np.asarray([7.5]),
    el_angles: np.ndarray = np.arange(75, 105, 0.5),
    x0_km=0.0,
    y0_km=0.0,
    s_max_km=3000.0,   # allow enough total path
    ds_km=0.01,         # 0.25–1.0 km is a good starting step
    y_max_km=1100.0,
    x_max_km=4000.0,
    keep_every=1,
    figure_file_name=None,
    homing_roots: np.ndarray = np.asarray([]),
    obscuration_peak_km: float = -300,
    obscuration_half_width_km: float = 450.0,
    ground_height_precision=2,
    homing_error_km=3,
):
    X, Z, Ne = IonosphereModels.chapman_with_grading_obscuration(
        x, hs, NmF2, hmF2, H_scale, Ne_floor,
        obscuration_peak_km = obscuration_peak_km,
        obscuration_half_width_km = obscuration_half_width_km,
    )
    rt = RayTracer2D(x, hs, Ne)
    outputs = rt.run_all_rays(
        frequencies, el_angles,
        x0_km, y0_km, s_max_km, ds_km,
        y_max_km, x_max_km, keep_every,
    )

    homing_roots = rt.homing_in_to_locate_roots(
        outputs, 
        ground_height_precision=ground_height_precision,
        homing_error_km=homing_error_km
    )
    ox = []
    for o in outputs:
        if (o.el0_deg in homing_roots) or (o.el0_deg.is_integer()):
            ox.append(o)
    txt = r"$\phi=[%.1f^{\circ},%.1f^{\circ}]$"%(el_angles[0], el_angles[-1]) 
    txt += "\n"
    txt += r"$f_0$=[%.1f] MHz"%(frequencies[0])
    rp = rt.plot_fan(X, Z, Ne, outputs=ox, figure_file_name=figure_file_name, text=txt)
    return X, Z, Ne, outputs, homing_roots, ox



def ray_trace_2d_ionosphereic_tid(
    x: np.ndarray = np.linspace(-1500, 1500, 601),  # horizontal distance [km]
    hs: np.ndarray = np.linspace(0, 1000, 501),  # altitude [km]
    NmF2: float = 3e11,
    hmF2: float = 300.0,
    H_scale: float = 50.0,
    Ne_floor: float = 2e10,
    A=1e-1,
    kx=2 * np.pi / 100,
    kz=2 * np.pi / 200,
    phi=0,
    frequencies: np.ndarray = np.asarray([5]),
    el_angles: np.ndarray = np.arange(80, 100, 0.5),
    homing_roots: np.ndarray = np.asarray([]),
    x0_km=0.0,
    y0_km=0.0,
    s_max_km=3000.0,   # allow enough total path
    ds_km=0.05,         # 0.25–1.0 km is a good starting step
    y_max_km=1100.0,
    x_max_km=4000.0,
    keep_every=1,
    figure_file_name=None,
    ground_height_precision=2,
    homing_error_km=3,
):
    X, Z, Ne, ndNe = IonosphereModels.cusp_function_tids(
        x, hs, NmF2, hmF2, H_scale, Ne_floor, A, kx, kz, phi
    )
    rt = RayTracer2D(x, hs, ndNe)
    outputs = rt.run_all_rays(
        frequencies, el_angles,
        x0_km, y0_km, s_max_km, ds_km,
        y_max_km, x_max_km, keep_every,
    )
    homing_roots = rt.homing_in_to_locate_roots(
        outputs, 
        ground_height_precision=ground_height_precision,
        homing_error_km=homing_error_km
    )

    ox = []
    for o in outputs:
        if (o.el0_deg in homing_roots) or (o.el0_deg.is_integer()):
            ox.append(o)
    txt = r"$\phi=[%.1f^{\circ},%.1f^{\circ}]$"%(el_angles[0], el_angles[-1]) 
    txt += "\n"
    txt += r"$f_0$=[%.1f] MHz"%(frequencies[0])

    rp = rt.plot_fan(X, Z, ndNe, outputs=ox, figure_file_name=figure_file_name, text=txt, pf_lim=[1,5])
    return X, Z, ndNe, outputs, homing_roots, ox

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
    frequencies: np.ndarray = np.asarray([8.3]),
    el_angles: np.ndarray = np.arange(50, 110, 0.5),
    homing_roots: np.ndarray = np.asarray([]),
    x0_km=0.0,
    y0_km=0.0,
    s_max_km=3000.0,   # allow enough total path
    ds_km=0.05,         # 0.25–1.0 km is a good starting step
    y_max_km=1100.0,
    x_max_km=4000.0,
    keep_every=1,
    figure_file_name=None,
    ground_height_precision=2,
    homing_error_km=50,
    eclipse_mask = 0.,
):
    delta = 1 - np.min([eclipse_mask, 1])
    el_angles = np.concatenate((el_angles, homing_roots))
    X, Z, Ne, alpha_X, Nex = IonosphereModels.cusp_function_alpha(
        x, hs, layer_names, layer_heights, layer_base_ne,
        layer_scales, Ne_floor, x_params, d_params
    )
    Nex = delta * Nex
    rt = RayTracer2D(x, hs, Nex)
    outputs = rt.run_all_rays(
        frequencies, el_angles,
        x0_km, y0_km, s_max_km, ds_km,
        y_max_km, x_max_km, keep_every,
    )
    homing_roots = rt.homing_in_to_locate_roots(
        outputs, 
        ground_height_precision=ground_height_precision,
        homing_error_km=homing_error_km
    )
    ox = []
    for o in outputs:
        if (o.el0_deg in homing_roots) or (o.el0_deg.is_integer()):
            ox.append(o)
    txt = r"$\phi=[%.1f^{\circ},%.1f^{\circ}]$"%(el_angles[0], el_angles[-1]) 
    txt += "\n"
    txt += r"$f_0$=[%.1f] MHz"%(frequencies[0])
    rp = rt.plot_fan(X, Z, Nex, outputs=ox, figure_file_name=figure_file_name, text=txt)
    return X, Z, Nex, outputs, homing_roots, ox


def simulate_bifurcations():
    response, dphi, f0 = [], 0.2, 8.3
    response.append(
        ray_trace_2d_ionosphereic_wave_front(
            figure_file_name="figures/rt/bifurcation_0.png",
            ground_height_precision=1,
            x_params = np.asarray([-40, 0, 40]),
            d_params = np.asarray([0, 0]),
            frequencies = np.asarray([f0]),
            el_angles = np.concatenate(
                [
                    np.round(np.arange(75, 90, dphi), 1),
                    np.round(np.arange(90, 105.2, dphi), 1),
                ]
            ),
            homing_error_km=3,
            eclipse_mask=0.0
        )
    )
    response.append(
        ray_trace_2d_ionosphereic_wave_front(
            figure_file_name="figures/rt/bifurcation_1.png",
            ground_height_precision=1,
            x_params = np.asarray([-40, 0, 40]),
            d_params = np.asarray([0, 0]),
            frequencies = np.asarray([f0]),
            el_angles = np.concatenate(
                [
                    np.round(np.arange(75, 90, dphi), 1),
                    np.round(np.arange(90, 105.2, dphi), 1),
                ]
            ),
            homing_error_km=3,
            eclipse_mask=0.2
        )
    )
    response.append(
        ray_trace_2d_ionosphereic_obs(
            frequencies = np.asarray([f0]),
            el_angles = np.concatenate(
                [
                    np.round(np.arange(75, 90, dphi), 1),
                    np.round(np.arange(90, 105.2, dphi), 1),
                ]
            ),
            homing_error_km=3,
        )
    )
    response.append(
        ray_trace_2d_ionosphereic_obs(
            frequencies = np.asarray([f0]),
            el_angles = np.concatenate(
                [
                    np.round(np.arange(75, 90, dphi), 1),
                    np.round(np.arange(90, 105.2, dphi), 1),
                ]
            ),
            homing_error_km=3,
            obscuration_peak_km=300,
        )
    )
    response.append(
        ray_trace_2d_ionosphereic_wave_front(
            figure_file_name="figures/rt/bifurcation_2.png",
            ground_height_precision=1,
            x_params = np.asarray([-40, 0, 40]),
            frequencies = np.asarray([f0]),
            el_angles = np.concatenate(
                [
                    np.round(np.arange(75, 90, dphi), 1),
                    np.round(np.arange(90, 105.2, dphi), 1),
                ]
            ),
            homing_error_km=3,
            eclipse_mask=0.2
        )
    )
    response.append(
        ray_trace_2d_ionosphereic_wave_front(
            figure_file_name="figures/rt/bifurcation_3.png",
            ground_height_precision=1,
            x_params = np.asarray([-40, 0, 40]),
            d_params= np.asarray([0.4, 0.05]),
            frequencies = np.asarray([f0]),
            el_angles = np.concatenate(
                [
                    np.round(np.arange(75, 90, dphi), 1),
                    np.round(np.arange(90, 105.2, dphi), 1),
                ]
            ),
            homing_error_km=3,
            eclipse_mask=0.2
        )
    )
    rp = PlotRays(nrows=3, ncols=2, ylim=[-100, 600], xlim=[-150, 150])
    texts = ["Pre-eclipse", r"$\mathcal{O}=20\%$", r"$\mathcal{O}=90\%$", r"$\mathcal{O}=90\%$", r"$\mathcal{O}=30\%$", r"$\mathcal{O}=30\%$"]
    for j, resp in enumerate(response):
        rp.set_density(resp[0], resp[1], resp[2], plasma_freq_hz(resp[2]) / 1e6)
        rp.lay_rays(
            resp[5],
            kind="pf",
            ped_angles=resp[4],
            text=fr"({chr(65+j)}) " + texts[j],
            xlabel=r"Ground range, km" if j in [2, 3] else "",
            ylabel=r"Height, km" if j in [0, 2] else "",
            add_cbar=j==3
        )
    rp.fig.get_axes()[0].text(
        0.05, 1.05,
        r"$f_0$=8.3 MHz, $\theta=[75^{\circ}, 105^{\circ}]$",
        ha="left", va="center",
        transform=rp.fig.get_axes()[0].transAxes
    )
    rp.save("figures/rt/bifurcation.png")
    rp.save("manuscript_figures/Figure10.png")
    rp.close()
    return

def simulate_tid_cusp():
    response, dphi, f0 = [], 0.05, 5.15
    response.append(
        ray_trace_2d_ionosphereic_tid(
            figure_file_name="figures/rt/cusp0.png",
            ground_height_precision=1,
            frequencies = np.asarray([f0]),
            el_angles = np.concatenate(
                [
                    np.round(np.arange(75, 90, dphi*10), 1),
                    np.round(np.arange(90, 105.2, dphi*10), 1),
                ]
            ),
            homing_error_km=3,
        )
    )
    response.append(
        ray_trace_2d_ionosphereic_tid(
            kx = 2 * np.pi / 200,
            kz = 2 * np.pi / 500,
            A=0.4,
            figure_file_name="figures/rt/cusp1.png",
            ground_height_precision=1,
            frequencies = np.asarray([f0]),
            el_angles = np.concatenate(
                [
                    np.round(np.arange(75, 90, dphi), 2),
                    np.round(np.arange(90, 105.2, dphi), 2),
                ]
            ),
            homing_error_km=3,
        )
    )

    rp = PlotRays(nrows=1, ncols=2, ylim=[-100, 600], xlim=[-150, 150])
    texts = ["", ""]
    for j, resp in enumerate(response):
        rp.set_density(resp[0], resp[1], resp[2], plasma_freq_hz(resp[2]) / 1e6)
        rp.set_param_lims(pf_lim=[1,5])
        rp.lay_rays(
            resp[5],
            kind="pf",
            ped_angles=resp[4],
            text=fr"({chr(65+j)}) " + texts[j],
            xlabel=r"Ground range, km",
            ylabel=r"Height, km" if j in [0] else "",
            add_cbar=j==1
        )
    rp.fig.get_axes()[0].text(
        0.05, 1.05,
        r"$f_0$=5.15 MHz, $\theta=[75^{\circ}, 105^{\circ}]$",
        ha="left", va="center",
        transform=rp.fig.get_axes()[0].transAxes
    )
    rp.save("figures/rt/cusp_tid.png")
    rp.save("manuscript_figures/Figure11.png")
    rp.close()
    return

if __name__ == "__main__":
    # simulate_bifurcations()
    simulate_tid_cusp()