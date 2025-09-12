
import numpy as np
from pynasonde.model.trace.ionosphere import IonosphereModels
from pynasonde.model.trace.rt2d import RayTracer2D

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
    hmf2_tilt_funct = lambda dx: (-0.1*dx),
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
    rp = rt.plot_fan(X, Z, Ne, figure_file_name=figure_file_name)
    return X, Z, Ne, outputs

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
):
    el_angles = np.concatenate((el_angles, homing_roots))
    X, Z, Ne, alpha_X, Nex = IonosphereModels.cusp_function_alpha(
        x, hs, layer_names, layer_heights, layer_base_ne,
        layer_scales, Ne_floor, x_params, d_params
    )
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
    rp = rt.plot_fan(X, Z, Nex, figure_file_name=figure_file_name)
    return X, Z, Ne, outputs

if __name__ == "__main__":
    for i, jx in enumerate(np.arange(-50, 50, 3)):
        if i == 9:
            x_params = np.asarray([-40, 0, 40]) + (jx*4)
            d_params = np.asarray([0.1, 0.1])
            ray_trace_2d_ionosphereic_wave_front(
                x_params=x_params,
                figure_file_name=f"figures/rt/wv{i}.png",
                ground_height_precision=1,
                el_angles = np.arange(50, 110, 0.1),
                homing_error_km=3,
            )
        # print(x_params)