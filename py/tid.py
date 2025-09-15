import numpy as np
import matplotlib.pyplot as plt

# --- Background ionosphere model (Chapman layer) ---
def background_electron_density(z, Nmax=1e12, hm=300e3, H=50e3):
    """Simple Chapman profile for background ionosphere."""
    return Nmax * np.exp(0.5 * (1 - (z - hm)/H - np.exp(-(z - hm)/H)))

# --- TID perturbation (AGW-like sinusoidal form, Section 3.3) ---
def tid_perturbation(x, z, A=0.1, kx=2*np.pi/200e3, kz=2*np.pi/300e3, phi=0):
    """
    TID density perturbation as fraction of background.
    A : relative amplitude (10% default)
    kx: horizontal wavenumber (default: λx=200 km)
    kz: vertical wavenumber (default: λz=300 km)
    """
    return A * np.sin(kx*x + kz*z + phi)

# --- Total perturbed ionosphere ---
def perturbed_density(x, z):
    Ne0 = background_electron_density(z)
    dNe = Ne0 * tid_perturbation(x, z)
    return Ne0 + dNe

# --- Grid setup ---
x = np.linspace(0, 600e3, 400)   # horizontal range (m)
z = np.linspace(100e3, 500e3, 300) # altitude (m)
X, Z = np.meshgrid(x, z)

# --- Compute densities ---
Ne_background = background_electron_density(Z)
Ne_total = perturbed_density(X, Z)

# --- Plotting ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Background only
c1 = axs[0].pcolormesh(x/1e3, z/1e3, Ne_background, shading='auto', cmap='viridis')
axs[0].set_title("Background Ionosphere")
axs[0].set_xlabel("Horizontal distance (km)")
axs[0].set_ylabel("Altitude (km)")
fig.colorbar(c1, ax=axs[0], label="Ne (m⁻³)")

# Perturbed (background + TID)
c2 = axs[1].pcolormesh(x/1e3, z/1e3, Ne_total, shading='auto', cmap='viridis')
axs[1].set_title("Perturbed Ionosphere with TID")
axs[1].set_xlabel("Horizontal distance (km)")
fig.colorbar(c2, ax=axs[1], label="Ne (m⁻³)")

plt.tight_layout()
plt.show()
