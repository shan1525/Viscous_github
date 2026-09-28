#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# Fix path so simplemc is importable
# ============================================================
here = os.path.dirname(os.path.abspath(__file__))
simplemc_root = os.path.abspath(os.path.join(here, "..", ".."))
project_root  = os.path.dirname(simplemc_root)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================
# Imports
# ============================================================
from simplemc.models.ViscCosmology import ViscCosmology
from simplemc.cosmo.paramDefs import (
    Om_par, h_par, w_par, gam_par, alp_par, n_par, s_par
)

# ============================================================
# LaTeX styling
# ============================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "text.usetex": True,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

# ============================================================
# Model
# ============================================================
model = ViscCosmology(
    vary_w=True,
    vary_gam=True,
    vary_alp=True,
    vary_n=True,
    vary_s=True,
    use_sn_M=False,
    use_cmb=True
)

MODEL_LABEL = "Viscous Non-minimal DE"

fig, ax = plt.subplots(figsize=(8.5, 6))
# ============================================================
# LOAD POSTERIOR (visc chain)
# ============================================================
chain_dir = os.path.join(simplemc_root, "chains", "visc25")
chain_file = os.path.join(
    chain_dir,
    "Visc_phy_Union3+DR2+PLK18_nested_multi_posterior.txt"
)

print(f"[INFO] Loading chain: {chain_file}")

raw = np.loadtxt(chain_file)
raw = np.atleast_2d(raw)

# Remove NaNs
raw = raw[np.isfinite(raw).all(axis=1)]

# ------------------------------------------------------------
# IMPORTANT: match column ordering to paramnames file
# Typical order (CHECK .paramnames):
# Om, Obh2, h, M, w, gam, alp, n, s
# ------------------------------------------------------------
cols = [0, 2, 3, 4, 5, 6, 7]  # adjust if needed!

samples = raw[:, cols]
weights = np.ones(len(samples))

print(f"[INFO] Loaded {len(samples)} samples")

# ============================================================
# DEFINE w_d(z) FUNCTION FOR fgivenx
# ============================================================
def make_func(model, quantity="rho"):

    def func(z_array, theta):
        z_array = np.atleast_1d(z_array).astype(float)

        Om0 = float(theta[0])
        h0  = float(theta[1])
        w0  = float(theta[2])
        gam = float(theta[3])
        alp = float(theta[4])
        n   = float(theta[5])
        s   = float(theta[6])

        # sanity cuts
        if not (0.0 < Om0 < 1.5 and 0.4 < h0 < 1.0):
            return np.nan * np.ones_like(z_array)

        # set params
        Om_par.setValue(Om0)
        h_par.setValue(h0)
        w_par.setValue(w0)
        gam_par.setValue(gam)
        alp_par.setValue(alp)
        n_par.setValue(n)
        s_par.setValue(s)

        ok = model.updateParams([Om_par, h_par, w_par, gam_par, alp_par, n_par, s_par])
        if not ok:
            return np.nan * np.ones_like(z_array)

        vals = []
        for zi in z_array:

            if quantity == "rho":
                y = model.rhoD_z(zi)

            elif quantity == "rho_m":
                a = 1.0 / (1.0 + zi)
                y = model.rhoDM_a(a) + model._Omega_b0() * a**(-3.0)

            elif quantity == "rho_plus_p":
                y = model.rhoPlusP_z(zi)

            elif quantity == "Omega_d":
                a = 1.0 / (1.0 + zi)
                y = model.rhoD_z(zi) / model.RHSquared_a(a)

            else:
                y = np.nan

            vals.append(y if np.isfinite(y) else np.nan)

        return np.array(vals)

    return func

def posterior_median(func, z_array, max_samples=500):
    number = min(max_samples, len(samples))
    indices = np.linspace(0, len(samples) - 1, number, dtype=int)
    values = np.array([func(z_array, samples[i]) for i in indices])
    return np.nanmedian(values, axis=0)


def posterior_values_at_z(func, z, max_samples=1000):
    number = min(max_samples, len(samples))
    indices = np.linspace(0, len(samples) - 1, number, dtype=int)
    values = np.array([func(np.array([z]), samples[i])[0] for i in indices])
    return values[np.isfinite(values)]

# ============================================================
# z grid
# ============================================================
z_grid = np.linspace(0.0, 5.0, 300)

rho_model = make_func(model, "rho")
rho_m_model = make_func(model, "rho_m")
Omega_d_model = make_func(model, "Omega_d")

fig, ax = plt.subplots(figsize=(8.5, 6))

from fgivenx import plot_contours

cset = plot_contours(
    rho_model,
    z_grid,
    samples,
    weights=weights,
    contour_line_levels=[1, 2],
    colors=plt.cm.viridis,
    ax=ax,
)

cbar = plt.colorbar(cset, ax=ax, ticks=[0, 1, 2])
cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$'])
rho_d_median = posterior_median(rho_model, z_grid)
ax.plot(z_grid, rho_d_median, color="darkred", linewidth=2.0, label=MODEL_LABEL)

z_inset = np.linspace(0.0, 10.0, 250)
rho_d_inset = posterior_median(rho_model, z_inset)
rho_m_inset = posterior_median(rho_m_model, z_inset)

inset = ax.inset_axes([0.08, 0.54, 0.44, 0.38])
inset.set_facecolor("white")
inset.patch.set_alpha(0.95)
inset.set_zorder(5)

inset.plot(z_inset, rho_d_inset, color="darkred", linewidth=1.8, label=MODEL_LABEL)
inset.plot(z_inset, rho_m_inset, color="navy", linewidth=1.8, label=r"Matter, $\rho_m$")
inset.set_xlim(0.0, 10.0)
inset.set_yscale("log")
inset.set_xlabel(r"$z$", fontsize=10)
inset.set_ylabel(r"$\rho_i/\rho_{c0}$", fontsize=10)
inset.tick_params(axis="both", direction="in", labelsize=9)
inset.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.5)
inset.grid(alpha=0.20)

for spine in inset.spines.values():
    spine.set_linewidth(1.1)
    spine.set_color("black")

ax.legend(frameon=False, loc="upper right")
ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$\rho_d(z)/\rho_{c0}$")


ax.grid(alpha=0.3)

# ============================================================
# Save
# ============================================================
outdir = os.path.join(simplemc_root, "chains", "Plots")
os.makedirs(outdir, exist_ok=True)

outfile = os.path.join(outdir, "visc_rhoD_fgiven.pdf")
plt.savefig(outfile, dpi=300, bbox_inches="tight")

print(f"[INFO] Saved → {outfile}")

print(f"[INFO] Saved → {outfile}")

rhoP_model = make_func(model, "rho_plus_p")

fig, ax = plt.subplots(figsize=(8.5, 6))

cset = plot_contours(
    rhoP_model,
    z_grid,
    samples,
    weights=weights,
    contour_line_levels=[1, 2],
    colors=plt.cm.inferno,
    ax=ax,
)

cbar = plt.colorbar(cset, ax=ax, ticks=[0, 1, 2])
cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$'])
rhoP_median = posterior_median(rhoP_model, z_grid)
ax.plot(z_grid, rhoP_median, color="darkred", linewidth=2.0, label=MODEL_LABEL)
# IMPORTANT physical line
ax.axhline(
    0.0,
    color="black",
    linestyle="--",
    linewidth=2.0,
    label=r"$\rho_d+p_d=0$"
)

ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$[\rho_d+p_d]/\rho_{c0}$")
#ax.set_title(r"$\rho_d(z)+p_d(z)$")

ax.grid(alpha=0.3)
ax.legend(frameon=False)

outfile = os.path.join(outdir, "visc_rhoPlusP_fgiven.pdf")
plt.savefig(outfile, dpi=300, bbox_inches="tight")

Omega_d_1100 = posterior_values_at_z(Omega_d_model, 1100.0)

if len(Omega_d_1100) > 0:
    median_1100 = np.percentile(Omega_d_1100, 50.0)
    upper_1100 = np.percentile(Omega_d_1100, 95.0)

    print(f"[INFO] {MODEL_LABEL}")
    print(f"[INFO] Median Omega_d(z=1100) = {median_1100:.6e}")
    print(f"[INFO] 95% upper Omega_d(z=1100) = {upper_1100:.6e}")
else:
    print(f"[WARNING] No valid Omega_d(z=1100) values for {MODEL_LABEL}")
print(f"[INFO] Saved → {outfile}")
#plt.show()
