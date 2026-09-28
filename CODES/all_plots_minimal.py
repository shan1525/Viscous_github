#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.random as npr

# ============================================================
# Fix Python path
# ============================================================
here = os.path.dirname(os.path.abspath(__file__))
simplemc_root = os.path.abspath(os.path.join(here, "..", ".."))
project_root  = os.path.dirname(simplemc_root)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.models.ViscCosmology import ViscCosmology
from simplemc.cosmo.paramDefs import (Om_par, h_par, w_par, gam_par, alp_par, n_par, s_par)
from fgivenx import plot_contours

# ============================================================
# Plot style (PRD-like)
# ============================================================
mpl.rcParams.update({
    "font.size": 16,
    "font.family": "serif",
    "text.usetex": True,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 13,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.figsize": (8.5, 6.5),
})

LCDM_COLOR = "black"
VISC_COLOR = "darkred"
VISC_CMAP  = plt.cm.Reds_r

# ============================================================
# Load viscous chain
# ============================================================
chain_dir = os.path.join(simplemc_root, "chains", "visc25")
chain_file = os.path.join(
    chain_dir,
    "Visc_sfixed_phy_Union3+DR2+PLK18_nested_multi_posterior.txt"
)

print(f"[INFO] Loading chain: {chain_file}")

raw = np.loadtxt(chain_file)
raw = np.atleast_2d(raw)
raw = raw[np.isfinite(raw).all(axis=1)]

# Om, Obh2, h, w, gam, alp, n, s
cols = [0, 2, 3, 4, 5, 6]
samples = raw[:, cols]
weights = np.ones(len(samples))

print(f"[INFO] Loaded {len(samples)} samples")

# ============================================================
# Build visc model
# ============================================================
model = ViscCosmology(
    vary_w=True,
    vary_gam=True,
    vary_alp=True,
    vary_n=True,
    vary_s=False,
    use_sn_M=False,
    use_cmb=True
)

# ============================================================
# DM/(z DH)
# ============================================================
def visc_dm_ratio(z_array, theta):

    Om0, h0, w0, gam, alp, n = theta

    if not (
        0.0 < Om0 < 1.5 and
        0.4 < h0 < 1.0 and
        -2.5 < w0 < -0.33 and
        alp >= 0.0 and
        0.0 <= n <= 0.8
    ):
        return np.nan * np.ones_like(z_array)

    Om_par.setValue(Om0)
    h_par.setValue(h0)
    w_par.setValue(w0)
    gam_par.setValue(gam)
    alp_par.setValue(alp)
    n_par.setValue(n)
    #s_par.setValue(s)

    ok = model.updateParams([Om_par, h_par, w_par, gam_par, alp_par, n_par])
    if not ok:
        return np.nan * np.ones_like(z_array)

    vals = []
    for z in z_array:
        if z <= 0:
            vals.append(np.nan)
            continue
        Da = model.DaOverrd(z)
        HI = model.HIOverrd(z)
        vals.append(Da / (z * HI))

    return np.array(vals)


# ============================================================
# DV/(rd z^{2/3})
# ============================================================
def visc_dv_ratio(z_array, theta):

    Om0, h0, w0, gam, alp, n = theta

    if not (
        0.0 < Om0 < 1.5 and
        0.4 < h0 < 1.0 and
        -2.5 < w0 < -0.33 and
        alp >= 0.0 and
        0.0 <= n <= 0.8
    ):
        return np.nan * np.ones_like(z_array)

    Om_par.setValue(Om0)
    h_par.setValue(h0)
    w_par.setValue(w0)
    gam_par.setValue(gam)
    alp_par.setValue(alp)
    n_par.setValue(n)
    #s_par.setValue(s)

    ok = model.updateParams([Om_par, h_par, w_par, gam_par, alp_par, n_par])
    if not ok:
        return np.nan * np.ones_like(z_array)

    vals = []
    for z in z_array:
        if z <= 0:
            vals.append(np.nan)
            continue
        dv = model.DVOverrd(z) / (z**(2/3))
        vals.append(dv)

    return np.array(vals)


# ============================================================
# Plot 1: DM/(zDH)
# ============================================================
def plot_dm_dh():

    plt.figure()

    # ============================
    # DESI DATA (clean)
    # ============================
    z_vals = {
        "LRG1": 0.510,
        "LRG2": 0.706,
        "LRG3+ELG1": 0.930,
        "ELG2": 1.317,
        "LyaQSO": 2.330
    }

    DM_data = {
        "LRG1": (13.62, 0.25),
        "LRG2": (16.85, 0.32),
        "LRG3+ELG1": (21.71, 0.28),
        "ELG2": (27.79, 0.69),
        "LyaQSO": (39.71, 0.94)
    }

    DH_data = {
        "LRG1": (20.98, 0.61),
        "LRG2": (20.08, 0.60),
        "LRG3+ELG1": (17.88, 0.35),
        "ELG2": (13.82, 0.42),
        "LyaQSO": (8.52, 0.17)
    }

    colors = {
        "LRG1": "red",
        "LRG2": "blue",
        "LRG3+ELG1": "green",
        "ELG2": "orange",
        "LyaQSO": "purple"
    }

    for key in DM_data:

        z = z_vals[key]

        DM, DM_err = DM_data[key]
        DH, DH_err = DH_data[key]

        # --- ratio
        ratio = DM / (z * DH)

        # --- propagated error (diagonal covariance)
        ratio_err = ratio * np.sqrt((DM_err / DM)**2 +(DH_err / DH)**2)

        plt.errorbar(z,ratio,yerr=ratio_err,fmt="s",markersize=7,color=colors[key],label=key)

    # ============================
    # LCDM
    # ============================
    lcdm = LCDMCosmology()
    Om_par.setValue(0.3012)
    h_par.setValue(0.6848)
    lcdm.updateParams([Om_par, h_par])

    z = np.linspace(0.4, 2.5, 200)
    lcdm_vals = [lcdm.DaOverrd(x)/(x*lcdm.HIOverrd(x)) for x in z]

    plt.plot(z, lcdm_vals, "--", color="black", lw=2, label=r"$\Lambda$CDM")

    # ============================
    # FGIVENX VISC
    # ============================
    plot_contours(visc_dm_ratio,z,samples,weights=weights,contour_line_levels=[1, 2],colors=plt.cm.Reds_r,ax=plt.gca(),)

    Y = np.array([visc_dm_ratio(z, s) for s in samples[:2000]])
    p50 = np.nanmedian(Y, axis=0)

    plt.plot(z, p50, color="darkred", lw=2.5, label="Viscous Minimal DE")

    plt.xlabel(r"$z$")
    plt.ylabel(r"$D_M/(z D_H)$")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("visc_dm_minimal.pdf", dpi=300)
    plt.close()

# ============================================================
# Plot 2: DV/(rd z^{2/3})
# ============================================================
def plot_dv():

    plt.figure()

    # ============================
    # DESI DV DATA
    # ============================
    z_vals = {"BGS": 0.295,"QSO": 1.491,}

    desi_data = [
        (z_vals["BGS"], 7.93, 0.15, "purple", r"$\mathrm{BGS}$"),
        (z_vals["QSO"], 26.07, 0.67, "cyan", r"$\mathrm{QSO}$"),
    ]

    for z, DV, err, color, label in desi_data:

        val = DV / (z**(2/3))
        err = err / (z**(2/3))

        plt.errorbar(z,val,yerr=err,fmt="s",markersize=7,color=color,label=label,)

    # ============================
    # LCDM
    # ============================
    lcdm = LCDMCosmology()
    Om_par.setValue(0.3012)
    h_par.setValue(0.6848)
    lcdm.updateParams([Om_par, h_par])

    z = np.linspace(0.25, 2.0, 200)
    lcdm_vals = [lcdm.DVOverrd(x)/(x**(2/3)) for x in z]

    plt.plot(z, lcdm_vals, "--", color="black", lw=2, label=r"$\Lambda$CDM")

    # ============================
    # FGIVENX VISC
    # ============================
    plot_contours(visc_dv_ratio,z,samples,weights=weights,contour_line_levels=[1, 2],colors=plt.cm.Reds_r,ax=plt.gca(),)

    Y = np.array([visc_dv_ratio(z, s) for s in samples[:2000]])
    p50 = np.nanmedian(Y, axis=0)

    plt.plot(z, p50, color="darkred", lw=2.5, label="Viscous Minimal DE")

    plt.xlabel(r"$z$")
    plt.ylabel(r"$D_V/(r_d z^{2/3})$")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("visc_dv_minimal.pdf", dpi=300)
    plt.close()


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    plot_dm_dh()
    plot_dv()
