#!/usr/bin/env python3
import numpy as np
from scipy.constants import c, G, parsec
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
import matplotlib as mpl

# ================================================================
# PLOT STYLE
# ================================================================
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 20,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 16,
    "font.weight": "bold",
    "axes.labelweight": "bold",
})


# ================================================================
# PHYSICAL BULK VISCOSITY
# ================================================================
def physical_viscosity_today(theta, h_index, Om_index, alp_index, n_index):
    """
    Calculate the present physical bulk-viscosity coefficient:

        zeta_d(0) = sqrt(3)*alpha*H0*c^2/(8*pi*G)*Omega_d0^n

    Parameters
    ----------
    theta : ndarray
        Filtered chain samples.
    h_index, Om_index, alp_index, n_index : int
        Column indices in theta.

    Returns
    -------
    zeta_phys : ndarray
        Present physical viscosity in Pa s.
    """
    Mpc = 1.0e6 * parsec
    Neff = 3.046
    omega_gamma = 2.469e-5
    omega_r = omega_gamma * (1.0 + 0.2271 * Neff)

    Om = theta[:, Om_index]
    h = theta[:, h_index]
    alp = theta[:, alp_index]
    n = theta[:, n_index]

    H0_SI = 100.0 * h * 1000.0 / Mpc
    Or0 = omega_r / h**2
    Od0 = 1.0 - Om - Or0

    if np.any(Od0 <= 0.0):
        raise ValueError("Some samples have Omega_d0 <= 0.")

    zeta_phys = np.sqrt(3.0) * alp * H0_SI * c**2 * Od0**n / (8.0 * np.pi * G)

    return zeta_phys

# ================================================================
# LOAD UNION3
# ================================================================
chain_u3 = "/Users/shaz/Desktop/shan/SimpleMC_latest/SimpleMC-final/SimpleMC-latest/simplemc/chains/visc25/Visc_sfixed_phy_Union3+DR2+PLK18_nested_multi_posterior.txt"

raw1 = np.loadtxt(chain_u3)
raw1 = np.atleast_2d(raw1)
raw1 = raw1[np.isfinite(raw1).all(axis=1)]

cols1 = [0, 1, 2, 3, 4, 5, 6]
theta1 = raw1[:, cols1]

Om, Obh2, h, w, gam, alp, n = theta1.T

mask1 = (
    (w > -2.5) & (w < -0.33) &
    (n >= 0.0) & (n <= 0.8) &
    (gam > -0.3) & (gam < 0.3)
)

theta1 = theta1[mask1]
zeta_phys1 = physical_viscosity_today(
    theta1,
    h_index=2,
    Om_index=0,
    alp_index=5,
    n_index=6,
)

# ================================================================
# LOAD PANTHEON+
# ================================================================
chain_p = "/Users/shaz/Desktop/shan/SimpleMC_latest/SimpleMC-final/SimpleMC-latest/simplemc/chains/visc25/Visc_sfixed_phy_PantheonPlusSH0ES+DR2+PLK18_nested_multi_posterior.txt"

raw2 = np.loadtxt(chain_p)
raw2 = np.atleast_2d(raw2)
raw2 = raw2[np.isfinite(raw2).all(axis=1)]

cols2 = [0, 1, 2, 4, 5, 6, 7]
theta2 = raw2[:, cols2]

Om, Obh2, h, w, gam, alp, n = theta2.T

mask2 = (
    (w > -2.5) & (w < -0.33) &
    (n >= 0.0) & (n <= 0.8) &
    (gam > -0.3) & (gam < 0.3)
)

theta2 = theta2[mask2]
zeta_phys2 = physical_viscosity_today(
    theta2,
    h_index=2,
    Om_index=0,
    alp_index=5,
    n_index=6,
)

# ================================================================
# PARAMS
# ================================================================
names = ["Om", "Obh2", "h", "w", "gam", "alp", "n"]

labels = [
    r"\Omega_m",
    r"\Omega_b h^2",
    r"h",
    r"w",
    r"\gamma",
    r"\alpha",
    r"n",
    #r"\xi",
]

# ================================================================
# MCSAMPLES
# ================================================================
samples_u3 = MCSamples(
    samples=theta1,
    names=names,
    labels=labels,
    label=r"\mathrm{Union3+DR2+CMB}")

samples_p = MCSamples(
    samples=theta2,
    names=names,
    labels=labels,
    label=r"\mathrm{PantheonPlus\&SH0ES+DR2+CMB}")


samples_u3.addDerived(
    zeta_phys1,
    name="zeta_phys",
    label=r"\zeta_d(0)\,[\mathrm{Pa\,s}]",)

samples_p.addDerived(
    zeta_phys2,
    name="zeta_phys",
    label=r"\zeta_d(0)\,[\mathrm{Pa\,s}]",)

print("Physical bulk viscosity at z = 0:")
print("Union3:",samples_u3.getInlineLatex("zeta_phys", limit=1),)
print("PantheonPlus+SH0ES:",samples_p.getInlineLatex("zeta_phys", limit=1),)

for s in [samples_u3, samples_p]:

    s.updateSettings({"smooth_scale_1D": 0.6,"smooth_scale_2D": 0.6,"boundary_correction_order": 1,})

    s.setRanges({
        "w": (-2.5, -0.33),
        "n": (0.0, 0.8),
        "gam": (-0.3, 0.3),
    })

# ================================================================
# BOLD STYLE
# ================================================================
def make_bold(g):

    for ax in g.subplots.flatten():

        if ax is not None:

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

            ax.xaxis.label.set_fontweight('bold')
            ax.yaxis.label.set_fontweight('bold')

            if ax.get_legend():
                for text in ax.get_legend().get_texts():
                    text.set_fontweight('bold')

# ================================================================
# TRIANGLE PLOT
# ================================================================
params_to_plot = ["Om", "h", "w", "gam", "alp", "n"]

g = plots.get_subplot_plotter()

g.settings.lab_fontsize = 20
g.settings.axes_fontsize = 16
g.settings.legend_fontsize = 16
g.settings.linewidth = 2.0
g.settings.alpha_filled_add = 0.85

g.triangle_plot(
    [samples_u3, samples_p],
    params_to_plot,
    filled=True,
    contour_colors=["darkgreen", "darkorange"],
    legend_labels=[
        r"\textbf{Union3+DR2+CMB}",
        r"\textbf{PantheonPlus\&SH0ES+DR2+CMB}"
    ],
)

# ================================================================
# OPTIONAL LCDM LINES
# ================================================================
for i, p in enumerate(params_to_plot):

    for j, q in enumerate(params_to_plot):

        if i > j:

            ax = g.subplots[i][j]

            if p == "w":
                ax.axhline(-1.0,
                           color="black",
                           ls="--",
                           lw=1.2)

            if q == "w":
                ax.axvline(-1.0,
                           color="black",
                           ls="--",
                           lw=1.2)

# ================================================================
# FINAL TOUCH
# ================================================================
make_bold(g)

g.export("visc_combined_triangle.pdf")

print("====================================================")
print("Saved:")
print("visc_combined_triangle.pdf")
print("====================================================")
