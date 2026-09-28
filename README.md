# Viscous Dark Energy: Chains, Analysis Code, and Figures

This repository collects posterior samples, nested-sampling outputs, Python analysis scripts, and PDF figures for minimal and non-minimal viscous dark-energy models, with ΛCDM reference outputs.

The supplied runs use the dataset combinations identified in the filenames as `Union3+DR2+PLK18` and `PantheonPlusSH0ES+DR2+PLK18`. The parameter metadata identifies the CMB likelihood as `SPlanck_18`; the distance-plot scripts contain DESI comparison points. Full likelihood definitions and sampling configurations are not included here.

The archive supports inspection of existing results and further posterior analysis. Regenerating the density and distance figures additionally requires the matching custom SimpleMC implementation described below. It is not a complete package for rerunning the original inference.

## Repository contents

```text
.
├── README.md
├── CODES /                       # The current directory name has a trailing space
│   ├── Getdist_combined_minimal.py
│   ├── Getdist_combined_nonminimal.py
│   ├── all_plots_minimal.py
│   ├── all_plots_non-minimal.py
│   ├── rho_fgiven_minimal.py
│   └── rho_fgiven_nonminimal.py
├── DATA/
│   ├── visc25/                   # Outputs used by the supplied plotting scripts
│   ├── revision_prior/           # Separate run collection
│   └── revision_gamma/           # Separate collection with different parameter columns
└── PLOTS/
    ├── MINIMAL/
    └── NON-MINIMAL/
```

The commands below quote `CODES ` to preserve its trailing space. If the directory is renamed to `CODES`, update those commands accordingly.

## Models and run collections

| Filename prefix | Interpretation in the supplied scripts |
| --- | --- |
| `LCDM_phy_` | ΛCDM reference runs; available in `DATA/visc25/` |
| `Visc_sfixed_phy_` | Minimal viscous model; the scripts set `vary_s=False` |
| `Visc_phy_` | Non-minimal viscous model; the scripts set `vary_s=True` |

The non-minimal triangle script labels the parameter stored as `s` with the symbol ξ. The fixed value used for `s` in the minimal model depends on the external model implementation and is not specified by these plotting scripts.

- `visc25` contains six run roots: three models for each of the two dataset combinations. It also contains four additional PDF plots and a `_test.txt` file.
- `revision_prior` contains four viscous-model run roots. This checks the prior dependency for the viscous runs.
- `revision_gamma` contains four viscous-model run roots. This check the case where gamma=0.

These directories represent separate analyses. Do not concatenate them or substitute them in the scripts without checking parameter definitions, priors, and model settings.

## Data formats

Each run uses a shared filename root, for example:

```text
Visc_phy_Union3+DR2+PLK18_nested_multi
```

| Suffix | Contents and use |
| --- | --- |
| `_posterior.txt` | Posterior draws used by the supplied plotting scripts, which treat rows with equal weight. Each supplied posterior file contains 20,000 rows. Columns follow the corresponding `.paramnames` file, including likelihood and prior diagnostic columns. |
| `_1.txt` | Chain output with two leading columns in the conventional GetDist layout (weight and negative log likelihood), followed by the columns named in `.paramnames`. It is not interchangeable with `_posterior.txt`. |
| `.paramnames` | Column names and display labels. Use the file belonging to the exact run being loaded. |
| `_summary.txt` | Sampling diagnostics, parameter summaries, log evidence with its uncertainty, and AIC, as recorded by the original analysis. |
| `.covmat` | Comma-separated covariance matrix for the sampled parameters. |
| `.maxlike` | Maximum-likelihood output recorded by the original analysis. |
| `_temp.txt` | Auxiliary sampling output; not used by the supplied plotting scripts. |
| `_test.txt` | Additional file in `visc25`; not used by the supplied plotting scripts. |

For viscous runs in `visc25` and `revision_prior`, the sampled-parameter columns in `_posterior.txt` begin as follows:

| Model / dataset | Leading parameter order |
| --- | --- |
| Minimal / Union3 | `Om, Obh2, h, w, gam, alp, n` |
| Non-minimal / Union3 | `Om, Obh2, h, w, gam, alp, n, s` |
| Minimal / PantheonPlusSH0ES | `Om, Obh2, h, M, w, gam, alp, n` |
| Non-minimal / PantheonPlusSH0ES | `Om, Obh2, h, M, w, gam, alp, n, s` |

The remaining columns are likelihood components and `theory_prior`. In `revision_gamma`, remove `gam` from these sequences. The extra `M` column in PantheonPlusSH0ES runs also shifts the indices of subsequent parameters.

## Software requirements

The scripts use Python 3 and the following libraries:

- NumPy and Matplotlib;
- SciPy and GetDist for the combined posterior plots;
- fgivenx for functional posterior contours;
- a custom SimpleMC installation for the density and distance calculations.

A starting environment for posterior plotting can be created with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib getdist fgivenx
```

This installs the general plotting dependencies, not the custom SimpleMC model. Package versions for the original analysis are not recorded in this archive, so this is not a pinned reproduction environment.

All six supplied scripts enable `text.usetex=True`. To use them unchanged, install a working LaTeX distribution and the rendering tools required by Matplotlib. Disabling this setting may also require simplifying TeX-specific labels.

The four density/distance scripts import `simplemc.models.ViscCosmology` and parameters including `gam_par`, `alp_par`, `n_par`, and `s_par`. They require model methods such as `rhoD_z`, `rhoPlusP_z`, and the distance functions used in the scripts. Obtain the matching implementation and make `simplemc` importable before running them. Neither this implementation nor the original sampler/likelihood configuration is included.

## Quick start: inspect a posterior

The following example runs from the repository root and makes a basic triangle plot directly from an included posterior file. It does not require SimpleMC or the GetDist GUI and does not reproduce all the cuts and styling of the supplied publication scripts.

```python
from pathlib import Path
import numpy as np
from getdist import MCSamples, plots

root = Path("DATA/visc25/Visc_phy_Union3+DR2+PLK18_nested_multi")
raw = np.atleast_2d(np.loadtxt(str(root) + "_posterior.txt"))
raw = raw[np.isfinite(raw).all(axis=1)]

# Select sampled parameters only; omit trailing likelihood/prior columns.
names = ["Om", "Obh2", "h", "w", "gam", "alp", "n", "s"]
labels = [r"\Omega_m", r"\Omega_b h^2", "h", "w",
          r"\gamma", r"\alpha", "n", r"\xi"]
samples = MCSamples(samples=raw[:, :8], names=names, labels=labels)

g = plots.get_subplot_plotter()
g.triangle_plot(samples, ["Om", "h", "w", "gam", "alp", "n", "s"],
                filled=True)
g.export("posterior_example.pdf")
```

When using the GetDist GUI, load the run root associated with `_1.txt` and `.paramnames`; do not load a parameter-only `_posterior.txt` as though it contained leading weight and likelihood columns.

## Using the supplied plotting scripts

### Configure paths first

The scripts retain paths from the original analysis environment and need local configuration before use:

1. In `Getdist_combined_minimal.py` and `Getdist_combined_nonminimal.py`, replace `chain_u3` and `chain_p` with paths to the matching files in `DATA/visc25/`.
2. In the four `all_plots_*` and `rho_fgiven_*` scripts, configure the SimpleMC import path independently of the chain location. Their current `simplemc_root` calculation assumes an older directory layout. Point `chain_dir` to this repository's `DATA/visc25/`.
3. In the density scripts, update `outdir` to your desired output directory.
4. The two triangle scripts both export `visc_combined_triangle.pdf` in the working directory. Give the output files distinct names or run the scripts in separate output directories to retain both results. The archived figures have distinct minimal/non-minimal filenames.

For example, within a script located in `CODES `, a repository-relative data path can be constructed as:

```python
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
chain_dir = repo_root / "DATA" / "visc25"
```

### Script guide

| Script in `CODES ` | Purpose |
| --- | --- |
| `Getdist_combined_minimal.py` | Compare Union3 and PantheonPlusSH0ES posterior constraints for the minimal model; report the derived present-day physical bulk viscosity. |
| `Getdist_combined_nonminimal.py` | Corresponding comparison for the non-minimal model, including `s` (plotted as ξ). |
| `all_plots_minimal.py` | Minimal-model distance-ratio plots with observational points and a ΛCDM reference curve. |
| `all_plots_non-minimal.py` | Corresponding non-minimal distance-ratio plots. |
| `rho_fgiven_minimal.py` | Minimal-model density and density-plus-pressure evolution with functional posterior contours. |
| `rho_fgiven_nonminimal.py` | Corresponding non-minimal density and density-plus-pressure plots. |

After configuring paths and dependencies, run the desired script, for example:

```bash
python "CODES /Getdist_combined_minimal.py"
```

The combined triangle scripts filter samples to `-2.5 < w < -0.33`, `0 <= n <= 0.8`, and `-0.3 < gam < 0.3`, in addition to removing non-finite rows. These are explicit post-processing selections, not documentation of the original sampling priors. The density and distance scripts use their own validity checks; inspect those before comparing outputs or adapting scripts to the revision chains.

The distance scripts embed observational comparison values and propagate the displayed distance-ratio errors using diagonal terms. Their ΛCDM curves use explicit reference parameter values rather than fitting the included ΛCDM chains. The density/distance scripts do not propagate every sampled parameter through the model updates (for example, they do not update `Obh2` for each sample), so their output also depends on the external model defaults.

## Figures

The [minimal-model figures](PLOTS/MINIMAL/) include:

- [Combined posterior triangle](PLOTS/MINIMAL/visc_combined_triangle_minimal.pdf)
- [Distance-ratio plot](PLOTS/MINIMAL/visc_dm_minimal.pdf)
- [Volume-distance plot](PLOTS/MINIMAL/visc_dv_minimal.pdf)
- [Dark-energy density](PLOTS/MINIMAL/visc_rhoD_minimal.pdf)
- [Density plus pressure](PLOTS/MINIMAL/visc_rhoPlusP_minimal.pdf)

The [non-minimal-model figures](PLOTS/NON-MINIMAL/) include:

- [Combined posterior triangle](PLOTS/NON-MINIMAL/visc_combined_triangle_non-minimal.pdf)
- [Distance-ratio plot](PLOTS/NON-MINIMAL/visc_dm.pdf)
- [Volume-distance plot](PLOTS/NON-MINIMAL/visc_dv.pdf)
- [Dark-energy density](PLOTS/NON-MINIMAL/visc_rhoD_fgiven.pdf)
- [Density plus pressure](PLOTS/NON-MINIMAL/visc_rhoPlusP_fgiven.pdf)
- Additional triangle PDFs named `visc_triangle_temperature_w.pdf`, `visc_triangle_temperature_alp.pdf`, and `visc_triangle_temperature_n.pdf`.

Four further color-coded PDFs are stored in `DATA/visc25/`. Separate generation scripts for these additional figures are not included. The archived PDFs have not been regenerated or matched numerically to the supplied scripts as part of preparing this documentation.

## Citation and reuse

The associated article's title, author list, DOI, and arXiv identifier are not supplied in this archive. A formal citation should be added when those details are available.

No license file is included. No additional reuse license is asserted by this README; contact the repository owner for clarification. Third-party software and observational data remain subject to their respective terms and citation requirements.
