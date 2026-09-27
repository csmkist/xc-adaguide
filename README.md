# XC-AdaGuiDE

Adaptive differential evolution with a post-run consistency test between global (Sobol total-order) and local (forward-difference) variable importance.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22994483.svg)](https://doi.org/10.5281/zenodo.22994483)

Code and results for the paper submitted to *Discover Computing*. Archived release: https://doi.org/10.5281/zenodo.22994483

## Contents

```
notebooks/
  01_main_study.ipynb            benchmark study E1-E6 and maglev case study (Colab-ready)
  02_extended_experiments.ipynb  experiments P1-P8: noise, difference schemes, gate,
                                 timing, block rotation, quadrotor, tuning set, on/off
src/
  xc_engine.py                   benchmarks, optimizers, Sobol, statistics, maglev, figures
  xc_extended.py                 P1-P8 modules (imports xc_engine)
results/                         result archives behind the tables in the paper
CHANGELOG.md
requirements.txt
```

Each notebook contains the full engine and runs on its own in Google Colab. The `src/` files hold the same code as Python modules.

## Reproducing the results

| Step | Notebook / preset | Colab CPU time | Output |
|---|---|---|---|
| 1 | `01_main_study.ipynb`, `PRESET = "PUBLICATION"` | about 32 min | `results/` |
| 2 | notebook 01, optional bound-revision sweep cell (10000*D budget) | about 85 min | `results_boundary_sweep/E6_param_sweep.csv` |
| 3 | `02_extended_experiments.ipynb`, `PRESET = "FULL"` | about 75 min | `extended/` |

A `SMOKE` preset in each notebook checks the pipeline in under a minute.

Local run:
```bash
pip install -r requirements.txt
cd src
python -c "import xc_engine as E; E.main(dict(E.CFG), E.SUITE)"                                    # step 1
python -c "import xc_extended as X; c = dict(X.CFG, **X.EXT_CFG); print(X.p6_quadrotor(c))"       # one module
```

## Reproducibility

- All random streams are seeded. Permutation and bootstrap seeds use `zlib.crc32`, so they do not depend on the Python hash seed.
- Two independent Colab runs of step 1 gave byte-identical result files.
- A self-test runs before every experiment. It checks that every shifted and rotated optimum lies inside the box (44 checks) and compares the Sobol estimator with the analytic Ishigami indices (maximum error 0.0013).

## License

MIT (see `LICENSE`).
