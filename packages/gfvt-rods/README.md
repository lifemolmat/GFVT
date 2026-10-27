# Rods + PEG phase-diagram widget

This standalone bundle provides the two-cell notebook
`notebooks/gfvt_rods_widget.ipynb` and its Python code under `gfvt_rods`.
It uses the spherocylinder theory implemented from Lekkerkerker & Tuinier
(2011), Chapter 6: https://doi.org/10.1007/978-94-007-1223-2_6.

## Install

Use Python 3.10 or newer; validation used Python 3.12. From this directory,
in a dedicated virtual environment:

```sh
python -m pip install ".[notebook,test]"
```

Open the notebook in VS Code, select that environment as its Python kernel,
and run its two cells in order. For browser-based notebooks, also install
JupyterLab in that environment. The original GFVT checkout, experimental
datasets, saved pickles, and book PDF are not runtime dependencies.

## Use

- Enter rod radius, cylindrical rod length excluding caps, and PEG radius
  in nm. Enter molecular weights in kDa.
- Choose Gaussian, numerical ODF, or both for each calculation. Gaussian
  alone is selected initially. The plot checkboxes are independent.
- The initial case uses PEG35K, rod radius 1 nm, cylindrical length 8 nm,
  rod molecular weight 16.5 kDa, reservoir limit 20, all coexistence types,
  and interacting PEG.
- The shared reservoir and orientation resolutions default to 61/128/256.
- Calculations report progress. Numerical failures are reported explicitly;
  successful results remain available.
- Compare reservoir/system fractions and system mM/g/L concentrations.
  Zoom/pan using the plot toolbar; Save PNG + EPS exports the current view
  with parameter legends. Figures default to `figures/rods` relative to the
  kernel's working directory.

## Verify independence

From outside this source directory, after a regular (non-editable) install:

```sh
python -c "import gfvt_rods; print(gfvt_rods.__file__)"
python -m pytest /path/to/this/bundle/tests
```

The import should resolve inside the new environment's `site-packages`.
The tests must pass without installing or adding the original `gfvt`
package to Python's search path.

## Source and scope

The first standalone release copies the full, small Python source package
under a distinct name to retain its internal relative imports. Some helper
modules support models beyond the widget's rods calculation. They are kept
to avoid incomplete dependencies rather than changing thermodynamic code
during packaging. The numerical equations are unchanged.

The notebook and test imports are adapted to `gfvt_rods`; this project does
not install a package called `gfvt`. `SOURCE_MANIFEST.json` records hashes
of the original inputs and packaged files. Existing GFVT notebooks and code
are not modified by the packaging script.
