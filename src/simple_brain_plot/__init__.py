"""simple_brain_plot: simple line-art brain plots, in pure Python.

A Python port of the MATLAB `Simple-Brain-Plot
<https://github.com/dutchconnectomelab/Simple-Brain-Plot>`_ package by
Scholtens, de Lange & van den Heuvel (2021).

Example
-------
>>> import numpy as np
>>> import simple_brain_plot as sbp
>>> regions = sbp.list_regions("aparc")
>>> values = np.random.randn(len(regions))
>>> result = sbp.plot_brain(regions, values, atlas="aparc", save_path="figures/plot")
>>> result.svg_path
PosixPath('figures/plot_aparc.svg')
"""

from .core import (
    PlotBrainResult,
    list_atlases,
    list_regions,
    plot_brain,
    plot_brain_figure,
    three_point_cmap,
    common_cmap
)

__all__ = [
    "plot_brain",
    "plot_brain_figure",
    "list_atlases",
    "list_regions",
    "PlotBrainResult",
    "three_point_cmap",
    "common_cmap"
    ""
]
__version__ = "1.1.0"
