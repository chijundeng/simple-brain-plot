import numpy as np
import simple_brain_plot as sbp

OUT_DIR = "figures"

# Colormap for brain regions. Any matplotlib colormap name is accepted.
# Common choices: 'RdBu_r', 'Spectral_r', 'viridis', 'plasma', etc.
cmap = sbp.common_cmap('c2')  # c1-12 are custom colormaps

## Option 1
for atlas in sbp.list_atlases():
    regions = sbp.list_regions(atlas)
    values = np.random.randn(len(regions))

    fig = sbp.plot_brain_figure(
        regions,
        values,
        cmap=cmap,
        atlas=atlas,
        save_path=f"{OUT_DIR}/{atlas}.png",
    )

## Option 2
for atlas in sbp.list_atlases():
    regions = sbp.list_regions(atlas)
    values = np.random.randn(len(regions))

    result = sbp.plot_brain(
        regions,
        values,
        cmap=cmap,
        atlas=atlas,
        save_path=f"{OUT_DIR}/{atlas}.svg",
    )
