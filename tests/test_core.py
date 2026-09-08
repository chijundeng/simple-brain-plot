import numpy as np
import pytest

import simple_brain_plot as sbp


@pytest.mark.parametrize("atlas", list(sbp.list_atlases()))
def test_plot_brain_all_atlases(tmp_path, atlas):
    regions = sbp.list_regions(atlas)
    values = np.linspace(-1, 1, len(regions))
    result = sbp.plot_brain(
        regions, values, atlas=atlas, save_path=str(tmp_path / "out")
    )
    assert result.svg_path.exists()
    assert "<svg" in result.svg
    # no leftover placeholders
    for placeholder in ("colorbarpath", "minvalue", "maxvalue"):
        assert placeholder not in result.svg


def test_rgb_matrix_colormap(tmp_path):
    regions = sbp.list_regions("aparc")
    values = np.zeros(len(regions))
    cm = np.array([[1, 0, 0], [0, 0, 1]])  # red -> blue
    result = sbp.plot_brain(
        regions, values, cmap=cm, atlas="aparc", save_path=str(tmp_path / "out")
    )
    assert result.svg_path.exists()


def test_unknown_region_raises():
    with pytest.raises(ValueError):
        sbp.plot_brain(["not-a-real-region"], [1.0], atlas="aparc")


def test_mismatched_lengths_raise():
    regions = sbp.list_regions("aparc")
    with pytest.raises(ValueError):
        sbp.plot_brain(regions, [1.0, 2.0], atlas="aparc")


def test_plot_brain_figure_returns_matplotlib_figure(tmp_path):
    import matplotlib.figure

    pytest.importorskip("cairosvg")

    regions = sbp.list_regions("aparc")
    values = np.random.randn(len(regions))
    fig = sbp.plot_brain_figure(
        regions,
        values,
        atlas="aparc",
        colorbar_label="z-score",
        colorbar_fontsize=6,
        colorbar_fontfamily="Helvetica",
    )
    assert isinstance(fig, matplotlib.figure.Figure)

    out = tmp_path / "fig.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    assert out.exists() and out.stat().st_size > 0

    # tick labels should have the requested font size / family applied
    cb_ax = fig.axes[-1]
    for label in cb_ax.get_yticklabels():
        assert label.get_fontsize() == 6
        assert label.get_fontfamily() == ["Helvetica"]


def test_plot_brain_figure_colorbar_size_changes(tmp_path):
    pytest.importorskip("cairosvg")

    regions = sbp.list_regions("aparc")
    values = np.random.randn(len(regions))

    fig_small = sbp.plot_brain_figure(
        regions, values, atlas="aparc", colorbar_fraction=0.01, colorbar_shrink=0.3
    )
    fig_big = sbp.plot_brain_figure(
        regions, values, atlas="aparc", colorbar_fraction=0.06, colorbar_shrink=0.9
    )

    small_cb_bbox = fig_small.axes[-1].get_position()
    big_cb_bbox = fig_big.axes[-1].get_position()

    small_area = small_cb_bbox.width * small_cb_bbox.height
    big_area = big_cb_bbox.width * big_cb_bbox.height
    assert big_area > small_area


def test_specific_region_gets_expected_color(tmp_path):
    # A single very high value with a two-color extreme colormap should
    # paint that region's fill with (approximately) the top color.
    regions = sbp.list_regions("aparc")
    values = np.zeros(len(regions))
    idx = regions.index("ctx-lh-insula")
    values[idx] = 1.0
    cm = np.array([[0, 0, 0], [1, 1, 1]])  # black -> white
    result = sbp.plot_brain(
        regions, values, cmap=cm, atlas="aparc", limits=(0, 1),
        save_path=str(tmp_path / "out"),
    )
    assert 'id="ctx-lh-insula_' in result.svg
    # region with value 1.0 should have a white-ish fill nearby its id
    assert "fill=\"#ffffff\"" in result.svg
