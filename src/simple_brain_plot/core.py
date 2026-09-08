"""Core implementation of :func:`plot_brain` and :func:`plot_brain_figure`.

This module ports the logic of the original MATLAB ``plotBrain.m`` function
(from the `Simple-Brain-Plot
<https://github.com/dutchconnectomelab/Simple-Brain-Plot>`_ repository) to
pure Python. It colors the regions of a line-art brain atlas SVG according
to a vector of values and a colormap, and can return either:

* a self-contained SVG file (:func:`plot_brain`), or
* a Matplotlib ``Figure`` (:func:`plot_brain_figure`) that can be tweaked
  further and saved with ``fig.savefig(...)``.
"""

from __future__ import annotations

import base64
import io
import json
import re
import tempfile
import webbrowser
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from matplotlib.colors import LinearSegmentedColormap


__all__ = [
    "plot_brain",
    "plot_brain_figure",
    "list_atlases",
    "list_regions",
    "PlotBrainResult",
    "three_point_cmap",
    "common_cmap"
]

# ----------------------------------------------------------------------
# Atlas metadata
# ----------------------------------------------------------------------

_ATLAS_PACKAGE = "simple_brain_plot.atlases"
_DATA_PACKAGE = "simple_brain_plot.data"

# Atlases whose region names need a trailing underscore appended before
# substring-matching against SVG element ids. This mirrors the original
# MATLAB code, which does this to disambiguate region names that are
# prefixes of other region names (e.g. "FE" vs. "FEE" in wbb47).
_NEEDS_TRAILING_UNDERSCORE = {"aparc", "aparc_aseg", "wbb47"}

_ATLAS_DESCRIPTIONS = {
    "aparc": "Desikan-Killiany cortical atlas (68 regions)",
    "aparc_aseg": "Desikan-Killiany cortical atlas + subcortical ASEG "
    "segmentation (82 regions)",
    "lausanne120": "120-region Cammoun sub-parcellation of the "
    "Desikan-Killiany atlas (114 regions)",
    "lausanne120_aseg": "120-region Cammoun sub-parcellation + subcortical "
    "ASEG segmentation (128 regions)",
    "lausanne250": "250-region Cammoun sub-parcellation of the "
    "Desikan-Killiany atlas (219 regions)",
    "wbb47": "39-region combined Walker-von Bonin and Bailey parcellation "
    "atlas of the macaque",
}


@dataclass
class PlotBrainResult:
    """Result of a :func:`plot_brain` call.

    Attributes
    ----------
    svg_path:
        Path to the generated, self-contained SVG file (colorbar is
        embedded as a base64 data URI, so the file has no external
        dependencies).
    svg:
        The SVG contents as a string.
    """

    svg_path: Path
    svg: str

    def _repr_svg_(self) -> str:  # pragma: no cover - notebook convenience
        # Lets Jupyter/IPython render the result inline automatically.
        return self.svg


def list_atlases() -> dict:
    """Return a mapping of available atlas names to short descriptions."""

    return dict(_ATLAS_DESCRIPTIONS)


def list_regions(atlas: str) -> list:
    """Return the list of region names expected for ``atlas``."""

    return list(_load_region_descriptions()[_normalize_atlas(atlas)])


def _normalize_atlas(atlas: str) -> str:
    atlas = atlas.lower()
    if atlas not in _ATLAS_DESCRIPTIONS:
        raise ValueError(
            f"Unknown atlas {atlas!r}. Available atlases: "
            f"{', '.join(sorted(_ATLAS_DESCRIPTIONS))}"
        )
    return atlas


def _load_region_descriptions() -> dict:
    with resources.files(_DATA_PACKAGE).joinpath(
        "region_descriptions.json"
    ).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_atlas_svg(atlas: str) -> str:
    path = resources.files(_ATLAS_PACKAGE).joinpath(f"{atlas}_template.svg")
    return path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Colors
# ----------------------------------------------------------------------


def _resolve_colormap(cmap):
    """Turn ``cmap`` into ``(cmap_rgb, mpl_colormap)``.

    ``cmap_rgb`` is an (N, 3) array of RGB values in [0, 1], used for the
    discretized nearest-color region lookup (matching the original MATLAB
    behavior). ``mpl_colormap`` is a genuine Matplotlib ``Colormap``
    instance (continuous, when possible) suitable for drawing a smooth
    colorbar gradient.

    ``cmap`` may be:

    * the name of a Matplotlib colormap (``str``), e.g. ``"viridis"``.
    * a Matplotlib ``Colormap`` instance.
    * an (N, 3) or (N, 4) array-like of RGB(A) values in [0, 1], matching
      the ``cm`` argument of the original MATLAB function.
    """

    if isinstance(cmap, str):
        import matplotlib as mpl

        try:
            mpl_cmap = mpl.colormaps[cmap]
        except AttributeError:  # pragma: no cover - older Matplotlib
            import matplotlib.cm as mcm

            mpl_cmap = mcm.get_cmap(cmap)
        colors = mpl_cmap(np.linspace(0, 1, 256))[:, :3]
        return colors, mpl_cmap

    # Matplotlib Colormap instances expose a __call__ + N attribute.
    if hasattr(cmap, "__call__") and hasattr(cmap, "N"):
        colors = np.asarray(cmap(np.linspace(0, 1, max(cmap.N, 256)))[:, :3])
        return colors, cmap

    colors = np.asarray(cmap, dtype=float)
    if colors.ndim != 2 or colors.shape[1] not in (3, 4):
        raise ValueError(
            "cmap must be a colormap name, a Matplotlib Colormap, or an "
            "(N, 3)/(N, 4) array of RGB(A) values."
        )
    colors = colors[:, :3]

    from matplotlib.colors import LinearSegmentedColormap

    mpl_cmap = LinearSegmentedColormap.from_list("simple_brain_plot_cmap", colors)
    return colors, mpl_cmap


def _values_to_colors(
    values: np.ndarray, cmap_rgb: np.ndarray, vmin: float, vmax: float
) -> list:
    """Map ``values`` to hex color strings using ``cmap_rgb``.

    Mirrors the discretized nearest-color lookup used by the original
    MATLAB implementation.
    """

    values = np.clip(values, vmin, vmax)
    n_colors = cmap_rgb.shape[0]

    if vmax > vmin:
        values_norm = (values - vmin) / (vmax - vmin)
        idx = np.round(values_norm * (n_colors - 1)).astype(int)
    else:
        # All values identical (or vmin == vmax): use the last color,
        # matching the MATLAB fallback behavior.
        idx = np.full(values.shape, n_colors - 1, dtype=int)

    idx = np.clip(idx, 0, n_colors - 1)
    rgb = cmap_rgb[idx]

    return [_rgb_to_hex(r, g, b) for r, g, b in rgb]


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(round(255 * float(r))),
        int(round(255 * float(g))),
        int(round(255 * float(b))),
    )


def _colorbar_png_base64(cmap_rgb: np.ndarray, height: int = 200) -> str:
    """Render ``cmap_rgb`` as a vertical colorbar PNG, base64-encoded."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.flipud(cmap_rgb.reshape(-1, 1, 3))
    image = np.repeat(image, height, axis=1)

    buf = io.BytesIO()
    fig = plt.figure(figsize=(0.4, 4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, aspect="auto")
    ax.axis("off")
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode("ascii")


# ----------------------------------------------------------------------
# SVG manipulation
# ----------------------------------------------------------------------

# Applied on top of the user-supplied `scaling` to shrink the previously
# very large default output size down to 1/4 of what it used to be.
SIZE_FACTOR = 0.25

_TAG_RE = re.compile(r"<[a-zA-Z]+\b[^<>]*?/>")
_ID_RE = re.compile(r'\bid="([^"]*)"')
_FILL_RE = re.compile(r'fill="[^"]*"')


def _color_svg(svg: str, region_colors: dict) -> str:
    """Return ``svg`` with the ``fill`` of matching elements recolored.

    ``region_colors`` maps region *keys* (already suffixed if needed) to
    hex color strings. An SVG element is recolored if its ``id`` attribute
    contains one of the region keys as a substring; when several keys
    match, the longest (most specific) one wins.
    """

    # Sort once, longest first, so substring matching prefers specificity.
    keys_sorted = sorted(region_colors, key=len, reverse=True)

    def _recolor_tag(match: re.Match) -> str:
        tag = match.group(0)
        id_match = _ID_RE.search(tag)
        if not id_match:
            return tag
        element_id = id_match.group(1)

        hit = next((k for k in keys_sorted if k in element_id), None)
        if hit is None:
            return tag

        color = region_colors[hit]
        if _FILL_RE.search(tag):
            return _FILL_RE.sub(f'fill="{color}"', tag, count=1)
        # Element had no fill attribute; add one just before the closing "/>".
        return tag[:-2] + f' fill="{color}"/>'

    return _TAG_RE.sub(_recolor_tag, svg)


_COLORBAR_IMAGE_RE = re.compile(
    r'<image\b[^>]*?xlink:href="colorbarpath"[^>]*?(?:/>|>.*?</image>)',
    re.DOTALL,
)
_COLORBAR_TEXT_RE = re.compile(
    r'<text\b[^>]*?>(?:(?!</text>).)*?(?:minvalue|maxvalue)(?:(?!</text>).)*?</text>',
    re.DOTALL,
)


def _strip_builtin_colorbar(svg: str) -> str:
    """Remove the template's baked-in colorbar image/labels entirely.

    Used when the caller wants to draw their own (Matplotlib) colorbar
    instead, e.g. in :func:`plot_brain_figure`.
    """

    svg = _COLORBAR_IMAGE_RE.sub("", svg)
    svg = _COLORBAR_TEXT_RE.sub("", svg)
    return svg


def _validate_and_normalize(
    regions: Sequence[str],
    values: Iterable[float],
    atlas: str,
    limits: tuple | None,
):
    """Shared validation/normalization for the public plotting functions."""

    atlas = _normalize_atlas(atlas)

    regions = list(regions)
    values = np.asarray(list(values), dtype=float)
    if len(regions) != len(values):
        raise ValueError(
            "regions and values must have the same length "
            f"(got {len(regions)} and {len(values)})."
        )
    if not regions:
        raise ValueError("regions must not be empty.")

    valid_regions = set(_load_region_descriptions()[atlas])
    unknown = [r for r in regions if r not in valid_regions]
    if unknown:
        raise ValueError(
            f"The following regions are not part of the {atlas!r} atlas: "
            f"{unknown[:10]}{' ...' if len(unknown) > 10 else ''}"
        )

    if limits is not None:
        vmin, vmax = limits
    else:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))

    return atlas, regions, values, vmin, vmax


def _build_colored_svg(regions, values, cmap, atlas, vmin, vmax):
    """Return (svg, cmap_rgb, mpl_cmap) with regions colored, no scaling
    or colorbar substitutions applied yet."""

    cmap_rgb, mpl_cmap = _resolve_colormap(cmap)
    hex_colors = _values_to_colors(values, cmap_rgb, vmin, vmax)

    suffix = "_" if atlas in _NEEDS_TRAILING_UNDERSCORE else ""
    region_colors = {
        region + suffix: color for region, color in zip(regions, hex_colors)
    }

    svg = _load_atlas_svg(atlas)
    svg = _color_svg(svg, region_colors)
    return svg, cmap_rgb, mpl_cmap


def _finalize_svg(
    svg: str,
    colorbar_base64: str,
    vmin: float,
    vmax: float,
    scaling: float,
) -> str:
    """Apply the placeholder substitutions used by the atlas templates."""

    svg = svg.replace(
        "colorbarpath", f"data:image/png;base64,{colorbar_base64}"
    )
    svg = svg.replace("minvalue", f"{vmin:.4g}")
    svg = svg.replace("maxvalue", f"{vmax:.4g}")
    # The native template dimensions (1756mm x 4224.5mm) are enormous;
    # SIZE_FACTOR shrinks the previously-used output size by another 1/4
    # on top of the user-supplied `scaling`, so the default scaling=0.1
    # now produces a quarter of the size it used to.
    svg = re.sub(
        r'height="1756mm"',
        f'height="{scaling * SIZE_FACTOR * 1756:f}mm"',
        svg,
    )
    svg = re.sub(
        r'width="4224\.5mm"',
        f'width="{scaling * SIZE_FACTOR * 4224.5:f}mm"',
        svg,
    )
    return svg


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def plot_brain(
    regions: Sequence[str],
    values: Iterable[float],
    cmap="RdBu_r",
    *,
    atlas: str = "aparc",
    limits: tuple | None = None,
    scaling: float = 0.1,
    save_path: str | Path | None = None,
    viewer: bool = False,
) -> PlotBrainResult:
    """Create a simple line-art brain plot.

    Parameters
    ----------
    regions:
        Sequence of region names. Must match (a subset of) the region
        names expected by ``atlas`` -- see :func:`list_regions`.
    values:
        Numeric values, one per region in ``regions``.
    cmap:
        A Matplotlib colormap name (default ``"RdBu_r"``), a Matplotlib
        ``Colormap`` instance, or an (N, 3) array of RGB values in
        ``[0, 1]`` (as in the original MATLAB ``cm`` argument).
    atlas:
        Which brain atlas/parcellation to use. One of ``"aparc"``,
        ``"aparc_aseg"``, ``"lausanne120"``, ``"lausanne120_aseg"``,
        ``"lausanne250"``, ``"wbb47"``. See :func:`list_atlases`.
    limits:
        Optional ``(vmin, vmax)`` tuple. Values are assigned to the first
        and last color in the colormap respectively. Defaults to
        ``(min(values), max(values))``.
    scaling:
        Scale factor (0, 1] applied to the (very large) native SVG
        dimensions. Defaults to ``0.1``.
    save_path:
        Where to write the output SVG. Defaults to a file in a temporary
        directory.
    viewer:
        If ``True``, open the resulting SVG in the default web browser.

    Returns
    -------
    PlotBrainResult
        Dataclass with the output path (``svg_path``) and SVG contents
        (``svg``).
    """

    if not (0 < scaling <= 1):
        raise ValueError("scaling must be > 0 and <= 1.")

    atlas, regions, values, vmin, vmax = _validate_and_normalize(
        regions, values, atlas, limits
    )
    svg, cmap_rgb, _mpl_cmap = _build_colored_svg(
        regions, values, cmap, atlas, vmin, vmax
    )

    colorbar_b64 = _colorbar_png_base64(cmap_rgb)
    svg = _finalize_svg(svg, colorbar_b64, vmin, vmax, scaling)

    if save_path is not None:
        out_path = Path(save_path)
        if out_path.suffix.lower() != ".svg":
            out_path = out_path.with_name(out_path.name + f"_{atlas}.svg")
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="simple_brain_plot_"))
        out_path = tmp_dir / f"plot_{atlas}.svg"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")

    if viewer:
        webbrowser.open(out_path.resolve().as_uri(), new=2)

    return PlotBrainResult(svg_path=out_path, svg=svg)


def _autocrop_transparent(image: np.ndarray, pad_frac: float = 0.02) -> np.ndarray:
    """Crop ``image`` (H, W, 4) to the bounding box of its non-transparent
    pixels, with a small padding margin.

    The atlas SVG templates reserve a lot of extra canvas space to the
    right for the built-in colorbar/labels; once those are stripped for
    :func:`plot_brain_figure`, that space is fully transparent and would
    otherwise show up as dead whitespace next to the brain artwork.
    """

    if image.ndim != 3 or image.shape[2] < 4:
        return image

    alpha = image[:, :, 3]
    rows = np.where(alpha.max(axis=1) > 0)[0]
    cols = np.where(alpha.max(axis=0) > 0)[0]
    if rows.size == 0 or cols.size == 0:
        return image

    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]

    pad_y = int(round((bottom - top) * pad_frac))
    pad_x = int(round((right - left) * pad_frac))

    top = max(0, top - pad_y)
    bottom = min(image.shape[0] - 1, bottom + pad_y)
    left = max(0, left - pad_x)
    right = min(image.shape[1] - 1, right + pad_x)

    return image[top : bottom + 1, left : right + 1]


def plot_brain_figure(
    regions: Sequence[str],
    values: Iterable[float],
    cmap="RdBu_r",
    *,
    atlas: str = "aparc",
    limits: tuple | None = None,
    dpi: int = 300,
    fig_width: float = 6.0,
    raster_width: int = 2000,
    colorbar_label: str | None = None,
    colorbar_fraction: float = 0.025,
    colorbar_shrink: float = 0.6,
    colorbar_pad: float = 0.02,
    colorbar_aspect: float = 15,
    colorbar_fontsize: float = 6,
    colorbar_fontfamily: str = "Helvetica",
    save_path: str | None = None
):
    """Create a brain plot as a Matplotlib ``Figure``.

    Unlike :func:`plot_brain` (which writes a self-contained SVG), this
    renders the colored brain atlas as a raster image inside a Matplotlib
    ``Axes`` and draws a real Matplotlib colorbar next to it, so the
    result can be tweaked further and saved with ``fig.savefig(...)``.

    Requires the optional ``cairosvg`` dependency

    Parameters
    ----------
    regions, values, cmap, atlas, limits:
        Same as :func:`plot_brain`.
    dpi:
        Resolution of the returned figure.
    fig_width:
        Width of the figure in inches; height follows the atlas's aspect
        ratio automatically.
    raster_width:
        Pixel width used when rasterizing the underlying SVG artwork
        before it's placed in the figure. Increase for crisper output at
        large ``fig_width``/``dpi``.
    colorbar_label:
        Optional label drawn alongside the colorbar.
    colorbar_fraction, colorbar_shrink, colorbar_pad, colorbar_aspect:
        Passed straight through to ``Figure.colorbar`` to control the
        colorbar's size and placement. ``colorbar_fraction`` is the
        fraction of the image axes' width used for the colorbar (make
        this smaller/larger to shrink/grow it), ``colorbar_shrink``
        scales its length, and ``colorbar_aspect`` controls how
        long/thin it is.
    colorbar_fontsize:
        Font size (points) of the colorbar tick labels and label.
        Defaults to ``6``.
    colorbar_fontfamily:
        Font family of the colorbar tick labels and label. Defaults to
        ``"Helvetica"``.
     save_path:
        Save path for figure

    Returns
    -------
    matplotlib.figure.Figure
    """

    try:
        import cairosvg
    except ImportError as exc:  # pragma: no cover - exercised via error path
        raise ImportError(
            "plot_brain_figure() requires the optional 'cairosvg' "
            "dependency. Install it with: conda install cairosvg"
        ) from exc

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import matplotlib
    
    matplotlib.rcParams['svg.fonttype'] = 'none'

    atlas, regions, values, vmin, vmax = _validate_and_normalize(
        regions, values, atlas, limits
    )
    svg, _cmap_rgb, mpl_cmap = _build_colored_svg(
        regions, values, cmap, atlas, vmin, vmax
    )
    svg = _strip_builtin_colorbar(svg)

    png_bytes = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"), output_width=raster_width,
    )
    image = plt.imread(io.BytesIO(png_bytes))
    image = _autocrop_transparent(image)

    height_px, width_px = image.shape[:2]
    fig_height = fig_width * (height_px / width_px)

    fig, ax_img = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax_img.imshow(image)
    ax_img.axis("off")

    mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=mpl_cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(
        mappable,
        ax=ax_img,
        fraction=colorbar_fraction,
        shrink=colorbar_shrink,
        pad=colorbar_pad,
        aspect=colorbar_aspect,
    )
    colorbar.ax.tick_params(labelsize=colorbar_fontsize)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontsize(colorbar_fontsize)
        label.set_fontfamily(colorbar_fontfamily)
    if colorbar_label is not None:
        colorbar.set_label(
            colorbar_label,
            fontsize=colorbar_fontsize,
            family=colorbar_fontfamily,
        )

    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure hase saved to: {save_path}")
    
    return fig

# Available three-point colour palettes
CCMAP_COLORS = {
    "c1":  ("#84accf", "white", "#d85e51"),
    "c2":  ("#81adcf", "white", "#ab90bd"),
    "c3":  ("#84a7da", "white", "#a97ca1"),
    "c4":  ("#84a7da", "white", "#efa29e"),
    "c5":  ("#5dbed3", "white", "#e54d36"),
    "c6":  ("#663d74", "white", "#fac03d"),
    "c7":  ("#5a83bc", "white", "#b65454"),
    "c8":  ("#4c8364", "white", "#aa5cb9"),
    "c9":  ("#478497", "white", "#c3593d"),
    "c10": ("#4dbbd5", "white", "#e64b35"),
    "c11": ("#57147d", "#fa815e", "#fce6a8"),
    "c12": ("#c0e4d6", "#6a90b1", "#4e4678"),
}


def three_point_cmap(cool="#79C", mid="white", warm="#b85d4a", name="ccmap"):
    """Create a three-point continuous colormap."""
    return LinearSegmentedColormap.from_list(
        name,
        [cool, mid, warm],
    )


def common_cmap(name="c1"):
    """Return a predefined colormap."""
    if name not in CCMAP_COLORS:
        available = ", ".join(CCMAP_COLORS)
        raise ValueError(
            f"Unknown colormap '{name}'. Available colormaps: {available}"
        )

    cool, mid, warm = CCMAP_COLORS[name]
    return three_point_cmap(cool, mid, warm, name=name)









