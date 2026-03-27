"""Minimal quiver-legend helper preserved from the original project."""

from __future__ import annotations

from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


LocationOption = Literal["lower right", "lower left", "upper right", "upper left"]


def simple_quiver_legend(
    ax: plt.Axes,
    quiver: matplotlib.quiver.Quiver,
    *,
    reference_value: float = 10.0,
    unit: str = "m/s",
    legend_location: LocationOption = "lower right",
    box_width: float = 0.11,
    box_height: float = 0.15,
    text_offset: float = 0.02,
    font_size: int = 7,
    label_separation: float = 0.1,
    box_facecolor: str = "white",
    box_edgecolor: str = "k",
    box_linewidth: float = 0.8,
    zorder: float = 10.0,
) -> None:
    """Add a compact quiver legend in one corner of the axes."""
    positions = {
        "lower right": (1 - box_width / 2, box_height / 2),
        "lower left": (box_width / 2, box_height / 2),
        "upper right": (1 - box_width / 2, 1 - box_height / 2),
        "upper left": (box_width / 2, 1 - box_height / 2),
    }
    try:
        center_x, center_y = positions[legend_location]
    except KeyError as exc:
        raise ValueError(f"Invalid legend location: {legend_location}") from exc

    rect = Rectangle(
        (center_x - box_width / 2, center_y - box_height / 2),
        box_width,
        box_height,
        transform=ax.transAxes,
        facecolor=box_facecolor,
        edgecolor=box_edgecolor,
        linewidth=box_linewidth,
    )
    ax.add_patch(rect)
    label_text = f"{reference_value} {unit}" if unit else str(reference_value)

    ax.quiverkey(
        quiver,
        X=center_x,
        Y=center_y + text_offset,
        U=reference_value,
        label=label_text,
        labelpos="S",
        coordinates="axes",
        fontproperties={"size": font_size, "weight": "normal"},
        labelsep=label_separation,
        zorder=zorder,
    )
