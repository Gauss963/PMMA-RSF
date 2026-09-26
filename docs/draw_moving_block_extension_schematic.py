"""Draw the proposed geometry only; this script changes no simulation inputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


OUT = Path(__file__).resolve().parent
STEM = "moving_block_extension_schematic"
# The extension length is illustrative, not a proposed simulation setting.
L_DRAW = 200.0
INK = "#213547"
BLUE = "#286a91"
TEAL = "#137b68"
OCHRE = "#b17625"
RED = "#b54536"
GRAY = "#64717d"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "mathtext.fontset": "dejavusans",
    "text.color": INK,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def dimension(ax, start, end, label, offset=(0, 0), rotation=0):
    ax.annotate("", end, start,
                arrowprops={"arrowstyle": "<->", "color": GRAY, "lw": 1.0,
                            "shrinkA": 0, "shrinkB": 0})
    middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    ax.text(middle[0] + offset[0], middle[1] + offset[1], label,
            ha="center", va="center", rotation=rotation, fontsize=10.5,
            color=GRAY, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})


def note(ax, y, number, title, lines, color=INK):
    ax.text(0, y, number, color=color, fontsize=12, weight="bold", va="top")
    ax.text(0.095, y, title, color=color, fontsize=12.8, weight="bold", va="top")
    ax.text(0.095, y - 0.052, "\n".join(lines), fontsize=10.6,
            linespacing=1.55, va="top")


def main():
    fig = plt.figure(figsize=(13.0, 10.3), facecolor="white")
    fig.text(0.045, 0.965, "Moving-block extension toward $-y$",
             fontsize=21, weight="bold", va="top")
    fig.text(0.045, 0.92,
             "Proposed series compliance for the loading system | Geometry and boundary conditions",
             fontsize=11.4, color=GRAY)
    fig.add_artist(plt.Line2D([0.045, 0.96], [0.9, 0.9],
                             transform=fig.transFigure, color="#d8dfe3", lw=1))

    ax = fig.add_axes([0.035, 0.11, 0.555, 0.76])
    ax.set(xlim=(-120, 450), ylim=(-315, 650), aspect="equal")
    ax.axis("off")

    # One continuous moving block, with a color change only to identify added material.
    ax.add_patch(Rectangle((0, 0), 200, 500, fc="#e4eff4", ec="none"))
    ax.add_patch(Rectangle((0, -L_DRAW), 200, L_DRAW,
                           fc="#faead2", ec="none"))
    ax.add_patch(Rectangle((0, -L_DRAW), 200, 500 + L_DRAW,
                           fill=False, ec=INK, lw=1.65))
    ax.add_patch(Rectangle((200, 0), 145, 550,
                           fc="#edf0f2", ec=INK, lw=1.65))

    ax.text(100, 303, "MOVING\nBLOCK", ha="center", va="center",
            fontsize=13, weight="bold", linespacing=1.45)
    ax.text(100, 245, "Original region", ha="center", color=GRAY, fontsize=10)
    ax.text(272.5, 290, "STATIONARY\nBLOCK", ha="center", va="center",
            fontsize=11.5, weight="bold", linespacing=1.45)
    ax.text(272.5, 235, "Unchanged", ha="center", color=GRAY, fontsize=10)
    ax.text(100, -62, "ADDED ELASTIC\nEXTENSION", ha="center", va="center",
            fontsize=11.3, color=OCHRE, weight="bold", linespacing=1.4)
    ax.text(100, -143, "Connected solid\nNo joint at $y=0$", ha="center",
            fontsize=10.5, color=OCHRE, linespacing=1.45)

    # Original fault and normal-traction support do not grow with L_ext.
    ax.plot([200, 200], [0, 500], color=TEAL, lw=4, solid_capstyle="butt")
    ax.text(190, 350, "RSF fault unchanged", rotation=90, ha="right",
            va="center", color=TEAL, fontsize=10.5)
    ax.plot([0, 200], [0, 0], color=GRAY, lw=1.1, ls=(0, (5, 4)))
    ax.text(100, 18, "Original loading-end section", fontsize=7.6, ha="center",
            color=GRAY)

    for y in range(45, 500, 55):
        ax.annotate("", (0, y), (-40, y),
                    arrowprops={"arrowstyle": "-|>", "color": BLUE,
                                "lw": 1.4, "mutation_scale": 12})
    ax.text(-72, 265, "$p_n=16$ MPa   (only $0<y<500$)", rotation=90,
            color=BLUE, ha="center", va="center", fontsize=10.5)
    ax.text(-15, -100, "$p_n=0$", rotation=90, color=OCHRE,
            ha="center", va="center", fontsize=10.5)
    ax.annotate("No contact\n$\\mu=0$", (201, -92), (258, -58),
                fontsize=11, color=OCHRE, ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": OCHRE, "lw": 1.0})

    # The old stationary supports fix only their normal components, not both DOFs.
    for y in (75, 175, 275, 375, 475):
        ax.add_patch(Circle((353, y), 5, fc="white", ec=GRAY, lw=0.9))
    ax.plot([360, 360], [40, 515], color=GRAY, lw=1)
    ax.text(377, 280, "$u_x=0$", rotation=90, color=GRAY, va="center")
    for x in (220, 270, 320):
        ax.add_patch(Circle((x, 558), 5, fc="white", ec=GRAY, lw=0.9))
    ax.plot([208, 338], [565, 565], color=GRAY, lw=1)
    ax.text(272.5, 583, "$u_y=0$", ha="center", color=GRAY)

    # Displacement enters at the new remote end, along the original +y direction.
    ax.plot([0, 200], [-L_DRAW, -L_DRAW], color=RED, lw=3.2)
    for x in (25, 75, 125, 175):
        ax.annotate("", (x, -L_DRAW - 2), (x, -L_DRAW - 55),
                    arrowprops={"arrowstyle": "-|>", "color": RED,
                                "lw": 1.7, "mutation_scale": 13})
    ax.text(100, -285, "Prescribed $u_y(t)$ in $+y$", ha="center",
            color=RED, fontsize=11.3, weight="bold")

    ax.text(-3, 506, "$y=500$", ha="right", va="bottom", fontsize=10)
    ax.text(351, 545, "$y=550$", ha="left", va="top", fontsize=9.5)
    ax.text(214, 5, "$y=0$", ha="left", va="bottom", fontsize=10)
    ax.text(210, -L_DRAW - 2, "$y=-L_{\\rm ext}$", ha="left", va="top",
            fontsize=10.5, color=RED)
    dimension(ax, (-63, -L_DRAW), (-63, 0), "$L_{\\rm ext}$", (-15, 0), 90)
    dimension(ax, (417, 0), (417, 500), "Fault length: 500 mm", (0, 0), 90)
    for x in (0, 200, 345):
        ax.plot([x, x], [609, 637], color=GRAY, lw=0.7)
    dimension(ax, (0, 623), (200, 623), "200 mm")
    dimension(ax, (200, 623), (345, 623), "145 mm")

    origin = (328, -245)
    ax.annotate("", (393, origin[1]), origin,
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.2})
    ax.annotate("", (origin[0], -183), origin,
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.2})
    ax.text(398, origin[1], "$+x$", va="center")
    ax.text(origin[0], -178, "$+y$", ha="center")

    notes = fig.add_axes([0.61, 0.11, 0.355, 0.76])
    notes.axis("off")
    note(notes, 0.98, "01", "Geometry", [
        "Only the moving block extends toward -y.",
        "$x: 0\\rightarrow200$ mm; $y: -L_{\\rm ext}\\rightarrow500$ mm.",
        "Stationary block stays at $y=0\\rightarrow550$ mm.",
        "The 500 mm fault is not lengthened.",
    ])
    note(notes, 0.75, "02", "Original loaded / contact region", [
        "Normal traction acts on the original back face",
        "only: $x=0$, $0<y<500$ mm.",
        "Contact remains at $x=200$, $0<y<500$ mm.",
        "Keep the existing RSF profile on that fault.",
    ], BLUE)
    note(notes, 0.52, "03", "Added extension", [
        "No applied normal traction on its side faces.",
        "No opposing block, hence no frictional contact",
        "(schematically $\\mu=0$); side faces are free.",
        "Internal elastic stress is allowed.",
    ], OCHRE)
    note(notes, 0.29, "04", "Remote displacement loading", [
        "Move the loading face from $y=0$ to $y=-L_{\\rm ext}$.",
        "Apply the same +y displacement direction.",
        "The new segment supplies series compliance;",
        "do not prescribe displacement at the old section.",
    ], RED)

    fig.text(0.045, 0.064,
             "Energy intent: represent loading-system compliance. More length does not automatically store more energy at fixed displacement.",
             fontsize=10.1, color=GRAY)
    fig.text(0.045, 0.034,
             "FOR GEOMETRY APPROVAL ONLY  |  Extension length is illustrative; sweep values are not selected.  |  No job submitted.",
             fontsize=9.1, color=GRAY)
    for ext in ("png", "pdf", "svg"):
        path = OUT / f"{STEM}.{ext}"
        fig.savefig(path, dpi=240, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
