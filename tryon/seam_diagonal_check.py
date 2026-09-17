"""
Samples slightly INSIDE each layer's own region near the diagonal
seam, not directly ON the boundary line. Sampling exactly on an
anti-aliased edge is inherently ambiguous (alpha can land anywhere
0-255 regardless of whether the underlying cut is correct) — this
was the flaw in the original version of this script.

Run: python -m tryon.seam_diagonal_check
"""

import numpy as np
from PIL import Image

OUTPUT_DIR = "outputs"
OFFSET_PX = 3  # sample this far INSIDE each layer's own region, not on the edge


def main():
    upper = np.array(Image.open(f"{OUTPUT_DIR}/layer_upper_only.png").convert("RGBA"))
    lower = np.array(Image.open(f"{OUTPUT_DIR}/layer_lower_only.png").convert("RGBA"))

    # Same anchors used throughout tonight
    ua_left = (61, 229)
    ua_right = (162, 213)

    dx = ua_right[0] - ua_left[0]
    dy = ua_right[1] - ua_left[1]
    length = np.hypot(dx, dy)
    perp_x, perp_y = -dy / length, dx / length  # unit vector perpendicular to the seam

    print(f"{'t':<6}{'upper_pt':<12}{'lower_pt':<12}{'upper_alpha':<12}{'lower_alpha':<12}{'upper_rgb':<18}{'lower_rgb':<18}{'rgb_diff'}")
    print("-" * 100)

    h, w = upper.shape[0], upper.shape[1]

    for t in np.linspace(0, 1, 11):
        base_x = ua_left[0] + t * dx
        base_y = ua_left[1] + t * dy

        # Upper layer sits ABOVE the seam -> offset in -perp direction
        # Lower layer sits BELOW the seam -> offset in +perp direction
        ux = int(round(base_x - perp_x * OFFSET_PX))
        uy = int(round(base_y - perp_y * OFFSET_PX))
        lx = int(round(base_x + perp_x * OFFSET_PX))
        ly = int(round(base_y + perp_y * OFFSET_PX))

        if not (0 <= ux < w and 0 <= uy < h and 0 <= lx < w and 0 <= ly < h):
            print(f"{t:<6.2f}  out of bounds, skipping")
            continue

        ua = upper[uy, ux]
        la = lower[ly, lx]
        diff = np.linalg.norm(ua[:3].astype(int) - la[:3].astype(int))

        print(f"{t:<6.2f}({ux},{uy})   ({lx},{ly})   {ua[3]:<12}{la[3]:<12}{str(tuple(int(v) for v in ua[:3])):<18}{str(tuple(int(v) for v in la[:3])):<18}{diff:.1f}")

    print(f"\n>>> DECISION GATE: with sampling moved safely inside each "
          f"layer's own territory (not on the ambiguous edge itself), do "
          f"both layers now show solid alpha (~255) and closely matching "
          f"RGB all along t=0..1?")


if __name__ == "__main__":
    main()