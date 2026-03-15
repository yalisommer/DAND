"""
Preprocess an OBJ mesh: rotate from Y-up to Z-up and write a new OBJ.

OBJ files use Y-up by convention. This script applies x'=x, y'=-z, z'=y
so the mesh stands upright with Z as the vertical axis.

Usage:
    python normalize_mesh.py input.obj                # writes input_normalized.obj
    python normalize_mesh.py input.obj -o output.obj  # explicit output path
"""

import argparse
import os
import numpy as np


_Y_UP_TO_Z_UP = np.array([[1, 0, 0],
                           [0, 0, -1],
                           [0, 1, 0]], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="Rotate an OBJ mesh from Y-up to Z-up"
    )
    parser.add_argument("input", help="Path to input OBJ file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: input_normalized.obj)")
    args = parser.parse_args()

    # Read raw vertices and non-vertex lines
    vertices = []
    other_lines = []
    with open(args.input, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("v ") and len(stripped.split()) >= 4:
                parts = stripped.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif stripped and not stripped.startswith("#"):
                other_lines.append(line)

    vertices = np.array(vertices, dtype=np.float64)
    print(f"Loaded {args.input}: {len(vertices)} vertices")
    print(f"  Before: X=[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}] "
          f"Y=[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}] "
          f"Z=[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")

    # Rotate Y-up → Z-up
    vertices = vertices @ _Y_UP_TO_Z_UP.T

    print(f"  After:  X=[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}] "
          f"Y=[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}] "
          f"Z=[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}_normalized{ext}"
    else:
        out_path = args.output

    with open(out_path, "w") as f:
        f.write(f"# Rotated from Y-up to Z-up: {os.path.basename(args.input)}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for line in other_lines:
            f.write(line)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
