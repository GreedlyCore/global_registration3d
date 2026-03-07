#!/usr/bin/env python
"""
Minimal test: draw a single ellipsoid using glk.primitives.sphere() + transform.

The ellipsoid is defined by:
  mean       = [1, 2, 0]
  covariance = diag(4, 1, 0.25)   → semi-axes 2, 1, 0.5  at 1-sigma
               (rotated 45 deg around Z to make it visually obvious)

Run:
    python visualize/test_ellipsoid_primitive.py
"""
import numpy as np
from scipy.spatial.transform import Rotation
from pyridescence import guik, glk, imgui


def make_ellipsoid_transform(mean, covariance, n_sigma=2):
    """
    4x4 transform that maps the unit sphere onto the n-sigma ellipsoid.

      T[:3, :3] = V @ diag(n_sigma * sqrt(eigenvalues))
      T[:3,  3] = mean
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-9)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = eigenvectors @ np.diag(n_sigma * np.sqrt(eigenvalues))
    T[:3, 3] = mean
    return T


# --- define a test ellipsoid ---
mean = np.array([0.0, 0.0, 0.0])

# Elongated along X, flat along Z, rotated 45 deg around Z
R45 = Rotation.from_rotvec([0, 0, np.pi / 4]).as_matrix()
D   = np.diag([4.0, 1.0, 0.25])        # variances (σ² per axis before rotation)
cov = R45 @ D @ R45.T

n_sigma = 2

T = make_ellipsoid_transform(mean, cov, n_sigma)
print("Transform matrix:")
print(T)

# --- viewer ---
viewer = guik.LightViewer.instance()
viewer.set_title("Ellipsoid primitive test")
viewer.update_coord("world", guik.VertexColor().scale(1.5))

# wire ellipsoid – cyan
viewer.update_drawable("ellipsoid_wire",
                        glk.primitives.wire_sphere(),
                        guik.FlatColor(0.1, 0.8, 1.0, 1.0, T))

# solid ellipsoid – semi-transparent orange (drawn on top)
viewer.update_drawable("ellipsoid_solid",
                        glk.primitives.sphere(),
                        guik.FlatColor(1.0, 0.5, 0.1, 0.3, T))

print("Wire  ellipsoid: cyan")
print("Solid ellipsoid: semi-transparent orange")
print("Close the window to exit.")

while viewer.spin_once():
    pass
