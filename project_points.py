import numpy as np

from distort_points import distort_points


def project_points(points_3d: np.ndarray,
                   K: np.ndarray,
                   D: np.ndarray) -> np.ndarray:
    """
    Projects 3d points to the image plane, given the camera matrix,
    and distortion coefficients.

    Args:
        points_3d: 3d points (3xN) -> (N, 3)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1) -> (2x1)

    Returns:
        projected_points: 2d points (2xN) -> (Nx2)
    """

    # [TODO] get image coordinates
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
 
    k1, k2 = D.ravel() # p1, p2
    projected_points = []

    # X, Y, Z = points_3d[0], points_3d[1], points_3d[2]
    for X, Y, Z in points_3d:
        x, y = X/Z, Y/Z

        r2 = x**2 + y**2
        radial = 1 # 1 + k1 * r2 + k2 * r2**2

        # [TODO] apply distortion
        x_dis, y_dis = x * radial, y * radial

        u = fx * x_dis + cx
        v = fy * y_dis + cy
        projected_points.append((u, v))

    px_locs = np.array(projected_points)
    dd  = distort_points(px_locs, D, K)
    return dd

