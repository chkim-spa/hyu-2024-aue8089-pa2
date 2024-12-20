import math
import numpy as np

from distort_points import distort_points


def undistort_image(img: np.ndarray,
                    K: np.ndarray,
                    D: np.ndarray,
                    bilinear_interpolation: bool = False) -> np.ndarray:
    """
    Corrects an image for lens distortion.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)
        bilinear_interpolation: whether to use bilinear interpolation or not
    """

    height, width, *_ = img.shape
    undistorted_img = np.zeros_like(img)

    for x in range(width):
        for y in range(height):

            # apply distortion
            x_d = distort_points(np.array([[x, y]]), D, K)
            u, v = x_d[0, :]

            # bilinear interpolation
            u1 = math.floor(u)
            v1 = math.floor(v)

            in_image = (u1 >= 0) & (u1+1 < width) & (v1 >= 0) & (v1+1 < height)
            if not in_image:
                continue

            if bilinear_interpolation:
                a = u - u1
                b = v - v1

                # Bilinear interpolation
                I11 = img[v1, u1]       # Top-left
                I21 = img[v1, u1 + 1]   # Top-right
                I12 = img[v1 + 1, u1]   # Bottom-left
                I22 = img[v1 + 1, u1 + 1]  # Bottom-right

                # [TODO] weighted sum of pixel values in img
                value = (1 - a) * (1 - b) * I11 + a * (1 - b) * I21 + (1 - a) * b * I12 + a * b * I22
                undistorted_img[y, x] = np.clip(value, 0, 255)  # Clipping for safety
            else:
                # [TODO] nearest neighbor
                u_nearest = int(round(u))
                v_nearest = int(round(v))
                if 0 <= u_nearest < width and 0 <= v_nearest < height:
                    undistorted_img[y, x] = img[v_nearest, u_nearest]

    return undistorted_img
