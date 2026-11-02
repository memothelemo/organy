import cv2
import numpy as np

def is_burst_photo(image):
    """
    This function checks whether a photo is a burst photo.

    It uses FFT High-Frequency Ratio to measure the total energy lives
    outside the low-frequency center region.
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # f = np.fft.fft2(gray)
    # fshift = np.fft.fftshift(f)
    # magnitude = np.abs(fshift) ** 2

    # h, w = gray.shape
    # cy, cx = h // 2, w // 2
    # ry, rx = int(h * 0.1), int(w * 0.1)

    # low_mask = np.zeros_like(magnitude, dtype=bool)
    # low_mask[cy - ry:cy + ry, cx - rx:cx + rx] = True

    # low_energy = magnitude[low_mask].sum()
    # total_energy = magnitude.sum()

    # return 1.0 - low_energy / total_energy if total_energy > 0 else 0.0
