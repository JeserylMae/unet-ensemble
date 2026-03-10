import cv2
import numpy as np


class Frequency:
    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img
    
    def compute_frequency_spectrum(self, img):
        img = np.float32(img)

        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)

        magnitude = np.abs(fft_shift)
        magnitude_spectrum = np.log1p(magnitude)

        magnitude_spectrum = cv2.normalize(
            magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
        )

        return magnitude_spectrum
    
    