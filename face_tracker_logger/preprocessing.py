import numpy as np
import cv2


class ImagePreprocessing:
    """
    A Utility class for preprocessing the image
    """
    def __init__(self):
        pass

    def calculate_clearness(self, im, size=10):
        # Bigger mean = better
        if im.dtype == "float64" or im.dtype == "float32":
            im = np.array(im * 255, dtype=np.uint8)

        image = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        # grab the dimensions of the image and use the dimensions to
        # derive the center (x, y)-coordinates
        (h, w) = image.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))
        # compute the FFT to find the frequency transform, then shift
        # the zero frequency component (i.e., DC component located at
        # the top-left corner) to the center where it will be more
        # easy to analyze
        fft = np.fft.fft2(image)
        fftShift = np.fft.fftshift(fft)

        # zero-out the center of the FFT shift (i.e., remove low
        # frequencies), apply the inverse shift such that the DC
        # component once again becomes the top-left, and then apply
        # the inverse FFT
        fftShift[cY - size : cY + size, cX - size : cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)

        # compute the magnitude spectrum of the reconstructed image,
        # then compute the mean of the magnitude values
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        # the image will be considered "blurry" if the mean value of the
        # magnitudes is less than the threshold value
        return mean

    def prewhiten(self, x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError("Dimension should be 3 or 4")

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0 / np.sqrt(size))
        y = (x - mean) / std_adj
        return y

    def l2_normalize(self, x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(
            np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon)
        )
        return output


    def calc_embs(self, face_recognization_model, imgs, margin, batch_size):
        imgs = imgs.astype(np.float32)
        aligned_images = self.prewhiten(imgs)
        pd = []
        for start in range(0, len(aligned_images), batch_size):
            pd.append(
                face_recognization_model(aligned_images[start : start + batch_size])
            )
        embs = self.l2_normalize(np.concatenate(pd))

        return embs
