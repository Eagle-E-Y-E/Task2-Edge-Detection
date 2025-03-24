import cv2
import numpy as np


class Canny:
    def __init__(self):
        self.image = None

    def gaussian_kernel(self, size, sigma):
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        gaussina_blur = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
        return gaussina_blur

    def sobel_kernel(self, img):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        output_sobel_x = self.apply_kernel(img, sobel_x)
        output_sobel_y = self.apply_kernel(img, sobel_y)
        return output_sobel_x, output_sobel_y

    def apply_kernel(self, img, kernel):
        height, width = img.shape[0], img.shape[1]
        pad = kernel.shape[0] // 2
        # pad the edges with zeroes to perserve the boundary pixels
        padded_image = np.pad(img, pad, mode='reflect')
        # Make a zeros matrix of the same size as the image to store values later
        output_image = np.zeros_like(img)
        n = kernel.shape[0]
        for i in range(height):
            for j in range(width):
                # from i to i+n because the End in slicing is excluded
                region = padded_image[i:i + n, j:j + n]
                output_image[i, j] = np.sum(region * kernel) / 4

        return output_image

    def calc_gradient(self, img):
        blurred_image = self.apply_kernel(img, self.gaussian_kernel(5, 1.4))
        Gx, Gy = self.sobel_kernel(blurred_image)
        gradient_mag = np.sqrt(Gx ** 2 + Gy ** 2)
        gradient_angle = np.arctan2(Gy, Gx)
        gradient_mag = (gradient_mag / gradient_mag.max()) * 255
        gradient_mag = gradient_mag.astype(np.uint8)
        return gradient_mag, gradient_angle

    def suppress(self, img):
        gradient_mag, gradient_angle = self.calc_gradient(img)
        weak_threshold = gradient_mag.max() * 0.4
        strong_threshold = gradient_mag.max() * 0.8
        height, width = img.shape[0], img.shape[1]
        for x in range(width):
            for y in range(height):
                current_angle = gradient_angle[y, x]
                current_angle = abs(current_angle - 180) if abs(current_angle) >= 180 else abs(current_angle)

                #Finding Neighbors

                if current_angle <= 22.5:
                    neighb_1_x, neighb_1_y = x - 1, y
                    neighb_2_x, neighb_2_y = x + 1, y

                elif 22.5 < current_angle <= (22.5 + 45):
                    neighb_1_x, neighb_1_y = x - 1, y - 1
                    neighb_2_x, neighb_2_y = x + 1, y + 1

                elif (22.5 + 45) < current_angle <= (22.5 + 90):
                    neighb_1_x, neighb_1_y = x, y - 1
                    neighb_2_x, neighb_2_y = x, y + 1

                elif (22.5 + 90) < current_angle <= (22.5 + 135):
                    neighb_1_x, neighb_1_y = x - 1, y + 1
                    neighb_2_x, neighb_2_y = x + 1, y - 1

                elif (22.5 + 135) < current_angle <= (22.5 + 180):
                    neighb_1_x, neighb_1_y = x - 1, y
                    neighb_2_x, neighb_2_y = x + 1, y

                # Non-maximum suppression
                if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                    if gradient_mag[y, x] < gradient_mag[neighb_1_y, neighb_1_x]:
                        gradient_mag[y, x] = 0
                        continue

                if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                    if gradient_mag[y, x] < gradient_mag[neighb_2_y, neighb_2_x]:
                        gradient_mag[y, x] = 0

        return gradient_mag, weak_threshold, strong_threshold

    def double_thresh(self, img):
        gradient_mag, weak_threshold, strong_threshold = self.suppress(img)
        height, width = img.shape[0], img.shape[1]

        for x in range(width):
            for y in range(height):
                if gradient_mag[y, x] < weak_threshold:
                    gradient_mag[y, x] = 0
                elif strong_threshold > gradient_mag[y, x] >= weak_threshold:
                    pass

        return gradient_mag

    def canny_detection(self):
        img = cv2.imread('ED-image1_gray.png', cv2.IMREAD_GRAYSCALE)
        color_img = cv2.imread('ED-image1_gray.png')
        gradient_mag = self.double_thresh(img)
        for y in range(gradient_mag.shape[0]):
            for x in range(gradient_mag.shape[1]):
                if gradient_mag[y, x] > 0:
                    color_img[y, x] = [0, 0, 255]
        return color_img


canny_detector = Canny()
cv2.imwrite('Canny_canny.jpg', canny_detector.canny_detection())
