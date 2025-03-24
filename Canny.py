import cv2
import numpy as np


class Canny:
    def __init__(self):
        self.strong_para = None
        self.weak_para = None
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
        weak_threshold = gradient_mag.max() * self.weak_para
        strong_threshold = gradient_mag.max() * self.strong_para
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

        weak_edge = 75
        STRONG = 255

        for y in range(height):
            for x in range(width):
                if gradient_mag[y, x] < weak_threshold:
                    gradient_mag[y, x] = 0
                elif strong_threshold > gradient_mag[y, x] >= weak_threshold:
                    gradient_mag[y, x] = weak_edge
                else:
                    gradient_mag[y, x] = STRONG

        return gradient_mag

    def hysteresis(self, img):
        height, width = img.shape
        weak_edge = 75
        strong_edge = 255

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if img[y, x] == weak_edge:
                    if (img[y + 1, x - 1] == strong_edge or img[y + 1, x] == strong_edge or img[y + 1, x + 1] == strong_edge
                            or img[y, x - 1] == strong_edge or img[y, x + 1] == strong_edge
                            or img[y - 1, x - 1] == strong_edge or img[y - 1, x] == strong_edge or img[y - 1, x + 1] == strong_edge):
                        img[y, x] = strong_edge
                    else:
                        img[y, x] = 0
        return img

    def canny_detection(self, img):
        gray_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        color_img = cv2.imread(img)

        gradient_mag = self.double_thresh(gray_image)
        final_edges = self.hysteresis(gradient_mag)

        for y in range(final_edges.shape[0]):
            for x in range(final_edges.shape[1]):
                if final_edges[y, x] > 0:
                    color_img[y, x] = [0, 0, 255]

        return final_edges, color_img

