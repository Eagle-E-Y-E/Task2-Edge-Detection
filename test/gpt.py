import cv2
import numpy as np
from skimage.draw import polygon_perimeter, polygon
from skimage.measure import find_contours, regionprops, label as measure_label
import matplotlib.pyplot as plt


def initialize_contour(image, threshold=100):
    """Finds an initial contour using edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # pixel value > threshold = 255, else 0
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def custom_initialize_contour(image, threshold=100):
    """Finds an initial contour using edge detection without OpenCV contour functions."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        # Simple RGB to grayscale conversion
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image.copy()

    # Manual thresholding
    binary = np.zeros_like(gray)
    binary[gray > threshold] = 255

    # Edge detection using Sobel operators
    # Apply simple gradient operators
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Compute gradients
    grad_x = np.zeros_like(gray, dtype=float)
    grad_y = np.zeros_like(gray, dtype=float)

    # Apply convolution for edge detection
    for i in range(1, binary.shape[0]-1):
        for j in range(1, binary.shape[1]-1):
            grad_x[i, j] = np.sum(binary[i-1:i+2, j-1:j+2] * gx)
            grad_y[i, j] = np.sum(binary[i-1:i+2, j-1:j+2] * gy)

    # Gradient magnitude
    edges = np.sqrt(grad_x**2 + grad_y**2)
    edges = (edges > 40).astype(np.uint8) * 255  # Threshold edges

    # Find connected components and extract the largest one as the contour
    labeled, num_labels = measure_label(edges, connectivity=2, return_num=True)
    if num_labels == 0:
        return None

    # Find the largest component
    largest_component = None
    max_area = 0

    for label in range(1, num_labels + 1):
        component = (labeled == label).astype(np.uint8)
        area = np.sum(component)

        if area > max_area:
            max_area = area
            largest_component = component

    # Extract contour points from the largest component
    contour_points = []
    for i in range(largest_component.shape[0]):
        for j in range(largest_component.shape[1]):
            if largest_component[i, j] > 0:
                # Check if it's an edge pixel (has at least one neighbor that is 0)
                if (i == 0 or i == largest_component.shape[0]-1 or
                    j == 0 or j == largest_component.shape[1]-1 or
                    largest_component[i-1, j] == 0 or largest_component[i+1, j] == 0 or
                        largest_component[i, j-1] == 0 or largest_component[i, j+1] == 0):
                    # Note: OpenCV format is [x, y] which is [column, row]
                    contour_points.append([j, i])

    # Convert to the format expected by the rest of the code (similar to cv2.findContours output)
    if not contour_points:
        return None

    contour_array = np.array(contour_points, dtype=np.int32).reshape(-1, 1, 2)
    return contour_array


def greedy_snake(image, contour, iterations=50):
    """Evolves the contour using a greedy algorithm."""
    for _ in range(iterations):
        for i, point in enumerate(contour):
            x, y = point[0]
            neighborhood = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            best_point = min(neighborhood, key=lambda p: image[p[1], p[0]] if 0 <=
                             p[1] < image.shape[0] and 0 <= p[0] < image.shape[1] else 255)
            contour[i] = np.array([[best_point[0], best_point[1]]])
    return contour


def compute_chain_code(contour):
    """Computes the Freeman Chain Code from the evolved contour."""
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1),
                  (-1, 0), (-1, -1), (0, -1), (1, -1)]
    chain_code = []
    for i in range(len(contour) - 1):
        dx, dy = contour[i+1][0] - contour[i][0]
        code = directions.index((dx, dy)) if (dx, dy) in directions else None
        if code is not None:
            chain_code.append(code)
    return chain_code


def compute_perimeter_area(contour):
    """Computes the perimeter and area using the Shoelace theorem."""
    perimeter = np.sum(
        np.sqrt(np.sum(np.diff(contour[:, 0, :], axis=0) ** 2, axis=1)))
    x, y = contour[:, 0, 0], contour[:, 0, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return perimeter, area


def main():
    image = cv2.imread('ED-image1_gray.png')
    if image is None:
        print("Error: Could not load image. Please check the file path.")
        return

    initial_contour = initialize_contour(image)
    if initial_contour is None:
        print("No contour found!")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    evolved_contour = greedy_snake(gray, initial_contour.copy())
    chain_code = compute_chain_code(evolved_contour)
    perimeter, area = compute_perimeter_area(evolved_contour)

    print("Initial Contour:", initialize_contour(image))
    print("custom initial Contour:", custom_initialize_contour(image))

    print("Chain Code:", chain_code)
    print("Perimeter:", perimeter)
    print("Area:", area)

    # Plot results
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.plot(initial_contour[:, 0, 0], initial_contour[:,
             0, 1], 'r--', label='Initial Contour')
    # plt.plot(evolved_contour[:, 0, 0], evolved_contour[:,
    #          0, 1], 'g-', label='Evolved Contour')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
