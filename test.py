import cv2
import numpy as np

from Snake import Snake
from RGB2GRAY import RGB2GRAY

# Read the image (only cv2.imread and cv2.imshow are allowed)
img = cv2.imread('ED-image1_gray.png')
if img is None:
    print("Error: Image not found.")
    exit(1)

# Convert image to grayscale manually (avoid using cv2.cvtColor)
if len(img.shape) == 3:
    gray = RGB2GRAY.convert_to_grayscale(img)
rows, cols, _ = img.shape

# Compute image gradient and external energy (edge attraction)
gradient = Snake.compute_gradient(gray)
E_ext = Snake.compute_external_energy(gradient)

# Initialize a circular contour around the image center
center = (cols // 2, rows // 2.2)         # (x, y)
radius = 200             # example: quarter of min dimension
num_points = 1000
init_contour = Snake.initialize_contour(center, radius, num_points)

# Evolve the snake using the greedy algorithm.
# Adjust the weights (alpha, beta, gamma) and iteration count as needed.
final_contour = Snake.evolve_snake(init_contour, E_ext, alpha=0.5, beta=0.1, gamma=1.0, iterations=100)

# Represent the output as chain code
chain_code = Snake.compute_chain_code(final_contour)

# Compute contour perimeter and area
perimeter = Snake.compute_perimeter(final_contour)
area = Snake.compute_area(final_contour)

# Print the results
# print("Chain Code:", chain_code)
print("Perimeter:", perimeter)
print("Area:", area)

# Draw the final contour on a copy of the original image
img_contour = img.copy()
img_contour = Snake.draw_contour(img_contour, final_contour, color=(0, 0, 255))

# Show the original and processed images using OpenCV
cv2.imshow("Original Image", img)
cv2.imshow("Final Contour", img_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()
