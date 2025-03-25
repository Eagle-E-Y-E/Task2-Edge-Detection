import cv2
import numpy as np
import matplotlib.pyplot as plt

def hough_transform(image, theta_res=1, rho_res=1):
    # Step 1: Convert to grayscale and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Edge detection

    # Step 2: Define Hough Space
    height, width = edges.shape
    theta_max = 180  # Degrees
    theta_vals = np.deg2rad(np.arange(0, theta_max, theta_res))  # Convert degrees to radians

    rho_max = int(np.hypot(height, width))  # Max possible rho (diagonal length)
    rhos = np.arange(-rho_max, rho_max, rho_res)  # Define rho range

    accumulator = np.zeros((len(rhos), len(theta_vals)), dtype=np.int32)  # Hough space accumulator

    # Step 3: Populate Accumulator
    edge_points = np.argwhere(edges)  # Get edge pixel coordinates

    for y, x in edge_points:
        for theta_idx, theta in enumerate(theta_vals):
            rho = int(x * np.cos(theta) + y * np.sin(theta))  # Compute rho
            rho_idx = np.argmin(np.abs(rhos - rho))  # Find nearest rho index
            accumulator[rho_idx, theta_idx] += 1  # Vote in the accumulator

    return accumulator, rhos, theta_vals

def draw_detected_lines(image, accumulator, rhos, thetas, threshold=100):
    detected_image = image.copy()
    
    # Get the indices of votes above the threshold
    rho_theta_pairs = np.argwhere(accumulator > threshold)

    for rho_idx, theta_idx in rho_theta_pairs:
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]

        # Convert polar coordinates back to Cartesian (draw lines)
        a = np.cos(theta)
        b = np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(detected_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return detected_image

# Load image
image = cv2.imread("image.jpg")

# Apply Hough Transform
accumulator, rhos, thetas = hough_transform(image)

# Visualize Accumulator
plt.imshow(accumulator, cmap='hot', aspect='auto', extent=[0, 180, -max(rhos), max(rhos)])
plt.title("Hough Transform Accumulator")
plt.xlabel("Theta (degrees)")
plt.ylabel("Rho (pixels)")
plt.colorbar(label="Votes")
plt.show()

# Draw detected lines
result_image = draw_detected_lines(image, accumulator, rhos, thetas)

# Show final image
cv2.imshow("Detected Lines", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
