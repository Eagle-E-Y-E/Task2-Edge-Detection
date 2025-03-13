import math
import numpy as np

class Snake():
    @staticmethod
    def compute_gradient(img_gray):
        """
        Compute the gradient magnitude of the grayscale image manually
        using a Sobel operator (ignoring the border pixels).
        """
        rows, cols = img_gray.shape
        grad = np.zeros((rows, cols), dtype=float)
        # Loop over interior pixels
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Sobel masks approximations for x and y derivatives
                gx = (-1 * img_gray[i - 1][j - 1] + 1 * img_gray[i - 1][j + 1] +
                    -2 * img_gray[i][j - 1]     + 2 * img_gray[i][j + 1] +
                    -1 * img_gray[i + 1][j - 1] + 1 * img_gray[i + 1][j + 1])
                gy = ( 1 * img_gray[i - 1][j - 1] + 2 * img_gray[i - 1][j] + 1 * img_gray[i - 1][j + 1] +
                    0 * img_gray[i][j - 1]     + 0 * img_gray[i][j]     + 0 * img_gray[i][j + 1] +
                    -1 * img_gray[i + 1][j - 1] - 2 * img_gray[i + 1][j] - 1 * img_gray[i + 1][j + 1])
                grad[i][j] = math.sqrt(gx * gx + gy * gy)
        return grad

    @staticmethod
    def compute_external_energy(grad):
        """
        Define the external energy at each pixel.
        Here we choose E_ext = –gradient so that high gradients (edges)
        give low energy.
        """
        rows, cols = grad.shape
        E_ext = np.zeros((rows, cols), dtype=float)
        for i in range(rows):
            for j in range(cols):
                E_ext[i][j] = -grad[i][j]
        return E_ext

    @staticmethod
    def initialize_contour(center, radius, num_points):
        """
        Initialize a closed contour in the shape of a circle.
        center: (x, y) coordinate (origin at top-left, x is column)
        radius: radius of the circle
        num_points: number of contour points
        Returns a list of [x, y] coordinates.
        """
        contour = []
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            x = int(center[0] + radius * math.cos(theta))
            y = int(center[1] + radius * math.sin(theta))
            contour.append([x, y])
        return contour

    @staticmethod
    def evolve_snake(contour, E_ext, alpha, beta, gamma, iterations):
        """
        Evolve the snake using a greedy algorithm.
        For each point on the snake we search in a 3x3 window.
        The total energy for a candidate position is computed as the weighted sum of:
        - Continuity energy: punishes deviation from the average of the two neighbors.
        - Curvature energy: approximates the second derivative (smoothness).
        - External energy from the image.
        The snake is updated until no point moves or a maximum number of iterations is done.
        """
        prev_movement = [[0, 0] for _ in range(len(contour))]
        # current_alpha = alpha
        # current_beta = beta
        # current_gamma = gamma
        rows, cols = E_ext.shape
        # Make a deep copy of the initial contour
        new_contour = [pt[:] for pt in contour]
        
        for it in range(iterations):
            change = 0
            new_points = new_contour[:]  # prepare a copy for simultaneous updates
            # if it > 0 and it % 100 == 0:
            #     current_alpha *= 0.9
            #     current_beta *= 0.9
            #     current_gamma *= 0.9
            for i in range(len(new_contour)):
                x, y = new_contour[i]
                best_energy = float('inf')
                best_x, best_y = x, y
                # Search in neighbors (3x3 window)
                for dx in [-3, -2, -1, 0, 1, 2, 3]:
                    for dy in [-3, -2, -1, 0, 1, 2, 3]:
                        candidate_x = x + dx
                        candidate_y = y + dy
                        # Check image boundaries
                        if candidate_x < 0 or candidate_x >= cols or candidate_y < 0 or candidate_y >= rows:
                            continue
                        # Internal energy: continuity and curvature terms
                        # Get previous and next points (the list is circular)
                        prev_pt = new_contour[i - 1]      # in Python, negative indices work as wrap-around
                        next_pt = new_contour[(i + 1) % len(new_contour)]
                        
                        # Continuity energy: enforce closeness to average of neighbors
                        avg_x = (prev_pt[0] + next_pt[0]) / 2.0
                        avg_y = (prev_pt[1] + next_pt[1]) / 2.0
                        E_cont = (candidate_x - avg_x) ** 2 + (candidate_y - avg_y) ** 2
                        
                        # Curvature energy: second derivative approximation
                        E_curv = (prev_pt[0] - 2 * candidate_x + next_pt[0]) ** 2 \
                            + (prev_pt[1] - 2 * candidate_y + next_pt[1]) ** 2
                        
                        # External energy: from the image (note: index order is [row, col])
                        E_image = E_ext[candidate_y][candidate_x]
                        
                        total_energy = alpha * E_cont + beta * E_curv + gamma * E_image
                        
                        if total_energy < best_energy:
                            best_energy = total_energy
                            best_x = candidate_x
                            best_y = candidate_y
                # Update the point if a better candidate is found
                if best_x != x or best_y != y:
                    # movement = [best_x - x, best_y - y]
                    # Add momentum (e.g., 0.2 * previous movement)
                    # best_x += int(0.2 * prev_movement[i][0])
                    # best_y += int(0.2 * prev_movement[i][1])
                    new_points[i] = [best_x, best_y]
                    # prev_movement[i] = movement
                    change += 1
            # Update contour points simultaneously after taking the local minima.
            new_contour = [pt[:] for pt in new_points]
            if change == 0:
                break
        return new_contour

    @staticmethod
    def compute_chain_code(contour):
        """
        Given a closed contour (a list of [x,y] points ordered sequentially),
        compute the chain code (using 8-connectivity).
        The mapping is as follows:
            0 : (1,  0)
            1 : (1, -1)
            2 : (0, -1)
            3 : (-1, -1)
            4 : (-1,  0)
            5 : (-1,  1)
            6 : (0,  1)
            7 : (1,  1)
        """
        chain_code = []
        directions = {
            (1, 0): 0,
            (1, -1): 1,
            (0, -1): 2,
            (-1, -1): 3,
            (-1, 0): 4,
            (-1, 1): 5,
            (0, 1): 6,
            (1, 1): 7
        }
        n = len(contour)
        for i in range(n):
            curr = contour[i]
            nxt = contour[(i + 1) % n]
            dx = nxt[0] - curr[0]
            dy = nxt[1] - curr[1]
            # Normalize the step to one pixel (if moving more than one, reduce to sign)
            if dx != 0:
                dx = int(dx / abs(dx))
            if dy != 0:
                dy = int(dy / abs(dy))
            code = directions.get((dx, dy), None)
            if code is not None:
                chain_code.append(code)
            else:
                chain_code.append('?')  # for any irregular move
        return chain_code
    
    @staticmethod
    def compute_perimeter(contour):
        """
        Compute the contour perimeter by summing the Euclidean distances between
        consecutive contour points. This method is less sensitive to the number
        of contour points than using a chain code-based approach.
        """
        perimeter = 0.0
        n = len(contour)
        for i in range(n):
            x1, y1 = contour[i]
            x2, y2 = contour[(i + 1) % n]  # wrap around to the first point
            dx = x2 - x1
            dy = y2 - y1
            perimeter += math.sqrt(dx * dx + dy * dy)
        return perimeter

    @staticmethod
    def compute_area(contour):
        """
        Compute the area enclosed by the contour using the shoelace formula.
        """
        area = 0.0
        n = len(contour)
        for i in range(n):
            x1, y1 = contour[i]
            x2, y2 = contour[(i + 1) % n]
            area += x1 * y2 - y1 * x2
        return abs(area) / 2.0

    @staticmethod
    def draw_contour(image, contour, color=(0, 0, 255)):
        """
        Draw the contour over the image by setting the pixel (if within bounds)
        to the given color.
        """
        for pt in contour:
            x, y = pt
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image[y][x] = list(color)
        return image