import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget, QSlider, QLabel, QFileDialog, )
from PyQt5.QtGui import QPixmap, QImage, QPainterPath, QPen, QColor, QPainter
from PyQt5.QtCore import QPointF, Qt
from PyQt5 import uic

from Canny import Canny
from utils import load_pixmap_to_label, display_image_Graphics_scene


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.filter_input = None
        self.setWindowTitle("Active Contour Model with PyQt5")
        self.setGeometry(100, 100, 1000, 00)

        uic.loadUi("UI.ui", self)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.contour_item = None
        layout = QVBoxLayout(self.widget)
        layout.addWidget(self.view)

        # Buttons and sliders
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_btn)

        self.init_btn.clicked.connect(self.toggle_drawing)
        self.reset_btn.clicked.connect(self.reset_contour)
        self.evolve_btn.clicked.connect(self.evolve_snake)
        # self.analyze_contour_btn.clicked.connect(self.analyze_contour)
        self.save_btn.clicked.connect(self.save_chain_code)
        self.filter_btn.clicked.connect(self.canny_detection)

        self.image = None
        self.E_ext = None
        self.contour_points = []
        self.drawing_mode = False
        self.chain_code = []
        self.alpha_slider.setValue(1)
        self.beta_slider.setValue(1)
        self.gamma_slider.setValue(10)
        self.iterations_slider.setValue(200)
        self.update_label(self.alpha_slider, self.alpha_label)
        self.update_label(self.beta_slider, self.beta_label)
        self.update_label(self.gamma_slider, self.gamma_label)
        self.update_label(self.iterations_slider, self.iterations_label)

        self.alpha_slider.valueChanged.connect(
            lambda: self.update_label(self.alpha_slider, self.alpha_label))
        self.beta_slider.valueChanged.connect(
            lambda: self.update_label(self.beta_slider, self.beta_label))
        self.gamma_slider.valueChanged.connect(
            lambda: self.update_label(self.gamma_slider, self.gamma_label))
        self.iterations_slider.valueChanged.connect(
            lambda: self.update_label(self.iterations_slider, self.iterations_label))

        ######################  Tab 2

        self.filter_input.mouseDoubleClickEvent = self.doubleClickHandler
        # filter_output1 for edges
        # filter_output2 for result
        # filter_btn for applying canny

        # slider1 , slider2 , slider3 if well use them
        self.slider1.valueChanged.connect(
            lambda: self.alpha_2_label.setText(f"{self.slider1.value()/100}"))
        self.slider2.valueChanged.connect(
            lambda: self.beta_2_label.setText(f"{self.slider2.value()/100}"))

    def doubleClickHandler(self, event):
        self.img_path = load_pixmap_to_label(self.filter_input)

    def update_label(self, slider, label):
        label.setText(f"{slider.value()}")

    def load_image(self):
        """Load and process the image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.image is None:
                QMessageBox.critical(self, "Error", "Could not load image.")
                return
        else:
            return

        # Compute external energy
        sobel_operator = Canny()
        grad_x, grad_y = sobel_operator.sobel_kernel(self.image)
        # grad_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=3)
        # # grad_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        self.E_ext = -grad_mag ** 2

        # Display image
        self.reset_contour()

    def toggle_drawing(self):
        """Toggle drawing mode."""
        self.drawing_mode = not self.drawing_mode
        self.init_btn.setText(
            "Stop Initializing" if self.drawing_mode else "Initialize Contour")

    def reset_contour(self):
        h, w = self.image.shape
        qimage = QImage(self.image.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, w, h)
        # Fit the image to the view
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.contour_points = []
        self.contour_item = None
        self.chain_code = []

    def mousePressEvent(self, event):
        """Handle mouse clicks to add contour points."""
        if self.drawing_mode and event.button() == Qt.LeftButton:
            pos = self.view.mapToScene(
                self.view.mapFromGlobal(event.globalPos()))
            x2, y2 = int(pos.x()), int(pos.y())
            if 0 <= x2 < self.image.shape[1] and 0 <= y2 < self.image.shape[0]:
                if len(self.contour_points) > 0:
                    x1, y1 = self.contour_points[-1].x(
                    ), self.contour_points[-1].y()
                    n = int(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5) + 1
                    x_values = np.linspace(x1, x2, n)
                    y_values = np.linspace(y1, y2, n)
                    qpoints = [QPointF(x, y)
                               for x, y in zip(x_values, y_values)]
                    self.contour_points += qpoints
                    self.last_points = len(qpoints)
                else:
                    self.contour_points.append(QPointF(x2, y2))
                    self.last_points = 1

        if event.button() == Qt.MiddleButton and self.contour_points:
            self.contour_points = self.contour_points[:-self.last_points]
        self.update_contour()

        if event.button() == Qt.MiddleButton and self.contour_points:
            self.contour_points = self.contour_points[:-self.last_points]
        self.update_contour()

    def resample_contour_points(self, num_points=None):
        """
        Resample the contour points so that they are uniformly spaced along the curve.

        If num_points is None, the function keeps the current number of points.
        For a closed contour (not in drawing mode), the first point is appended again at the end for interpolation.
        """
        if len(self.contour_points) < 2:
            return

        # Use the current number of points if not specified.
        if num_points is None:
            num_points = len(self.contour_points)

        # Convert the list of QPointF to separate x and y arrays.
        x = np.array([p.x() for p in self.contour_points])
        y = np.array([p.y() for p in self.contour_points])

        # If the contour is closed (i.e. not in drawing mode), append the first point to the end.
        if not self.drawing_mode:
            x = np.append(x, x[0])
            y = np.append(y, y[0])

        # Calculate the cumulative arc length along the contour.
        # The first element is 0. Each subsequent element is the sum of distances so far.
        distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        cumulative_length = np.concatenate(([0], np.cumsum(distances)))
        total_length = cumulative_length[-1]

        # Generate uniformly spaced "target" distances along the total length.
        target_lengths = np.linspace(0, total_length, num_points)

        # Interpolate new x and y coordinates for the given target distances.
        new_x = np.interp(target_lengths, cumulative_length, x)
        new_y = np.interp(target_lengths, cumulative_length, y)

        # Update the contour points with the new, uniformly spaced points.
        self.contour_points = [QPointF(nx, ny) for nx, ny in zip(new_x, new_y)]

    def update_contour(self):
        """Update the displayed contour."""
        if self.contour_item:
            self.scene.removeItem(self.contour_item)
        if len(self.contour_points) > 1:
            # Create a QPainterPath
            path = QPainterPath()
            path.moveTo(self.contour_points[0])
            for p in self.contour_points[1:]:
                path.lineTo(p)

            # Close the contour if initialized
            if not self.drawing_mode and self.contour_points:
                path.closeSubpath()

            # Create a QPen with the desired color (e.g., red) and optionally set width
            pen = QPen(QColor("red"))
            pen.setWidth(2)  # Adjust the width as needed

            # Add the path to the scene with the specified pen
            self.contour_item = self.scene.addPath(path, pen)
            self.resample_contour_points()

    def evolve_snake(self):
        """Evolve the contour using the greedy algorithm."""
        if not self.contour_points or self.drawing_mode:
            return
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        self.drawing_mode = False
        alpha = self.alpha_slider.value() / 100.0
        beta = self.beta_slider.value() / 100.0
        gamma = self.gamma_slider.value() / 10.0
        print(alpha, beta, gamma)
        max_iterations = self.iterations_slider.value()

        for iteration in range(max_iterations):
            moved = False
            new_points = self.contour_points.copy()
            for i in range(len(self.contour_points)):
                p = self.contour_points[i]
                prev = self.contour_points[i - 1]
                next = self.contour_points[(i + 1) % len(self.contour_points)]
                min_energy = float('inf')
                best_pos = p

                # Check neighborhood
                for dx in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
                    for dy in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
                        cx, cy = int(p.x() + dx), int(p.y() + dy)
                        if not (0 <= cx < self.image.shape[1] and 0 <= cy < self.image.shape[0]):
                            continue
                        candidate = QPointF(cx, cy)
                        # Elasticity
                        elast = ((cx - prev.x()) ** 2 + (cy - prev.y()) ** 2 +
                                 (next.x() - cx) ** 2 + (next.y() - cy) ** 2)
                        # Stiffness
                        stiff = ((prev.x() - 2 * cx + next.x()) ** 2 +
                                 (prev.y() - 2 * cy + next.y()) ** 2)
                        # External energy
                        e_ext = self.E_ext[cy, cx]
                        energy = alpha * elast + beta * stiff + gamma * e_ext
                        if energy < min_energy:
                            min_energy = energy
                            best_pos = candidate
                if best_pos != p:
                    new_points[i] = best_pos
                    moved = True
            self.contour_points = new_points
            self.update_contour()
            QApplication.processEvents()  # Keep GUI responsive
            if not moved:
                break
        print("Finished after", iteration, "iterations")
        self.analyze_contour()

    def generate_chain_code(self):
        """
        Generate the 8-directional Freeman chain code for the contour.

        Direction coding:
        3 2 1
        4   0
        5 6 7
        """
        if len(self.contour_points) < 2:
            return []

        # Direction vectors for 8-directional chain code
        dirs = [(1, 0), (1, -1), (0, -1), (-1, -1),
                (-1, 0), (-1, 1), (0, 1), (1, 1)]

        chain_code = []

        for i in range(len(self.contour_points)):
            current = self.contour_points[i]
            next_point = self.contour_points[(
                                                     i + 1) % len(self.contour_points)]

            # Calculate difference vector
            dx = int(next_point.x() - current.x())
            dy = int(next_point.y() - current.y())

            # Find the closest direction
            if dx == 0 and dy == 0:
                continue  # Skip duplicate points

            # Normalize the direction vector for comparison
            length = max(abs(dx), abs(dy))
            dx_norm = dx / length
            dy_norm = dy / length

            best_dir = 0
            best_similarity = -float('inf')

            for dir_idx, (dir_x, dir_y) in enumerate(dirs):
                # Calculate dot product as similarity measure
                similarity = dx_norm * dir_x + dy_norm * dir_y
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_dir = dir_idx

            # For longer steps, repeat the direction
            for _ in range(length):
                chain_code.append(best_dir)

        return chain_code

    def calculate_perimeter(self):
        """
        Calculate the perimeter of the contour manually by summing the 
        Euclidean distances between consecutive points.
        """
        if len(self.contour_points) < 2:
            return 0

        perimeter = 0
        for i in range(len(self.contour_points)):
            current = self.contour_points[i]
            next_point = self.contour_points[(
                                                     i + 1) % len(self.contour_points)]

            # Calculate Euclidean distance between current and next point
            dx = next_point.x() - current.x()
            dy = next_point.y() - current.y()
            distance = (dx ** 2 + dy ** 2) ** 0.5

            perimeter += distance

        return perimeter

    def calculate_area(self):
        """
        Calculate the area inside the contour using the Shoelace formula 
        (Gauss's area formula).

        This formula computes the area of a simple polygon by using the 
        coordinates of its vertices:
        Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
        """
        if len(self.contour_points) < 3:
            return 0

        # Extract x and y coordinates
        x = [p.x() for p in self.contour_points]
        y = [p.y() for p in self.contour_points]

        # Add the first point at the end to close the polygon
        x.append(x[0])
        y.append(y[0])

        # Apply the Shoelace formula
        area = 0
        for i in range(len(x) - 1):
            area += (x[i] * y[i + 1]) - (x[i + 1] * y[i])

        # Take the absolute value and multiply by 0.5
        area = abs(area) * 0.5

        return area

    def analyze_contour(self):
        """Analyze the contour and update information labels."""
        if not self.contour_points or len(self.contour_points) < 3:
            QMessageBox.warning(
                self, "Warning", "Please initialize a valid contour first.")
            return

        # Calculate perimeter
        perimeter = self.calculate_perimeter()

        # Calculate area
        area = self.calculate_area()

        # Generate chain code
        self.chain_code = self.generate_chain_code()

        print("Perimeter:", perimeter)

        # Update labels
        self.perimeter_label.setText(f"Perimeter: {perimeter:.2f} pixels")
        self.area_label.setText(f"Area: {area:.2f} square pixels")
        self.chain_code_label.setText(
            f"Chain Code Length: {len(self.chain_code)} elements")

    def save_chain_code(self):
        """Save the chain code to a file."""
        if not self.chain_code:
            QMessageBox.warning(
                self, "Warning", "Please analyze the contour first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Chain Code", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w') as f:
                f.write(','.join(map(str, self.chain_code)))
            QMessageBox.information(
                self, "Success", "Chain code saved successfully.")

    def canny_detection(self):
        canny_detector = Canny()
        print(self.slider1.value())
        print(self.slider2.value())
        canny_detector.weak_para = self.slider1.value() / 100
        canny_detector.strong_para = self.slider2.value() / 100
        final_edges, marked_image = canny_detector.canny_detection(self.img_path)
        display_image_Graphics_scene(self.filter_output1, final_edges)
        display_image_Graphics_scene(self.filter_output2, marked_image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
