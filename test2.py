import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget, QSlider, QLabel, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage, QPainterPath, QPen, QColor
from PyQt5.QtCore import QPointF, Qt

class SnakeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Active Contour Model with PyQt5")
        self.setGeometry(100, 100, 800, 600)

        # Graphics setup
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.contour_item = None

        # Control panel
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.view)

        # Buttons and sliders
        self.load_btn = QPushButton("Load Image")
        self.init_btn = QPushButton("Initialize Contour")
        self.reset_btn = QPushButton("Reset contour")
        self.evolve_btn = QPushButton("Evolve")
        self.load_btn.clicked.connect(self.load_image)
        self.init_btn.clicked.connect(self.toggle_drawing)
        self.reset_btn.clicked.connect(self.reset_contour)
        self.evolve_btn.clicked.connect(self.evolve_snake)

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.beta_slider = QSlider(Qt.Horizontal)
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.iterations_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(1, 1000)
        self.beta_slider.setRange(1, 1000)
        self.gamma_slider.setRange(1, 100)
        self.iterations_slider.setRange(10, 1000)
        self.alpha_slider.setValue(1)
        self.beta_slider.setValue(1)
        self.gamma_slider.setValue(10)
        self.iterations_slider.setValue(200)

        layout.addWidget(self.load_btn)
        layout.addWidget(self.init_btn)
        layout.addWidget(self.reset_btn)
        layout.addWidget(self.evolve_btn)
        layout.addWidget(QLabel("Alpha (Elasticity)"))
        layout.addWidget(self.alpha_slider)
        layout.addWidget(QLabel("Beta (Stiffness)"))
        layout.addWidget(self.beta_slider)
        layout.addWidget(QLabel("Gamma (External)"))
        layout.addWidget(self.gamma_slider)
        layout.addWidget(QLabel("Iterations"))
        layout.addWidget(self.iterations_slider)

        # State variables
        self.image = None
        self.E_ext = None
        self.contour_points = []
        self.drawing_mode = False

    #     self.alpha_slider.valueChanged.connect(lambda: self.update_label(self.alpha_slider, self.alpha_label))
    #     self.beta_slider.valueChanged.connect(lambda: self.update_label(self.beta_slider, self.beta_label))
    #     self.gamma_slider.valueChanged.connect(lambda: self.update_label(self.gamma_slider, self.gamma_label))
    #     self.iterations_slider.valueChanged.connect(lambda: self.update_label(self.iterations_slider, self.iter_label))

    # def update_label(self, slider, label):
    #     label.setText(f"{slider.value()}")

    def load_image(self):
        """Load and process the image."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.image is None:
                QMessageBox.critical(self, "Error", "Could not load image.")
                return
        else: 
            return
        # Compute external energy
        grad_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        self.E_ext = -grad_mag**2

        # Display image
        self.reset_contour()

    def toggle_drawing(self):
        """Toggle drawing mode."""
        self.drawing_mode = not self.drawing_mode
        self.init_btn.setText("Stop Initializing" if self.drawing_mode else "Initialize Contour")

    def reset_contour(self):
        h, w = self.image.shape
        qimage = QImage(self.image.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, w, h)
        self.contour_points = []
        self.contour_item = None

    def mousePressEvent(self, event):
        """Handle mouse clicks to add contour points."""
        if self.drawing_mode and event.button() == Qt.LeftButton:
            pos = self.view.mapToScene(event.pos())
            x2, y2 = int(pos.x()), int(pos.y())
            if 0 <= x2 < self.image.shape[1] and 0 <= y2 < self.image.shape[0]:
                if len(self.contour_points) > 0:
                    x1, y1 = self.contour_points[-1].x(), self.contour_points[-1].y()
                    n = int(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5) + 1
                    print(n)
                    x_values = np.linspace(x1, x2, n)
                    y_values = np.linspace(y1, y2, n)
                    qpoints = [QPointF(x, y) for x, y in zip(x_values, y_values)]
                    self.contour_points += qpoints
                    self.last_points = len(qpoints)
                else:
                    self.contour_points.append(QPointF(x2, y2))
                    self.last_points = 1
        
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
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
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

                # Check 3x3 neighborhood
                for dx in [-5,-4,-3,-2, -1, 0, 1, 2,3,4,5]:
                    for dy in [-5,-4,-3,-2, -1, 0, 1, 2,3,4,5]:
                        cx, cy = int(p.x() + dx), int(p.y() + dy)
                        if not (0 <= cx < self.image.shape[1] and 0 <= cy < self.image.shape[0]):
                            continue
                        candidate = QPointF(cx, cy)
                        # Elasticity
                        elast = ((cx - prev.x())**2 + (cy - prev.y())**2 +
                                 (next.x() - cx)**2 + (next.y() - cy)**2)
                        # Stiffness
                        stiff = ((prev.x() - 2*cx + next.x())**2 +
                                 (prev.y() - 2*cy + next.y())**2)
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SnakeApp()
    window.show()
    sys.exit(app.exec_())