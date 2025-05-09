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
from PyQt5 import QtWidgets
import Hough

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.filter_input = None
        self.setWindowTitle("Active Contour Model")
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

        self.original_image = None
        self.image = None
        self.E_ext = None
        self.contour_points = []
        self.drawing_mode = False
        self.chain_code = []
        self.alpha_slider.setValue(1)
        self.beta_slider.setValue(1)
        self.gamma_slider.setValue(10)
        self.iterations_slider.setValue(200)
        self.window_size_slider.setValue(3)
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
        # self.window_size_slider.valueChanged.connect(
        #     lambda: self.winow_size_label.setText(f"{self.window_size_slider.value()}"))

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
        self.window_size_slider.valueChanged.connect(self.adjust_to_odd_value)



        ############# Hough UI link
         # Setup Hybrid Button
        # self.btn_hybrid.clicked.connect(self.hybrid_image)

        # # Setup Hough Button
        self.apply_Hough_btn.clicked.connect(self.hough_transform)

        self.btn_load_7.clicked.connect(self.load_image_Hough)

        self.setup_images_view()
        self.hough_settings_layout.setEnabled(True)

    def setup_images_view(self):
        """
        Adjust the shape and scales of the widgets
        Remove unnecessary options
        :return:
        """
        for widget in [self.img4_output, self.img4_input_2]:
            widget.ui.histogram.hide()
            widget.ui.roiBtn.hide()
            widget.ui.menuBtn.hide()
            widget.ui.roiPlot.hide()
            widget.getView().setAspectLocked(False)
            widget.view.setAspectLocked(False)

    def load_image_Hough(self):
        repo_path = "./src/Images"
        filename, file_format = QtWidgets.QFileDialog.getOpenFileName(None, "Load Image", repo_path,
                                                                      "*;;" "*.jpg;;" "*.jpeg;;" "*.png;;")
        img_name = filename.split('/')[-1]
        if filename == "":
            pass
        else:
            image = cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE).T

            bgr_img = cv2.imread(filename)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            imgbyte_rgb = cv2.transpose(rgb_img)
            self.display_image(imgbyte_rgb, self.img4_input_2)
            self.Hough_image_org = imgbyte_rgb


    def hough_transform(self):
        """
        Apply a hough transformation to detect lines or circles in the given image
        :return:
        """

        hough_image = None

        # Get Parameters Values from the user
        min_radius = int(self.text_min_radius.text())
        max_radius = int(self.text_max_radius.text())
        num_votes = int(self.text_votes.text())

        if self.radioButton_lines.isChecked():
            hough_image = Hough.hough_lines(source=self.Hough_image_org, num_peaks=num_votes)
        elif self.radioButton_circles.isChecked():
            hough_image = Hough.hough_circles(source=self.Hough_image_org, min_radius=min_radius,
                                              max_radius=max_radius)

        try:
            self.display_image(source=hough_image, widget=self.img4_output)
        except TypeError:
            print("Cannot display Image")


    def display_image(self,source, widget):
        """
        Display the given data
        :param source: 2d numpy array
        :param widget: ImageView object
        :return:
        """
        widget.setImage(source)
        widget.view.setRange(xRange=[0, source.shape[0]], yRange=[0, source.shape[1]],
                             padding=0)
        widget.ui.roiPlot.hide()


        

    def adjust_to_odd_value(self):
        value = self.window_size_slider.value()
        if value % 2 == 0:
            self.window_size_slider.setValue(value + 1 if value < self.window_size_slider.maximum() else value - 1)
        self.window_size_label.setText(f"{self.window_size_slider.value()}")


    def doubleClickHandler(self, event):
        self.img_path = load_pixmap_to_label(self.filter_input)

    def update_label(self, slider, label):
        label.setText(f"{slider.value() / 100}")

    def sobel_manual(self, image):
        """
        Manually compute the Sobel gradients of a grayscale image.

        Parameters:
            image (np.array): Grayscale input image as a 2D NumPy array.

        Returns:
            (grad_x, grad_y): Tuple of 2D NumPy arrays corresponding to the
                            gradients in x and y directions.
        """
        # Define Sobel kernels for x and y directions.
        kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float64)

        kernel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=np.float64)
        
        # Get the dimensions of the image
        height, width = image.shape
        
        # Convert image to float for precision if it's not already.
        image = image.astype(np.float64)
        
        # Create output arrays to store gradients
        grad_x = np.zeros_like(image)
        grad_y = np.zeros_like(image)
        
        # Zero-pad the image on all sides to handle the borders.
        padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
        
        # Iterate over every pixel in the original image.
        for i in range(height):
            for j in range(width):
                # Extract the current 3x3 region.
                region = padded_image[i:i+3, j:j+3]
                # Compute the convolution (element-wise multiplication and sum).
                grad_x[i, j] = np.sum(region * kernel_x)
                grad_y[i, j] = np.sum(region * kernel_y)
        
        return grad_x, grad_y
        

    def load_image(self):
        """Load and process the image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                QMessageBox.critical(self, "Error", "Could not load image.")
                return
        else:
            return

        self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.blur(self.image, (9, 9))
        # Compute gradients using the manually implemented sobel operator.
        grad_x, grad_y = self.sobel_manual(self.image)
        # Compute gradient magnitude.
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        # Compute external energy (example usage; adjust as needed).
        self.E_ext = -grad_mag ** 2

        # Display image and reset contour (implementation not shown).
        self.reset_contour()

    def toggle_drawing(self):
        """Toggle drawing mode."""
        self.drawing_mode = not self.drawing_mode
        self.init_btn.setText(
            "Stop Initializing" if self.drawing_mode else "Initialize Contour")

    def reset_contour(self):
        h, w, c = self.original_image.shape 
        qimage = QImage(self.original_image.data, w, h, 3 * w, QImage.Format_BGR888)  # Use Format_BGR888 if the image is in BGR format (typical for OpenCV)
        pixmap = QPixmap.fromImage(qimage)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, w, h)
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
                    n = int(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.25) + 1
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

        x = np.array([p.x() for p in self.contour_points])
        y = np.array([p.y() for p in self.contour_points])

        # ensure the contour is closed
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

        new_x = np.interp(target_lengths, cumulative_length, x)
        new_y = np.interp(target_lengths, cumulative_length, y)

        self.contour_points = [QPointF(nx, ny) for nx, ny in zip(new_x, new_y)]

    def update_contour(self):
        if self.contour_item:
            self.scene.removeItem(self.contour_item)
        if len(self.contour_points) > 1:
            path = QPainterPath()
            path.moveTo(self.contour_points[0])
            for p in self.contour_points[1:]:
                path.lineTo(p)

            # Close the contour
            if not self.drawing_mode and self.contour_points:
                path.closeSubpath()

            pen = QPen(QColor("red"))
            pen.setWidth(2)  

            self.contour_item = self.scene.addPath(path, pen)
            self.resample_contour_points()

    def evolve_snake(self):
        """the greedy algorithm."""
        if not self.contour_points or self.drawing_mode:
            return
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        self.drawing_mode = False
        alpha = self.alpha_slider.value() / 100.0
        beta = self.beta_slider.value() / 100.0
        gamma = self.gamma_slider.value() / 100.0
        # print(alpha, beta, gamma)
        max_iterations = self.iterations_slider.value()
        window_lookUP = {
            3: [-1, 0, 1],
            5: [-2, -1, 0, 1, 2],
            7: [-3, -2, -1, 0, 1, 2, 3],
            9: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
            11: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
            13 : [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        }
        window_size = self.window_size_slider.value()

        for iteration in range(max_iterations):
            moves = 0
            new_points = self.contour_points.copy()
            for i in range(len(self.contour_points)):
                p = self.contour_points[i]
                prev = self.contour_points[i - 1]
                next = self.contour_points[(i + 1) % len(self.contour_points)]
                min_energy = float('inf')
                best_pos = p

                # Check neighborhood
                for dx in window_lookUP[window_size]:
                    for dy in window_lookUP[window_size]:
                        cx, cy = int(p.x() + dx), int(p.y() + dy)
                        if not (0 <= cx < self.image.shape[1] and 0 <= cy < self.image.shape[0]):
                            continue
                        candidate = QPointF(cx, cy)
                        # Elasticity: Encourages the snake to remain connected smoothly. (measures distance)
                        elast = ((cx - prev.x()) ** 2 + (cy - prev.y()) ** 2 +
                                 (next.x() - cx) ** 2 + (next.y() - cy) ** 2)
                        # Stiffness: Promotes smoothness and penalizes abrupt changes
                        stiff = ((prev.x() - 2 * cx + next.x()) ** 2 + # derived from second derivative (change in direction)
                                 (prev.y() - 2 * cy + next.y()) ** 2)
                        # External energy: Pulls the snake toward edges (the closest to edges, the highest the external energy)
                        e_ext = self.E_ext[cy, cx]
                        energy = alpha * elast + beta * stiff + gamma * e_ext
                        if energy < min_energy:
                            min_energy = energy
                            best_pos = candidate
                if best_pos != p:
                    new_points[i] = best_pos
                    moves += 1
            self.contour_points = new_points
            self.update_contour()
            QApplication.processEvents()  # Keep GUI responsive
            print(moves / len(new_points))
            if moves / len(new_points) < 0.3:
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

        area = abs(area) * 0.5

        return area

    def analyze_contour(self):
        """Analyze the contour and update information labels."""
        if not self.contour_points or len(self.contour_points) < 3:
            QMessageBox.warning(
                self, "Warning", "Please initialize a valid contour first.")
            return

        perimeter = self.calculate_perimeter()

        area = self.calculate_area()

        self.chain_code = self.generate_chain_code()

        # print("Perimeter:", perimeter)

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
        # print(self.slider1.value())
        # print(self.slider2.value())
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
