from PyQt5.QtWidgets import QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QLabel, QVBoxLayout, QWidget, QFileDialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer


def load_pixmap_to_label(label: QLabel):
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getOpenFileName(None, "Open Image File", "",
                                               "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)

    if file_path:
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        label.setPixmap(scaled_pixmap)
        label.setAlignment(Qt.AlignCenter)

def display_image_Graphics_scene(self, view, image):
        pixmap = self.convert_cv_to_pixmap(image)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        view.setScene(scene)
        view.fitInView(
            scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    
