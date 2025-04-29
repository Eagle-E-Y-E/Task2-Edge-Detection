# Task2-Edge-Detection

## 🖼️ Overview

**Task2-Edge-Detection** is a Python-based application that implements various edge detection algorithms to analyze and process images. It provides functionalities such as grayscale conversion, histogram equalization, noise addition, filtering, and edge detection using different operators. The application features a graphical user interface (GUI) for user-friendly interaction.

## Snake 🐍 (Active Contour)
![active_contour_gif](snake.gif)

## ✨ Features

- **Grayscale Conversion**: Transform RGB images to grayscale.
- **Histogram Equalization**: Enhance image contrast using histogram equalization.
- **Noise Addition**: Introduce different types of noise to images for testing.
- **Filtering Techniques**: Apply various filters to images.
- **Edge Detection**: Detect edges within images using operators like Sobel, Prewitt, and Canny.
- **Graphical User Interface**: Interact with the application through a GUI.

## 🛠️ Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Eagle-E-Y-E/Task2-Edge-Detection.git
   cd Task2-Edge-Detection
   ```

2. **Create a Virtual Environment (Optional but Recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: Ensure that `requirements.txt` is present in the repository with all necessary dependencies listed.*

## 🚀 Usage

1. **Run the Application**:

   ```bash
   python main.py
   ```

2. **Using the GUI**:

   - Load an image using the GUI.
   - Select the desired image processing operation.
   - View and save the processed image.

## 📁 File Structure

- `main.py`: Entry point of the application.
- `ui_2.ui`: Qt Designer file for the GUI layout.
- `ui_handler.py`: Handles GUI interactions and events.
- `RGB2GRAY.py`: Contains functions for grayscale conversion.
- `Histogram_Equalization.py`: Implements histogram equalization.
- `add_noise.py`: Functions to add noise to images.
- `apply_filter.py`: Applies various filters to images.
- `EdgeDetection.py`: Implements edge detection algorithms.
- `images/`: Directory containing sample images.
- `report/`: Contains project reports and documentation.

## 📌 Future Enhancements

- Implement additional edge detection algorithms.
- Enhance the GUI with more features and better user experience.
- Optimize performance for processing large images.
- Add support for batch processing of images.

