import numpy as np

class RGB2GRAY:
    @staticmethod
    def convert_to_grayscale(image):
        """
        Convert an image to grayscale using the standard luminance formula.

        Parameters:
        - image: The original BGR image as a NumPy array.

        Returns:
        - A new grayscale image as a NumPy array.
        """
        # Ensure the image has three channels (BGR)
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a BGR image with 3 channels.")

        # Get image dimensions
        height, width, channels = image.shape

        # Initialize a new array for the grayscale image
        grayscale_image = np.zeros((height, width), dtype=np.uint8)

        # Convert to grayscale using the luminance formula
        for y in range(height):
            for x in range(width):
                # Get BGR values (OpenCV uses BGR format)
                B, G, R = image[y, x]

                # Calculate grayscale value
                gray = int(0.299 * R + 0.587 * G + 0.114 * B)

                # Assign the grayscale value to the new image
                grayscale_image[y, x] = gray

        return grayscale_image
