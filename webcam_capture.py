import cv2
import numpy as np
from PIL import Image

class WebcamImageProcessor:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index

    def adjust_brightness_contrast(self, image, brightness=50, contrast=30):
        # Adjust brightness and contrast of the image
        beta = brightness - 50  # Brightness shift, 50 is neutral
        alpha = contrast / 50.0  # Contrast scaling, 50 is neutral
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted_image

    def color_balance(self, image):
        # Split the image into color channels (BGR format)
        b, g, r = cv2.split(image)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        # Merge the channels back together
        balanced_image = cv2.merge([b, g, r])
        return balanced_image

    def capture_and_process_image(self):
        # Open the webcam
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print("Error: Unable to access the camera")
            return None

        # Capture a frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print("Error: Failed to capture image from webcam")
            return None

        # Adjust brightness and contrast
        adjusted_image = self.adjust_brightness_contrast(frame)

        # Apply color balance
        color_balanced_image = self.color_balance(adjusted_image)

        # Convert the processed image from BGR to RGB (for displaying with PIL)
        processed_image_rgb = cv2.cvtColor(color_balanced_image, cv2.COLOR_BGR2RGB)

        # Return as a PIL Image
        return Image.fromarray(processed_image_rgb)
