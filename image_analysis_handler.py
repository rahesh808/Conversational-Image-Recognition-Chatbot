import io
import numpy as np
from PIL import Image
import google.generativeai as genai

class ImageHandler:
    def __init__(self, model_name="gemini-1.5-flash", temperature=0.7, max_output_tokens=100):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def image_analysis(self, image, prompt):
        # Handle the image input format
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            bytes_data = image

        # Convert image to bytes if not already in bytes format
        if 'pil_image' in locals():
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            bytes_data = img_byte_arr.getvalue()

        # Initialize the generative model
        model = genai.GenerativeModel(model_name=self.model_name)
        
        # Send the prompt and image as inline data
        response = model.generate_content(
            [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": bytes_data}},
            ],
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            },
        )

        # Resolve and return the response
        response.resolve()
        return response.text
