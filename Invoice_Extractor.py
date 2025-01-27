import google.generativeai as genai
from PIL import Image
import io


class InvoiceExtractor:
    def __init__(self):
        # Initialize the Gemini model
        self.model = genai.GenerativeModel("gemini-pro-vision")
        self.input_prompt = """
        You are an expert in understanding invoices.
        You will receive input images as invoices &
        you will have to answer questions based on the input image.
        """

    def input_image_setup(self, uploaded_file):
        """
        Converts the uploaded file into the required format for the Gemini API.
        """
        if uploaded_file is not None:
            # Read the file into bytes
            bytes_data = uploaded_file.read()
            image_parts = [
                {
                    "mime_type": "image/jpeg",  # Assuming the file is JPEG, adjust as needed
                    "data": bytes_data
                }
            ]
            return image_parts
        else:
            raise FileNotFoundError("No file uploaded")

    def get_gemini_response(self, user_input, image_parts):
        """
        Sends the input, image, and predefined prompt to the Gemini API and returns the response.
        """
        response = self.model.generate_content([self.input_prompt, image_parts[0], user_input])
        return response.text
