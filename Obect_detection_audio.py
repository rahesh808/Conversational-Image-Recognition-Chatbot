from lib2to3.pgen2 import token
from transformers import pipeline
import torch
from PIL import Image, ImageDraw, ImageFont
import scipy.io.wavfile as wavfile
from huggingface_hub import login
import os
os.environ["USER_AGENT"] = "my-awesome-app/1.0.0"

login(token="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")
# Initialize the DETR object detection pipeline
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else fallback to CPU
object_detector = pipeline("object-detection", model="facebook/detr-resnet-101", device=device)

# Initialize the TTS model
narrator = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs", device=device, use_auth_token="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")

# Define the function to generate audio from text
def generate_audio(text):
    # Generate the narrated text
    narrated_text = narrator(text)

    # Save the audio to a WAV file
    wavfile.write("output.wav", rate=narrated_text["sampling_rate"], data=narrated_text["audio"][0])

    # Return the path to the saved audio file
    return "output.wav"

# Function to read the objects and generate a descriptive text
def read_objects(detection_objects):
    object_counts = {}

    # Count the occurrences of each label
    for detection in detection_objects:
        label = detection['label']
        if label in object_counts:
            object_counts[label] += 1
        else:
            object_counts[label] = 1

    # Generate the response string
    response = "This picture contains"
    labels = list(object_counts.keys())
    for i, label in enumerate(labels):
        response += f" {object_counts[label]} {label}"
        if object_counts[label] > 1:
            response += "s"
        if i < len(labels) - 2:
            response += ","
        elif i == len(labels) - 2:
            response += " and"

    response += "."

    return response

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, detections, font_path=None, font_size=20):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    for detection in detections:
        box = detection['box']
        xmin, ymin, xmax, ymax = box

        # Draw the bounding box
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

        label = detection['label']
        score = detection['score']
        text = f"{label} {score:.2f}"

        text_size = draw.textbbox((xmin, ymin), text, font=font)
        draw.rectangle([(text_size[0], text_size[1]), (text_size[2], text_size[3])], fill="red")
        draw.text((xmin, ymin), text, fill="white", font=font)

    return draw_image

# Function to process the image, detect objects, and generate response with audio
def detect_object(image):
    # Detect objects using the object detection model
    raw_image = image
    output = object_detector(raw_image)

    # Draw bounding boxes around detected objects
    processed_image = draw_bounding_boxes(raw_image, output)

    # Generate descriptive text based on the detections
    natural_text = read_objects(output)

    # Generate audio based on the descriptive text
    processed_audio = generate_audio(natural_text)

    return processed_image, processed_audio

# Example usage
if __name__ == "__main__":
    # Load an image
    image_path = "example.jpg"  # Replace with your image path
    image = Image.open(image_path)

    # Process the image and generate audio
    processed_image, audio_file = detect_object(image)

    # Save or display the processed image
    processed_image.show()

    # Optionally, save the processed image
    processed_image.save("output_image.jpg")

    print(f"Audio saved to {audio_file}")
