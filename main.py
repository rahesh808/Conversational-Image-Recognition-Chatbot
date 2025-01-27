import os
from dotenv import load_dotenv
import google.generativeai as genai
import gradio as gr
from PIL import Image
from chat_handler import ChatHandler
from image_analysis_handler import ImageHandler
from webcam_capture import WebcamImageProcessor
from Invoice_Extractor import InvoiceExtractor
from langchain_community.vectorstores import FAISS
from pdf_processor import PDFProcessor
#from voice_interaction import VoiceInteraction
import io
import cv2

load_dotenv()
API_KEY = os.getenv("API_KEY")

if API_KEY is None:
    gr.Chatbot("API Key is not set. Please set the API key in the .env file.")
else:
    genai.configure(api_key=API_KEY)
    

chat_handler = ChatHandler()
def text_chat_wrapper(text, max_output_tokens):
    return chat_handler.text_chat(text, max_output_tokens)

image_handler = ImageHandler()


def image_analysis_wrapper(image, prompt):
    return image_handler.image_analysis(image, prompt)

invoice_extractor = InvoiceExtractor()

def process_invoice(uploaded_file, user_input):
    """
    Processes the uploaded invoice image and user query to return a response.
    """
    if uploaded_file is None:
        return "Error: No image uploaded."
    try:
        # Convert PIL image to bytes for Gemini API
        with io.BytesIO() as output:
            uploaded_file.save(output, format="JPEG")
            image_bytes = output.getvalue()

        image_parts = [
            {
                "mime_type": "image/jpeg",  # Assuming JPEG format
                "data": image_bytes
            }
        ]

        # Get response from Gemini
        response = invoice_extractor.get_gemini_response(user_input, image_parts)
        return response
    except Exception as e:
        return f"Error: {str(e)}"
    
faiss_vector_store = None

# Function to process PDF files
def process_pdf_files(pdf_files):
    global faiss_vector_store
    
    if not pdf_files:
        return "No PDFs uploaded!", None
    pdf_processor = PDFProcessor()
    text = pdf_processor.get_pdf_text(pdf_files)
    text_chunks = pdf_processor.get_text_chunks(text)
    vector_store = FAISS.from_texts(text_chunks, embedding=pdf_processor.embeddings_model.embed_documents)
    vector_store.save_local("faiss_index")
    faiss_vector_store = FAISS.load_local("faiss_index", pdf_processor.embeddings_model, allow_dangerous_deserialization=True) 
    return "PDFs processed successfully!", None


# Function to handle questions based on processed PDFs
def handle_question(question):
    global faiss_vector_store
    
    if not faiss_vector_store:
        return "Please process the PDFs before asking questions."
    pdf_processor = PDFProcessor()
    answer = pdf_processor.process_user_question(question)
    
    return answer




custom_body = """
body {
    background-color: #f0f8ff !important; /* Light blue background */
    color: black !important; /* Text color */
}
"""

custom_button = """
#custom-button {
    background-color: #4CAF50 !important; /* Green background */
    color: white !important; /* White text */
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
    cursor: pointer !important;
}
"""

# Gradio UI
with gr.Blocks(theme=custom_body) as demo:
    gr.Markdown("#Chatbot", elem_id="header")
    
    with gr.Tab("💬 Text Chat"):
        with gr.Row():
            with gr.Column(scale=4):
                text_input = gr.Textbox(label="Your message", placeholder="Type your message here...")
                max_tokens_slider = gr.Slider(minimum=100, maximum=1000, value=500, step=50, label="Max Output Tokens")
            with gr.Column(scale=1):
                text_button = gr.Button("Send", variant="primary", elem_id = "custom-button")
        
        text_output = gr.Chatbot(height=400, elem_id="text-chat-output")
        
        text_button.click(text_chat_wrapper, inputs=[text_input, max_tokens_slider], outputs=text_output)
        
    with gr.Tab("🖼️ Image Analysis"):
          with gr.Row():
              with gr.Column(scale=1):
                  with gr.Row():
                      image_input = gr.Image(label="Upload Image", type="pil")
                      webcam_button = gr.Button("Capture from Webcam")
              with gr.Column(scale=1):
                  image_prompt = gr.Textbox(label="Prompt", placeholder="Ask about the image...")
                  image_prompt_voice = gr.Audio(label="Prompt via Voice")
                  image_button = gr.Button("Analyze", variant="primary", elem_id = "custom-button")
          image_output = gr.Markdown(label="Analysis Result", elem_id="image-analysis-output")
          def capture_image_from_webcam():
              cap = cv2.VideoCapture(0)
              ret, frame = cap.read()
              cap.release()
              return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

          webcam_button.click(capture_image_from_webcam, outputs=image_input)
          image_button.click(image_analysis_wrapper, inputs=[image_input, image_prompt], outputs=image_output)
          image_prompt_voice.change(lambda x: image_analysis_wrapper(image_input.value, x), inputs=image_prompt_voice, outputs=image_output)
          
    with gr.Tab("Invoice Understanding"):
        with gr.Row():
            with gr.Column(scale=1):
                uploaded_file = gr.Image(label="Upload Invoice Image", type="pil")
            with gr.Column(scale=1):
                user_input = gr.Textbox(label="Your Question", placeholder="Ask a question about the invoice...")
        response_output = gr.Markdown(label="Gemini Response")

        # Process button
        process_button = gr.Button("Extract Invoice Details")

        # Link the button to the processing function
        process_button.click(
            fn=process_invoice,
            inputs=[uploaded_file, user_input],
            outputs=response_output
        )
    
    
        
    with gr.Tab("📥 Process & Chat with PDFs"):
        gr.Markdown("### Upload PDFs and Ask Questions")
        with gr.Row():
            pdf_input = gr.File(label="Upload PDF Files", file_types=[".pdf"], type="filepath", file_count="multiple")
        with gr.Row():
            process_output = gr.Textbox(label="Processing Status", interactive=False)
        with gr.Row():
            question_input = gr.Textbox(label="Ask a Question", placeholder="Type your question about the uploaded PDFs...")
        with gr.Row():
            question_output = gr.Textbox(label="Answer", interactive=False)
        with gr.Row():
            process_and_ask_button = gr.Button("Process PDFs and Get Answer")

        # Button Click
        
        process_and_ask_button.click(
        fn=lambda pdf_files, question: (
            process_pdf_files(pdf_files) if faiss_vector_store is None else ("PDFs already processed!", None),
            handle_question(question) if faiss_vector_store else None
        ),
        inputs=[pdf_input, question_input],
        outputs=[process_output, question_output],
    )

    
    

demo.launch(share=True)