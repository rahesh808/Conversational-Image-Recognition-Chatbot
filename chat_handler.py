import google.generativeai as genai
import google.generativeai.types as glm 

class ChatHandler:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name=model_name)

    @staticmethod
    def text_chat(text, max_output_tokens):
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
        response = model.generate_content(
        text,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": max_output_tokens,
        },
        stream=True,
    )
        response.resolve()
        return [("You", text), ("Assistant", response.text)]

