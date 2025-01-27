import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Initialize environment variables
load_dotenv()

class PDFProcessor:
    def __init__(self):
        # Load API key from environment and configure Google Generative AI
        self.api_key = os.getenv("API_KEY")
        genai.configure(api_key=self.api_key)
        self.embeddings_model = self.CustomGoogleEmbeddings(api_key=self.api_key)

    class CustomGoogleEmbeddings:
        def __init__(self, api_key, model="models/embedding-001"):
            self.api_key = api_key
            genai.configure(api_key=self.api_key)  # Configure Google Generative AI API
            self.model = model

        def embed_documents(self, texts):
       
            embeddings = []
            for text in texts:
            # Assuming 'genai.embed_content' is used for embedding text
                embedding = genai.embed_content(model=self.model, content=text)
                embeddings.append(embedding['embedding'])  # Assuming 'embedding' key holds the embeddings
            return embeddings
            

        def embed_query(self, text):
            """
            Embeds a single query (text) into a vector.
            """
            embedding = genai.embed_text(text)
            return embedding["embedding"]

    def get_pdf_text(self, pdf_docs):
        """Extracts text from the uploaded PDF files."""
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(self, text):
        """Splits text into manageable chunks."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)

    def get_vector_store(self, text_chunks):
        """Creates a FAISS vector store for the text chunks."""
        # Generate embeddings for text chunks and store them in FAISS
        embeddings = self.embeddings_model.embed_documents(text_chunks)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

    def get_conversational_chain(self):
        """Builds the conversational chain using LangChain."""
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
        If the answer is not in the provided context, say: "answer is not available in the context". Don't guess.\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)

    def process_user_question(self, user_question):
        """Processes a user question and retrieves the response from the vector store."""
        # Load the FAISS index and perform similarity search
        new_db = FAISS.load_local("faiss_index", self.embeddings_model, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        # Run the conversational QA chain to get the answer
        chain = self.get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]