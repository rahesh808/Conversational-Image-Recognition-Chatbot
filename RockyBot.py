import os
import pickle
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings  # Corrected import
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
import faiss
from langchain.chat_models import ChatOpenAI  # Correct import

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is found
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file or pass it manually.")

openai.api_key = api_key  # Set the API key globally

class RockyBot:
    def __init__(self, api_key, file_path="faiss_store_openai.pkl"):
        self.file_path = file_path
        self.api_key = api_key
        openai.api_key = api_key  # Ensure API key is set for OpenAI
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=self.api_key)  # Use ChatOpenAI model

    def load_and_process_urls(self, urls):
        """Loads and processes the URLs to create a vector store."""
        loader = UnstructuredURLLoader(urls=urls)
        print("Loading data from URLs...✅✅✅")
        data = loader.load()

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        print("Splitting text...✅✅✅")
        docs = text_splitter.split_documents(data)

        # Generate embeddings using OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)  # Pass API key explicitly
        print("Building embedding vectors...✅✅✅")
        
        # Create FAISS vector store from documents
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Save the FAISS index and necessary data to a file
        faiss.write_index(vectorstore.index, self.file_path)  # Save FAISS index using FAISS's write_index
        
        # Save docstore and index_to_docstore_id as well
        with open(self.file_path + "_docstore.pkl", "wb") as f:
            pickle.dump(vectorstore.docstore, f)
        with open(self.file_path + "_index_to_docstore_id.pkl", "wb") as f:
            pickle.dump(vectorstore.index_to_docstore_id, f)

        print("Vector store saved to file...✅✅✅")
        return vectorstore

    def load_vectorstore(self):
        """Load the FAISS vector store from file."""
        if os.path.exists(self.file_path):
            # Load the FAISS index using FAISS's read_index
            index = faiss.read_index(self.file_path)

            # Load docstore and index_to_docstore_id
            with open(self.file_path + "_docstore.pkl", "rb") as f:
                docstore = pickle.load(f)
            with open(self.file_path + "_index_to_docstore_id.pkl", "rb") as f:
                index_to_docstore_id = pickle.load(f)

            # Set embedding function (this is required for the FAISS object)
            embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)  # Ensure embedding function is set

            # Now, we create the FAISS object properly with index, embedding function, docstore, and index_to_docstore_id
            vectorstore = FAISS(index=index, embedding_function=embeddings, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
            print("Vector store loaded from file...✅✅✅")
            return vectorstore
        else:
            raise FileNotFoundError(f"Vector store file not found at {self.file_path}")

    def answer_query(self, query):
        """Answer the query using the FAISS vector store and ChatGPT."""
        vectorstore = self.load_vectorstore()
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=self.llm,  # Pass the correctly initialized OpenAI model
            retriever=vectorstore.as_retriever()
        )
        result = chain({"question": query}, return_only_outputs=True)
        return result["answer"], result.get("sources", "")
