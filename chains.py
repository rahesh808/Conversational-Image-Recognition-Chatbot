import os
import openai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

class Chain:
    def __init__(self):
        self.model_name = "gpt-3.5-turbo"  # Use gpt-4 if needed

    def extract_jobs(self, cleaned_text):
        # Define the prompt for extracting job details
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )

        # Construct the request to OpenAI API
        res = self._invoke_openai_api(prompt_extract, {"page_data": cleaned_text})

        try:
            # Parse the output to JSON format
            json_parser = JsonOutputParser()
            res = json_parser.parse(res['choices'][0]['message']['content'])
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        # Define the prompt for generating the cold email
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Mohan, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase AtliQ's portfolio: {link_list}
            Remember you are Mohan, BDE at AtliQ. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )

        # Construct the request to OpenAI API for generating the email
        res = self._invoke_openai_api(prompt_email, {"job_description": str(job), "link_list": links})
        return res['choices'][0]['message']['content']

    def _invoke_openai_api(self, prompt, inputs):
        # Construct the OpenAI API request using the correct chat endpoint
        # Use the chat-based model API
        messages = [
            {"role": "system", "content": "You are an assistant helping with job description extraction and email generation."},
            {"role": "user", "content": prompt.format(**inputs)}
        ]
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1500,  # Adjust token limit as per your requirement
            temperature=0.7  # Control randomness
        )
        return response

        

