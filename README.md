LangChain Docs Scraper & Search Engine
This repository contains a Python script that scrapes the LangChain documentation, processes the text, and creates an index using OpenAI's text-embedding-ada-002 and Pinecone's vector search engine. The main purpose of this script is to help users search through the LangChain documentation and generate responses using GPT-4.

Features
Scrapes LangChain documentation
Processes the text into chunks
Generates embeddings using OpenAI's text-embedding-ada-002
Creates a Pinecone index to store embeddings
Enables vector search through indexed documents
Generates responses using GPT-4
Dependencies
requests
beautifulsoup4
tiktoken
langchain.text_splitter
openai
python-dotenv
pinecone
tqdm
Getting Started
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/langchain-docs-scraper.git
Navigate to the project directory:
bash
Copy code
cd langchain-docs-scraper
Install the required dependencies:
Copy code
pip install -r requirements.txt
Set up your environment variables:
Create a .env file in the project directory and include your OpenAI API key and Pinecone API key:

makefile
Copy code
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
Run the script:
css
Copy code
python main.py
Usage
Once you've set up the project, you can perform searches and generate responses with GPT-4 using the following steps:

Import the required libraries and initialize the OpenAI and Pinecone connections:
python
Copy code
import pinecone
import openai
import os
from dotenv import load_dotenv

load_dotenv()
# environment variables

# Initialize openai API key
openai.api_key = os.getenv("OPENAI_API_KEY")

embed_model = "text-embedding-ada-002"

index_name = 'gpt-4-langchain-docs'

# Initialize connection to pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-central1-gcp"
)

# Connect to index
index = pinecone.Index(index_name)
Retrieve relevant documents using a query:
python
Copy code
query = "how do I use LLMChain in Langchain?"

res = openai.Embedding.create(
    input=[query],
    engine=embed_model
)

xq = res['data'][0]['embedding']

res = index.query(xq, top_k=5, include_metadata=True)
Generate a response using GPT-4:
python
Copy code
contexts = [item['metadata']['text'] for item in res['matches']]

augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

primer = f"""You are Q&A bot. A highly intelligent system that answers user questions based on the information provided by the user above each question. If the information can not be found in the information provided by the user you truthfully say "I don't know"."""

res = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)
Display the response:
python
Copy code
from IPython.display import Markdown

display(Markdown(res['choices'][0]['message']['content']))
