import pinecone
import openai
import os
from dotenv import load_dotenv
from IPython.display import Markdown, display

def display_answer(answer):
    display(Markdown(answer))

def get_gpt4_answer(messages):
    response = openai.ChatCompletion.create(
        #model="gpt-4",
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response['choices'][0]['message']['content']

def get_augmented_query(query, index, embed_model):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )
    xq = res['data'][0]['embedding']
    res = index.query(xq, top_k=5, include_metadata=True)
    contexts = [item['metadata']['text'] for item in res['matches']]
    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query
    return augmented_query

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
embed_model = "text-embedding-ada-002"

# Initialize Pinecone connection
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-central1-gcp"
)

# Connect to Pinecone index
index_name = 'gpt-4-langchain-docs'
index = pinecone.Index(index_name)

# System message to prime the GPT-4 model
#primer = f"""You are Q&A bot. A highly intelligent system that answers user questions based on the information provided by the user above each question. If the information can not be found in the information provided by the user you truthfully say "I don't know"."""
primer = f"""You are Q&A bot. A highly intelligent system that answers user questions. You will first try to answer based on the information provided by the user above each question. If the information cannot be found in the information provided by the user then use your own pre-trained model."""
# Initialize conversation with the system message
conversation = [{"role": "system", "content": primer}]

print("You can now chat with the AI. Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Add user input to the conversation
    conversation.append({"role": "user", "content": user_input})

    # Get the augmented query
    augmented_query = get_augmented_query(user_input, index, embed_model)

    # Add augmented query to the conversation
    conversation.append({"role": "user", "content": augmented_query})

    # Get GPT-4's response
    gpt4_response = get_gpt4_answer(conversation)

    # Add GPT-4's response to the conversation
    conversation.append({"role": "assistant", "content": gpt4_response})

    # Display the response
    print("AI:", gpt4_response)