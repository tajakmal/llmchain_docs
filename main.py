import pinecone
import openai
import os
from dotenv import load_dotenv
from IPython.display import Markdown, display
import tiktoken

def display_answer(answer):
    display(Markdown(answer))

tokenizer = tiktoken.get_encoding('p50k_base')
def count_tokens(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def truncate_conversation(conversation, max_tokens=4096):
    total_tokens = 0
    truncated_conversation = []

    for message in reversed(conversation):
        message_tokens = count_tokens(message['content'])
        total_tokens += message_tokens

        if total_tokens < max_tokens:
            truncated_conversation.insert(0, message)
        else:
            break

    return truncated_conversation

def get_gpt4_answer(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        #model="gpt-4",
        temperature=0.7,
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

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
embed_model = "text-embedding-ada-002"

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-central1-gcp"
)

index_name = 'memory'
index = pinecone.Index(index_name)

primer = f"""You are Q&A bot. A highly intelligent system that answers user questions. You will first try to answer based on the information provided by the user above each question. If the information cannot be found in the information provided by the user then use your own pre-trained model."""
#primer = f"""You are Q&A bot. A highly intelligent system that answers user questions."""
conversation = [{"role": "system", "content": primer}]

print("You can now chat with the AI. Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    conversation.append({"role": "user", "content": user_input})

    augmented_query = get_augmented_query(user_input, index, embed_model)
    conversation.append({"role": "user", "content": augmented_query})

    # Print the context being sent to the model
    #print("Context being sent to the model:")
    #print(augmented_query)
    #print(conversation)

    truncated_conversation = truncate_conversation(conversation)
    gpt4_response = get_gpt4_answer(truncated_conversation)

    conversation.append({"role": "assistant", "content": gpt4_response})

    print("AI:", gpt4_response,"\n")