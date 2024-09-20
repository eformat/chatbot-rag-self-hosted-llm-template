import os
import re
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
import gradio as gr
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import PGVector

# Check these classes for other options
# from langchain_community.llms import VLLMOpenAI - vllm
# from langchain_community.llms import OpenAI - /completions

load_dotenv()

# Parameters

APP_TITLE = 'Talk with your documentation'

MODEL_NAME = os.getenv('MODEL_NAME', "mistralai/Mistral-7B-Instruct-v0.2")
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 32768))
PRESENCE_PENALTY=float(os.getenv('PRESENCE_PENALTY', 1.03))

INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
TOP_K = int(os.getenv('TOP_K', 10))
TOP_P = float(os.getenv('TOP_P', 0.95))
TYPICAL_P = float(os.getenv('TYPICAL_P', 0.95))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))

DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')
DB_COLLECTION_NAME = os.getenv('DB_COLLECTION_NAME')

# Streaming implementation
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()

def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata['source'] not in unique_list:
            unique_list.append(item.metadata['source'])
    return unique_list

def stream(input_text) -> Generator:
    # Create a Queue
    job_done = object()

    # Create a function to call - this will run in a thread
    def task():
        resp = qa_chain.invoke({"query": input_text})
        sources = remove_source_duplicates(resp['source_documents'])
        if len(sources) != 0:
            q.put("\n*Sources:* \n")
            for source in sources:
                q.put("* " + str(source) + "\n")
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue

# A Queue is needed for Streaming implementation
q = Queue()

############################
# LLM chain implementation #
############################

# Document store: pgvector vector store
embeddings = HuggingFaceEmbeddings()
store = PGVector(
    connection_string=DB_CONNECTION_STRING,
    collection_name=DB_COLLECTION_NAME,
    embedding_function=embeddings,
    use_jsonb=True)

template="Q: {question} A:"

if re.search(r'Mistral', MODEL_NAME, flags=re.IGNORECASE):
    template="""<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant answering questions named HatBot.
    You will be given a question you need to answer, and a context to provide you with information. You must answer the question based as much as possible on this context.
    Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>
    
    Context:
    {context}
    
    Question: {question} [/INST]
    """

if re.search(r'LLama-2', MODEL_NAME, flags=re.IGNORECASE):
    template="""<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant answering questions named HatBot.
    You will be given a question you need to answer, and a context to provide you with information. You must answer the question based as much as possible on this context.
    Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    Context:
    {context}
    
    Question: {question} [/INST]
    """

if re.search(r'LLama-3', MODEL_NAME, flags=re.IGNORECASE):
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    You are a helpful, respectful and honest assistant answering questions named HatBot.
    You will be given a question you need to answer, and a context to provide you with information. You must answer the question based as much as possible on this context.
    Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Context:
    {context}
    
    Question: {question}<|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    """

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

import httpx

llm = ChatOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=INFERENCE_SERVER_URL,
    model_name=MODEL_NAME,
    top_p=TOP_P,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    presence_penalty=PRESENCE_PENALTY,
    streaming=True,
    verbose=False,
    callbacks=[QueueCallback(q)],
    async_client=httpx.AsyncClient(verify=False),
    http_client=httpx.Client(verify=False)
)

qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
            ),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
        )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Gradio implementation
def ask_llm(message, history):
    for next_token, content in stream(message):
        yield(content)

CSS ="""
footer {visibility: hidden}
"""

with gr.Blocks(title="RHOAI HatBot", css=CSS, fill_height=True) as demo:
    chatbot = gr.Chatbot(
        show_label=False,
        avatar_images=(None, 'https://avatars.githubusercontent.com/u/65787031?v=4'),
        render=True,
        likeable=False,
        height=800,
        )
    gr.ChatInterface(
        ask_llm,
        chatbot=chatbot,
        clear_btn=None,
        retry_btn=None,
        undo_btn=None,
        stop_btn=None,
        description=APP_TITLE
        )

if __name__ == "__main__":
    demo.queue().launch(
        server_name='0.0.0.0',
        share=False,
        favicon_path='./assets/robot-head.ico',
#        ssl_keyfile="key.pem", ssl_certfile="cert.pem",
#        ssl_verify=False
        )
