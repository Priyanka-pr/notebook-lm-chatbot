from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_community.chat_models import ChatOllama

load_dotenv()

os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")

##Creating chatbot

prompt =ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please provide response to the user queries"),
        ("user","Question:{question}")
    ]
)

#streamlit framework

# st.title("Langchain Demo with Open AI API")

st.title("Langchain Demo with mistral-nemo API")
input_text = st.text_input("Search the topic you want")

#Open AI LLM call
# 
llm = ChatOllama(
    model="mistral-nemo",  # or whatever model you're using
    base_url="http://172.16.6.101:11434",  # your Ollama base URL
    temperature=0.5
)
output_parser = StrOutputParser()

##Chain
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))