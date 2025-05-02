from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
mykey = os.getenv("GROQ_API_KEY")


chat = ChatGroq(groq_api_key=mykey, model_name="llama3-70b-8192")

st.title("Chat with any Website")

url_text = st.text_input("Enter a URL")
Prompt_text = st.text_input("Enter your prompt:")

submit_button = st.button("Submit")
if submit_button and url_text and Prompt_text:
    doc = WebBaseLoader(url_text)
    extractedText = doc.load()
    text = extractedText[0].page_content

    extract_prompt = PromptTemplate.from_template("""
    ---------------------
    The scrapped text is:
    {text}
    ---------------------

    Instruction:
    {Prompt_text}
    """)

    chain = extract_prompt | chat
    res = chain.invoke(input={'text': text[:1000], 'Prompt_text': Prompt_text})
    st.markdown(res.content)
