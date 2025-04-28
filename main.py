from  langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate 
import streamlit as st

from dotenv import load_dotenv
import os
load_dotenv()

mykey=os.getenv("GROQ_API_KEY")



chat = ChatGroq(groq_api_key=mykey  , model_name="llama-3.3-70b-versatile")
st.title("Chat with any Website")

url_text=st.text_input("Enter a URL ")
Prompt_text=st.text_input("Enter your massage")


submit_button=st.button("Submit")
if submit_button: 


    #doc = WebBaseLoader()

    doc = WebBaseLoader(url_text)

    extractedText=doc.load()
    text = extractedText[0].page_content

    extract_prompt=PromptTemplate.from_template("""
    --------------
    The scrapped text is:         
    {text}
    --------------   

    Instruction:
    {Prompt_text}
    --------------                            


    """ )

    Chain=extract_prompt | chat
    res=Chain.invoke(input={'text' : text[0:1000],"Prompt_text":Prompt_text})
    #print(res.content)

    st.markdown(res.content)


