import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize search tool and LLM
search_engine = DuckDuckGoSearchRun()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")



prompt = ChatPromptTemplate.from_template ("""
        Summarize the content given and try to keep all the facts that are told in it.
        Try to keep it brief and knowledgable.
        contnet : {result}
""")

def reply(text):
    main = prompt.invoke({'result': text})
    response = llm.invoke(main)
    back = response.content
    return back 

st.title("Web chatbot")
query = st.text_input("Enter your query here:")
send_button = st.button("Send")
if send_button:
    if query:
        result = search_engine.invoke(query)
        st.write(reply(result))
    else:
        st.write("Please enter a query")