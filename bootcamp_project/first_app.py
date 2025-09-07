## import modules
import os
from dotenv import load_dotenv
import streamlit as st
#from langchain_community.llms import ollama


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
 
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

## PROMPT
prompt=ChatPromptTemplate.from_messages([
    ("system","you are a helmpful assistaant.please respond to the question asked"),
    ("user","{question}")
])

## streamlit frontend
st.title("Fist GenAI APP")
st.write("this is your very first chatbot built in bootcamp")

## create a input box
input_text=st.text_input("ask me anything")

#llm=llama(model="gemma3:1b")
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(
    model="gpt-4o",
    temperature=0.2

)
parser=StrOutputParser()

## built your chain(LCEL)
chain=prompt|llm|parser

## generate answer when usert enter question
if input_text:
    answer=chain.invoke({"question":input_text})
    st.write("**BOT SAYS**")
    st.write(answer)