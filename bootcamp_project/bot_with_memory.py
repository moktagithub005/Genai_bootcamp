##step 1 import modules 
import os
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
 
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

## streamlit frontend
st.title("🤖 CHATBOT WITH GROQ")
st.write("now with **memory**")


## chat box 
user_question=st.text_input("Ask me anything")

## memory

if "memory" not in st.session_state:
    st.session_state.memory=ConversationBufferMemory(return_messages=True)

## llm 
llm=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2

)

## PROMPT
prompt=ChatPromptTemplate.from_messages([
    ("system","you are a helmpful assistaant.please respond to the question asked"),
    MessagesPlaceholder(variable_name="history"),
    ("user","{question}")
])


parser=StrOutputParser()

## built your chain(LCEL)
chain=prompt|llm|parser


if user_question:
    history=st.session_state.memory.load_memory_variables({})["history"]
    answer=chain.invoke({"history":history,"question":user_question})

    st.session_state.memory.chat_memory.add_user_message(user_question)
    st.session_state.memory.chat_memory.add_ai_message(answer)

    st.write("🤖 **bot says**")
    st.write(answer)


    st.write("------------------------")
    st.write("conversation so far.....")

    for msg in st.session_state.memory.chat_memory.messages:
        st.write(f"**{msg.type.capitalize()}**:msg.content")

        ## streamlit run bot_with_memory.py