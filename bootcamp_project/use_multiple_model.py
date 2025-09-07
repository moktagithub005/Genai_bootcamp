
import os
import streamlit as st
import base64
from io import BytesIO
from PIL import Image

# LLM imports
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage


# Page Configuration

st.set_page_config(
    page_title="Q&A Chatbot", 
    page_icon="🤖", 
    layout="wide"
)

st.title("🤖 Q&A Chatbot with Memory")
st.write("Chat with AI using Groq, OpenAI, or Ollama • Image processing supported")


# Sidebar - LLM Selection & Settings

with st.sidebar:
    st.header("⚙️ Settings")
    
    # LLM Provider Selection
    llm_provider = st.selectbox(
        "Choose LLM Provider:",
        ["Groq", "OpenAI", "Ollama"]
    )
    
    # API Key inputs based on provider
    api_key = None
    if llm_provider == "Groq":
        api_key = st.text_input("Groq API Key:", type="password", key="groq_key")
        model_name = st.selectbox("Model:", ["llama-3.1-8b-instant", "mixtral-8x7b-32768"])
    elif llm_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key:", type="password", key="openai_key")
        model_name = st.selectbox("Model:", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
    else:  # Ollama
        st.info("Make sure Ollama is running locally")
        model_name = st.text_input("Model Name:", value="llama2", key="ollama_model")
    
    # Temperature setting
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.2, 0.1)
    
    # Clear chat button
    if st.button(" 🗑️Clear Chat"):
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
        st.rerun()


# Initialize Memory

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)


# Initialize LLM

def get_llm(provider, api_key, model_name, temperature):
    try:
        if provider == "Groq":
            if not api_key:
                st.error("Please enter your Groq API key")
                return None
            os.environ["GROQ_API_KEY"] = api_key
            return ChatGroq(model=model_name, temperature=temperature)
        
        elif provider == "OpenAI":
            if not api_key:
                st.error("Please enter your OpenAI API key")
                return None
            return ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature)
        
        else:  # Ollama
            return Ollama(model=model_name, temperature=temperature)
            
    except Exception as e:
        st.error(f"Error initializing {provider}: {str(e)}")
        return None


# Image Processing Function

def encode_image(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def process_image_with_llm(llm, image, question="What do you see in this image?"):
    """Process image with LLM (works with OpenAI GPT-4V)"""
    try:
        if isinstance(llm, ChatOpenAI) and "gpt-4" in llm.model_name:
            # For OpenAI GPT-4V
            base64_image = encode_image(image)
            message = HumanMessage(
                content=[
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            )
            response = llm([message])
            return response.content
        else:
            return "Image processing is currently only supported with OpenAI GPT-4 models."
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Main Chat Interface


# Two columns for input
col1, col2 = st.columns([3, 1])

with col1:
    user_question = st.text_input("💬 Ask me anything...", key="question_input")

with col2:
    uploaded_file = st.file_uploader("📷 Upload Image", type=['png', 'jpg', 'jpeg'])

# Process Input

if user_question or uploaded_file:
    # Get LLM instance
    llm = get_llm(llm_provider, api_key, model_name, temperature)
    
    if llm is None:
        st.stop()
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer clearly and concisely."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{question}")
    ])
    
    # Create chain
    parser = StrOutputParser()
    chain = prompt | llm | parser
    
    # Process the input
    if uploaded_file:
        # Image processing
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        # Combine text question with image
        if user_question:
            image_question = f"{user_question} (analyzing the uploaded image)"
        else:
            image_question = "What do you see in this image? Describe it in detail."
        
        # Process image
        if isinstance(llm, ChatOpenAI) and "gpt-4" in model_name:
            answer = process_image_with_llm(llm, image, image_question)
        else:
            answer = "Image analysis requires OpenAI GPT-4. Processing text only..."
            if user_question:
                history = st.session_state.memory.load_memory_variables({})["history"]
                answer = chain.invoke({"history": history, "question": user_question})
        
        # Save to memory
        st.session_state.memory.chat_memory.add_user_message(image_question)
        st.session_state.memory.chat_memory.add_ai_message(answer)
        
    else:
        # Text-only processing
        history = st.session_state.memory.load_memory_variables({})["history"]
        answer = chain.invoke({"history": history, "question": user_question})
        
        # Save to memory
        st.session_state.memory.chat_memory.add_user_message(user_question)
        st.session_state.memory.chat_memory.add_ai_message(answer)
    
    # Display response
    st.write("🤖 **Bot Response:**")
    st.write(answer)


# Display Conversation History

st.write("---")
st.write("📝 **Conversation History:**")

if st.session_state.memory.chat_memory.messages:
    for i, msg in enumerate(st.session_state.memory.chat_memory.messages):
        if msg.type == "human":
            with st.chat_message("user"):
                st.write(msg.content)
        else:
            with st.chat_message("assistant"):
                st.write(msg.content)
else:
    st.write("No conversation yet. Start chatting!")

# Footer

st.write("---")
st.write("💡 **Tips:**")
st.write("• Upload images for visual analysis (Requires: gpt-4-turbo, gpt-4o, or gpt-4o-mini)")
st.write("• Use the sidebar to switch between different AI providers")
st.write("• Clear chat to start fresh conversations")