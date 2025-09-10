import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

load_dotenv()
## langsmith tracking 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "U_Tools"


## LLM setup using groq 
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

##prompt 
prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a professional translator. translate the following text to{language}"),
    ("user", "{text}")
])

## output parser 
parser = StrOutputParser()

## LCEL(langchain expression language)
chain = prompt | llm | parser

## test function 
def test_translation():
    result = chain.invoke(
        {"language": "hindi", "text": "how are you today"}
    )
    print(f"result Translation:{result}")

## create fastapi app 
app = FastAPI(
    title="Translation API",
    version="1.0.0",
    description="simple translation service with langsmith tracking"
)

#add langseerve endpoints
add_routes(
    app,
    chain,
    path="/translate",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=False,
    playground_type="default"

)

##health check point
@app.get("/")
async def health_check():
    return {"status": "Translation API is running", "endpoint": "/translate"}

## main execution block 
if __name__ == "__main__":
    test_translation()
    print("API will be available at: http://127.0.0.1:8000")
    print("translation endpoint: http://127.0.0.1:8000/translate")
    uvicorn.run(app, host="127.0.0.1", port=8000)
