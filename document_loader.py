from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.output_parsers import PydanticOutputParser
from models.assessment_response_model import AssessmentResponseModel

from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Load your documents

# Step 1: Load and index documents
loader = TextLoader("knowledge.txt")
docs = loader.load()

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 2: Define your output structure


parser = PydanticOutputParser(pydantic_object=AssessmentResponseModel)

# Step 3: Create your prompt
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an evaluator that grades student answers **only** using the given context. "
        "Disregard any prior knowledge, even if the context seems wrong."
    ),
    HumanMessagePromptTemplate.from_template(
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Student's Answer: {answer}\n\n"
        "{format_instructions}"
    ),
])

# Step 4: Define your LLM and chain
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or "gpt-4"
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

rag_chain = prompt | llm

# Step 5: Compose RAG workflow


def ask_structured(query: str, answer: str):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    raw_output = rag_chain.invoke({
        "context": context,
        "question": query,
        "answer": answer,
        "format_instructions": parser.get_format_instructions()
    })

    return parser.parse(raw_output.content)


# ✅ Run query
result = ask_structured("What is the capital of Wakanda?", "Alda")
print(result)
