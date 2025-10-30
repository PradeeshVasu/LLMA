import os
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS



# Set your OpenRouter API key and endpoint
# Step 1: Set LM Studio local endpoint
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"  # LM Studio local server
os.environ["OPENAI_API_KEY"] = "lm-studio"  # Any string works, LM Studio ignores it

# Load and process PDF
loader = PDFPlumberLoader("Basic_Home_Remedies.pdf") # share.csv
docs = loader.load()
print("Pages loaded:", len(docs))

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
documents = text_splitter.split_documents(docs)

# Vector DB
embedder = HuggingFaceEmbeddings()
vector = FAISS.from_documents(documents, embedder)
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 2})



# Step 2: Initialize the LLM (use the name of your loaded model)
llm = ChatOpenAI(
    model="tinyllama-1.1b-chat-v1.0",  # Must match the name shown in LM Studio
    temperature=0.7,
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    request_timeout=60,
    verbose=True
)

prompt = """
You are a domain expert assistant.
Use the provided context to answer the question clearly and accurately.
If the answer cannot be found in the context, say "The information is not available in the provided context."
Provide a well-structured answer in 3–4 sentences and keep it factual.

Context:
{context}

Question:
{question}

Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    callbacks=None,
)

qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
)

# Example query
result = qa("Home Remedies for Common Ailments")
print("Answer:", result["result"])