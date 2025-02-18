import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
import gradio as gr

os.environ['OPENAI_API_KEY'] = "your-api-key"


loader = PyPDFLoader("/Users/ravindargowrigari/Documents/GenAI Training/LLM App HR Assistant Project/1728286846_the_nestle_hr_policy_pdf_2012.pdf")
# loader = PyPDFLoader("/Users/ravindargowrigari/Downloads/4q24-fixed-income-presentation.pdf")
documents = loader.load()         

# Split text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(documents)

# Create embeddings and set up the RAG system
embeddings = OpenAIEmbeddings()
# vector_store = Chroma.from_documents(text_chunks, embedding=embeddings)
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
retriever = vector_store.as_retriever(search_type = 'similarity')
  
# Initialize GPT-3.5 Turbo model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create the Retrieval-based QA system
retriever = vector_store.as_retriever()
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template= """ 
    
    You are an document assistant AI agent specialized in analyzing the documents and providing information from the documents. 
    If the user says positive words like "Thank you" or "Good" or "You are the best" you reply as something like "Welcome", "Thank you!".
    If the user says negative words like "You are not great" or "bad" or "You dont have intelligence" you reply as something like "Thenk you for feedback. I am AI agent, still learing to provide intelligence. ".
    If you did not find answer to question, you should mentioned "I do not have this information, please ask other question related the document in scope."
    
    Answer the following question based on the provided context. 
    
    Context: {context} 
    Question: {question}\n\nAnswer:
    
    """
    
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": prompt_template,
        "verbose": False },
    return_source_documents=False,
)


# Define the chatbot function
def chatbot(query):
    response = qa_chain.run(query)
    return response

# Create Gradio interface
interface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="Enter your question to get information from the documents:"),
    outputs=gr.Textbox(label="Answer"),
    title="Document Intelligence Assistant",
    description="Ask any question related to the documents",
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(share=True)
