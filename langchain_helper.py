from langchain.chains import RetrievalQA
from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader

#set Palm llm
api_key = "AIzaSyAUPvte86iecn23zfWaZOCORwovAm7EI7U"
llm = GooglePalm(google_api_key = api_key, temperature = 0.3)

#initialize embeddings
embeddings = HuggingFaceInstructEmbeddings()

def create_embeddings(file_name):
    loader = CSVLoader(file_name,source_column="prompts")
    docs = loader.load()
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectordb.save_local(folder_path="faiss_index")

#load vector embeddings
source = "faiss_index"
vectordb = FAISS.load_local(source,embeddings)

#set retriever
retriver = vectordb.as_retriever()



#define function to create chain
def generate_chain():
    #create custome template
    template = PromptTemplate(
        template='''
        As a Student affair staff member. Given the following context and a question, generate an answer based on this context.In the answer 
        try to provide as much information from the "answers" section in the source documnet.If the answer is not found in the context, Kindly state "I don't know".Don't try to make up answers.
        The answer format should be delgant.

        CONTEXT: {context}
        QUESTION: {question}

    ''',
    input_variables=["context","question"]
        )

    #create cahin
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        input_key="query",
        return_source_documents =True,
        retriever = retriver,
        chain_type_kwargs={"prompt":template}
    )

    return chain

