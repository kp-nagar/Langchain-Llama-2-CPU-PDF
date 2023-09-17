from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


MODEL_BIN_PATH='models/llama-2-7b-chat.ggmlv3.q8_0.bin'
MODEL_TYPE='llama'
MAX_NEW_TOKENS=256
TEMPERATURE=0.7
DATA_PATH='data/'
CHUNK_SIZE=500
CHUNK_OVERLAP=50
DB_FAISS_PATH='embed_vector'
VECTOR_COUNT=2
RETURN_SOURCE_DOCUMENTS=True


qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def build_llm():
    # Local CTransformers model
    llm = CTransformers(model=MODEL_BIN_PATH,
                        model_type=MODEL_TYPE,
                        config={'max_new_tokens': MAX_NEW_TOKENS,
                                'temperature': TEMPERATURE}
                        )

    return llm

def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt


def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': VECTOR_COUNT}),
                                       return_source_documents=RETURN_SOURCE_DOCUMENTS,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return dbqa


def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = build_llm()
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa


# Main function
def main():
    query = input("Enter your query: ")
    qa_result = setup_dbqa()
    response = qa_result({'query': query})
    print(response)

if __name__ == "__main__":
    main()
