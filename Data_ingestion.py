import logging 


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


from logs import log_to_termianal


# creating data reading path and vectore store paths 
DATA_PATH = 'knowledge_base/'
DB_FAISS_PATH = 'VectorStores/db_faiss'
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")



@log_to_termianal
def vectore_db():

    logging.info("Loading the Data From Knowledge Base")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()

    logging.info("Starting the Splitting of the characters ")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    texts = text_splitter.split_documents(docs)
    logging.info(f"loaded the data and splitted it! size of the splitted text is {len(texts)}")


    logging.info("Calling the Sentence Transformer model")
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformer/all-MiniLM-L6-v2', model_kwargs = {'device':'cpu'}) # using cpu due to device constarints

    logging.info("Writing the Embedding to FAISS")
    db = FAISS.from_documents(texts,embeddings)
    db.save_local(DB_FAISS_PATH)


if __name__ == '__main__':
    vectore_db()