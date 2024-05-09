from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatOpenAI, ChatHuggingFace,ChatZhipuAI
from langchain_wenxin.chat_models import ChatWenxin
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import VectorDBQA
# pip install Chromadb  持久化向量数据库
# from langchain.document_loaders import DirectoryLoader, TextFileLoader
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

def create_embeddings():
    #============text================
    loader = TextLoader('demo_quick_start\Intro_ESTUN.txt', encoding='utf-8')
    # documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
    documents = loader.load_and_split(text_splitter=text_splitter)
    # print(documents)
    embeddings = HuggingFaceEmbeddings()
    docsearch = DocArrayInMemorySearch.from_documents(documents, embeddings)
    return docsearch
    #==========directory==================

    # loader = DirectoryLoader('demo_quick_start/rawdata', glob='*.txt')
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
    # texts = text_splitter.split_documents(documents)
    # print(texts)
    # embeddings = HuggingFaceEmbeddings()
    # docsearch = DocArrayInMemorySearch.from_documents(texts, embeddings)
    # return docsearch

def run_query(vectorstore, query):
    llm = ChatZhipuAI(
            model="glm-4",
            temperature=0.5,
        )
    qa_stuff = VectorDBQA.from_chain_type(llm=llm, chain_type="map_reduce", vectorstore=vectorstore, return_source_documents=False)
    response = qa_stuff({"query": query})
    return response

def main():
    _ = load_dotenv(find_dotenv())

    retriever = create_embeddings()
    
    query = "In what year was Estun established?(translate to chinese)"
    # query = "How much change did Goku learn?(translate to chinese),summerize at 20words"
    response = run_query( retriever, query)
    print("response:!!!!!!!!!!!!!!!!!!!!!!!!!",response)

if __name__ == '__main__':
    main()