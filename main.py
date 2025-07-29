# main.py

import os
import warnings
import hashlib
import json
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
 
# 忽略来自 HuggingFace Transformers 库的特定 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.bert.modeling_bert")
 
# --- 配置常量 ---
PDF_FILE_PATH = 'PlantUML_Language_Reference_Guide_en.pdf'
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = "gemini-2.5-pro" # 注意: 请确保有权限访问此模型
 
def setup_embeddings():
    """初始化并返回嵌入模型"""
    print("正在初始化嵌入模型...")
    model_kwargs = {'device': 'cpu'}  # 如果有GPU，可以设置为 'cuda'
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("嵌入模型初始化完成。")
    return embeddings

def get_pdf_hash(file_path):
    """计算文件的 SHA256 哈希值"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # 以 4K 块读取并更新哈希值
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def load_or_create_vectorstore(pdf_path, persist_directory, embeddings):
    """
    加载向量数据库。如果数据库不存在或源PDF已更改，则创建新的数据库。
    """
    source_info_path = os.path.join(persist_directory, "source_info.json")

    # 检查源PDF文件是否存在
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件未找到: {pdf_path}")

    current_pdf_hash = get_pdf_hash(pdf_path)

    # 检查向量数据库是否存在且源文件未改变
    if os.path.exists(source_info_path):
        try:
            with open(source_info_path, 'r', encoding='utf-8') as f:
                source_info = json.load(f)
            if source_info.get('pdf_hash') == current_pdf_hash:
                print(f"从 '{persist_directory}' 加载已存在的向量数据库 (源文件未改变)...")
                vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                print("向量数据库加载完成。")
                return vectorstore
            else:
                print(f"源PDF文件 '{pdf_path}' 已改变。正在重建向量数据库...")
                shutil.rmtree(persist_directory) # 删除旧的数据库
        except (json.JSONDecodeError, IOError) as e:
            print(f"读取源信息文件时出错: {e}。将重建数据库。")
            shutil.rmtree(persist_directory)

    # 如果数据库不存在或源文件已改变，则创建新的
    print(f"未找到有效的向量数据库，正在从 '{pdf_path}' 创建新的数据库...")
    os.makedirs(persist_directory, exist_ok=True) # 确保目录存在

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"文档被分割成了 {len(docs)} 个片段。")

    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    with open(source_info_path, 'w', encoding='utf-8') as f:
        json.dump({'pdf_hash': current_pdf_hash, 'pdf_path': pdf_path}, f)
    print(f"文档已成功向量化并存入 '{persist_directory}'！")
    return vectorstore
 
def setup_llm():
    """初始化并返回大语言模型"""
    print("正在初始化大语言模型...")
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("请设置 GOOGLE_API_KEY 环境变量")
    
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.3)
    print("大语言模型初始化完成。")
    return llm

def create_qa_chain(llm, vectorstore):
    """基于LLM和向量数据库创建并返回问答链"""
    print("正在创建问答链...")
    # 创建检索器，设置为返回最相关的2个结果
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 创建问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" 模式会把所有检索到的文档塞进一个 Prompt
        retriever=retriever,
        return_source_documents=True  # 要求返回源文档，方便我们检查
    )
    print("问答链创建完成。")
    return qa_chain

def ask_question(qa_chain, query):
    """使用问答链进行提问并格式化打印结果"""
    print("\n--- 开始提问 ---")
    print(f"问题: {query}")
    result = qa_chain.invoke({"query": query})
    
    print(f"\nLLM的回答: {result['result']}\n")
    print("检索到的源文档片段:")
    for doc in result['source_documents']:
        print("---")
        # 打印源文档的元数据，如来源和页码，更具参考价值
        print(f"来源: {doc.metadata.get('source', 'N/A')}, 页码: {doc.metadata.get('page', 'N/A')}")
        print(doc.page_content)
    print("\n--- 提问结束 ---\n")

def main():
    """主执行函数，编排整个RAG流程"""
    embeddings = setup_embeddings()
    vectorstore = load_or_create_vectorstore(PDF_FILE_PATH, PERSIST_DIRECTORY, embeddings)
    llm = setup_llm()
    qa_chain = create_qa_chain(llm, vectorstore)
    
    query = "如何绘制 JSON 数据图？"
    ask_question(qa_chain, query)

if __name__ == "__main__":
    main()
