# main.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.fake import FakeListLLM

# # 1. 加载文档
# loader = PyPDFLoader('PlantUML_Language_Reference_Guide_en.pdf')
# documents = loader.load()

# # 2. 分割文档
# # RecursiveCharacterTextSplitter 会尝试按段落、句子等智能地分割文本
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = text_splitter.split_documents(documents)
# print(f"文档被分割成了 {len(docs)} 个片段。")

# # 3. 初始化嵌入模型
# # 我们选择一个社区中流行且表现良好的中英文模型
# # 首次运行时会自动下载模型，请耐心等待
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_kwargs = {'device': 'cpu'} # 如果有GPU，可以设置为 'cuda'
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# # 4. 将文档片段和它们的向量存入 ChromaDB
# # 这会在当前目录下创建一个 "chroma_db" 文件夹来持久化存储数据
# vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
# print("文档已成功向量化并存入数据库！")


# 5. 从持久化的数据库中加载
# 如果你已经运行过第三步，可以注释掉之前的代码，直接从这里开始
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 6. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # 设置为返回最相关的2个结果

# 7. 创建一个假的 LLM，它的回答列表里只有一个元素
# 这能让我们看到 RAG 到底向 LLM 提供了什么信息
responses = ["我是 PlantUML 助手，我会根据你提供的上下文回答问题。"]
llm = FakeListLLM(responses=responses)

# 8. 创建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "stuff" 模式会把所有检索到的文档塞进一个 Prompt
    retriever=retriever,
    return_source_documents=True # 要求返回源文档，方便我们检查
)

# 9. 开始提问！
query = "如何绘制 JSON 数据图？"
result = qa_chain({"query": query})

print(f"提问: {result['query']}\n")
print(f"LLM的模拟回答: {result['result']}\n")
print(f"检索到的源文档片段:")
for doc in result['source_documents']:
    print("---")
    print(doc.page_content)


if __name__ == '__main__':
    pass
    