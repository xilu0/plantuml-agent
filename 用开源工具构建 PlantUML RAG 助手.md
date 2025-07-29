# 用开源工具构建一个智能 PlantUML RAG 助手

本项目旨在演示如何利用开源工具和最新的大语言模型（LLM）技术，构建一个针对特定领域知识的智能问答助手。我们以 PlantUML 的官方参考手册（PDF格式）为知识源，实现了一个能够理解并回答关于 PlantUML 语法和用法问题的 RAG (Retrieval-Augmented Generation) 系统。

**核心技术栈:**
*   **LLM:** Google Gemini Pro (`gemini-1.5-pro`)
*   **编排框架:** LangChain
*   **嵌入模型 (Embeddings):** HuggingFace (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
*   **向量数据库:** ChromaDB

---

## 🌟 项目亮点

`main.py` 脚本的设计包含了许多优秀的工程实践，使其不仅是一个简单的原型，更是一个健壮、高效且易于维护的应用。

1.  **智能数据持久化与增量更新**
    *   **痛点:** 每次启动应用时都重新解析和向量化整个 PDF 文档会耗费大量时间和计算资源。
    *   **解决方案:** 脚本通过 `get_pdf_hash` 函数计算源 PDF 文件的 `SHA256` 哈希值。首次运行时，它会将文档向量化并存入本地 ChromaDB，同时保存该 PDF 的哈希值。后续启动时，它会重新计算 PDF 的哈希并与已保存的哈希进行比对。**只有当 PDF 文件发生变化时**，才会触发数据库的重建。这极大地提升了应用的启动速度和效率。

2.  **清晰的模块化设计**
    *   代码被精心组织成一系列独立的函数，每个函数都承担单一的职责（如 `setup_embeddings`, `load_or_create_vectorstore`, `create_qa_chain` 等）。
    *   这种设计使得代码逻辑清晰，易于理解、测试和扩展。例如，如果想替换 LLM 或向量数据库，只需修改对应的函数即可，而不会影响其他部分。

3.  **高度可配置性**
    *   所有关键参数，如 PDF 文件路径、持久化目录、嵌入模型名称和 LLM 模型名称，都在文件顶部的常量区进行定义。
    *   这使得用户可以轻松地根据自己的环境和需求调整配置，而无需深入代码逻辑。

4.  **答案可溯源，结果更可信**
    *   在RAG应用中，仅仅给出答案是不够的，用户往往希望知道答案的来源。
    *   本脚本在创建 `RetrievalQA` 链时，通过设置 `return_source_documents=True`，确保在返回 LLM 生成的答案的同时，也一并返回用于生成该答案的原始文档片段（Source Documents）。
    *   输出结果会清晰地展示每个来源片段的出处（文件名和页码），大大增强了答案的可信度和可验证性。

5.  **健壮的错误处理**
    *   脚本包含了必要的检查，如启动时确认 `GOOGLE_API_KEY` 环境变量是否存在，以及源 PDF 文件是否存在，并在缺失时给出明确的错误提示。

---

## ⚙️ 工作流程

应用遵循一个标准的 RAG 流程，由 `main()` 函数统一编排：

1.  **初始化嵌入模型 (`setup_embeddings`)**:
    *   加载 HuggingFace 的 `paraphrase-multilingual-MiniLM-L12-v2` 模型。该模型负责将文本块转换为高维向量，以便进行相似度计算。

2.  **加载或创建向量数据库 (`load_or_create_vectorstore`)**:
    *   **检查:** 计算当前 `PlantUML_Language_Reference_Guide_en.pdf` 文件的哈希值。
    *   **比对:** 与 `./chroma_db/source_info.json` 中存储的旧哈希值进行比较。
    *   **决策:**
        *   如果哈希值相同，直接从 `./chroma_db` 目录加载已存在的向量数据库。
        *   如果哈希值不同或数据库不存在，则执行以下“创建”流程。
    *   **创建:**
        *   使用 `PyPDFLoader` 加载 PDF 文档。
        *   使用 `RecursiveCharacterTextSplitter` 将文档分割成大小适中（`chunk_size=500`）且有重叠（`chunk_overlap=50`）的文本块。
        *   使用第一步的嵌入模型将所有文本块转换为向量。
        *   将文本块和对应的向量存入 ChromaDB，并指定持久化目录。
        *   将新的 PDF 哈希值写入 `source_info.json` 文件。

3.  **初始化大语言模型 (`setup_llm`)**:
    *   检查 `GOOGLE_API_KEY` 是否设置。
    *   初始化 `ChatGoogleGenerativeAI` 模型，并设置 `temperature=0.3` 以获得更稳定、更具确定性的回答。

4.  **创建问答链 (`create_qa_chain`)**:
    *   从向量数据库创建一个检索器（Retriever），并配置它在每次查询时返回最相关的 `k=2` 个文档片段。
    *   使用 `RetrievalQA.from_chain_type` 创建一个问答链。
        *   `chain_type="stuff"`: 这种模式会将检索到的所有文档片段“塞入”到同一个 Prompt 中，一次性提交给 LLM。
        *   `return_source_documents=True`: 要求链返回其检索到的源文档。

5.  **执行问答 (`ask_question`)**:
    *   接收用户问题（Query）。
    *   调用问答链的 `invoke` 方法执行 RAG 流程。
    *   格式化并打印 LLM 的最终回答以及作为参考依据的源文档片段（包括来源和页码）。

---

## 🚀 如何运行

1.  **环境准备**
    *   确保您已安装 Python 3.8+。
    *   克隆项目并进入项目目录。
    *   安装所需的依赖包。建议创建一个虚拟环境：
        ```bash
        pip install langchain langchain-community langchain-chroma langchain-google-genai langchain-huggingface pypdf sentence-transformers transformers
        ```
    *   下载 PlantUML 语言参考指南，并将其命名为 `PlantUML_Language_Reference_Guide_en.pdf`，放置在项目根目录。
    *   获取您的 Google AI API 密钥，并将其设置为环境变量。
        *   **Linux/macOS:**
            ```bash
            export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
            ```
        *   **Windows (CMD):**
            ```bash
            set GOOGLE_API_KEY="YOUR_API_KEY_HERE"
            ```

2.  **执行脚本**
    *   在终端中运行主脚本：
        ```bash
        python main.py
        ```
    *   首次运行时，您会看到脚本正在创建新的向量数据库。后续运行将会直接加载现有数据库，速度会快很多。

3.  **自定义问题**
    *   打开 `main.py` 文件，修改 `main` 函数中的 `query` 变量为你自己的问题，然后重新运行脚本。
      ```python
      # in main() function
      query = "如何定义一个组件？" # <-- 修改这里的问题
      ask_question(qa_chain, query)
      ```

---

## 🔮 未来展望

这个项目为构建更复杂的 RAG 应用奠定了坚实的基础。未来可以从以下几个方面进行扩展：

*   **Web 界面:** 使用 Streamlit 或 Flask/FastAPI 为其创建一个用户友好的 Web 界面。
*   **支持更多文档格式:** 扩展加载器以支持如 `.docx`, `.md`, `.html` 等更多格式的文档。
*   **对话记忆:** 集成对话历史记录功能，使助手能够理解上下文，进行多轮对话。
*   **优化检索策略:** 尝试更高级的检索技术，如 Parent Document Retriever 或 HyDE (Hypothetical Document Embeddings)。