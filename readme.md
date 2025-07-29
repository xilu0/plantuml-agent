$env:GOOGLE_API_KEY=""

pip freeze > requirements.txt
pip install -U langchain-chroma


 ls ~\.cache\huggingface\hub\
    Directory: C:\Users\heish\.cache\huggingface\hub
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----         7/29/2025   9:56 AM                .locks
d-----         6/23/2024  10:55 PM                models--distilbert--distilbert-base-uncased-finetuned-sst-2-english
d-----         2/15/2025   2:02 PM                models--Qwen--Qwen2.5-VL-3B-Instruct
d-----         2/15/2025   1:56 PM                models--Qwen--Qwen2.5-VL-7B-Instruct
d-----         7/29/2025   9:56 AM                models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2
-a----         6/23/2024  10:54 PM              1 version.txt