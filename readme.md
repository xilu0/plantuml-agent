$env:GOOGLE_API_KEY=""

pip freeze > requirements.txt
pip install -U langchain-chroma

```powershell
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
```

## 运行结果示例：
--- 开始提问 ---
问题: 如何绘制 JSON 数据图？

LLM的回答: 根据提供的上下文，您可以使用 PlantUML 中的 `json` 关键字来绘制或显示 JSON 数据。

语法如下：
1.  以 `json` 关键字开头。
2.  给您的 JSON 数据起一个名字（例如 `J`）。
3.  将您的 JSON 内容放在大括号 `{}` 内。

这是一个具体的示例代码：
```plantuml
json J {
"fruit":"Apple",
"size":"Large",
"color": ["Red", "Green"]
}
@enduml
```
这段代码会生成一个可视化的图表来展示这个 JSON 对象的结构和内容。

检索到的源文档片段:
```
---
来源: PlantUML_Language_Reference_Guide_en.pdf, 页码: 272
11.13 Display JSON Data on State diagram 11 DISPLAY JSON DATA
json J {
"fruit":"Apple",
"size":"Large",
"color": ["Red", "Green"]
"size":"Large",
"color": ["Red", "Green"]
}
@enduml
[Ref. QA-17275]
}
@enduml
[Ref. QA-17275]
PlantUML Language Reference Guide (1.2023.11) 272 /550
---
来源: PlantUML_Language_Reference_Guide_en.pdf, 页码: 272
11.13 Display JSON Data on State diagram 11 DISPLAY JSON DATA
json J {
"fruit":"Apple",
"size":"Large",
"color": ["Red", "Green"]
}
@enduml
[Ref. QA-17275]
PlantUML Language Reference Guide (1.2023.11) 272 /550
```
--- 提问结束 ---
