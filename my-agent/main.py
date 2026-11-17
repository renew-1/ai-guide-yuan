import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ================= 1. 配置 DeepSeek =================
os.environ["OPENAI_API_KEY"] = "sk-c35333a2046f4ea386220d7818bfac5e" # 你的密钥
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

# ================= 2. 加载并切分文档 =================
print("正在读取文档...")
loader = TextLoader("my_notes.txt", encoding="utf-8")
documents = loader.load()

# 【新改动】给文档片段打上编号标签
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=150, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
for i, doc in enumerate(docs):
    doc.metadata["source_id"] = i + 1 # 给每个片段打上 1, 2, 3... 的标签
print(f"文档切分完成，共切分为 {len(docs)} 个片段，并已打上编号标签")

# ================= 3. 本地向量化 =================
print("正在加载本地向量模型...")
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
db = Chroma.from_documents(docs, embedding, persist_directory="./chroma_db")
print("向量库构建完成！\n")

# ================= 4. 构建问答链（加入引用要求的 Prompt） =================
llm = ChatOpenAI(model="deepseek-chat", temperature=0)

# 【新改动】要求大模型必须用 [数字] 的形式标注引用
prompt_template = """你是一个严谨的 AI 助手。请严格根据以下提供的上下文来回答问题。
如果你在回答中使用了某段上下文的信息，请在该句话的末尾加上对应的片段编号，格式为 [1] 或 [2]。注意必须使用英文半角的数字和方括号。
如果不知道答案，就说不知道，不要编造。

上下文：
{context}

问题：{question}

请给出带有引用编号的回答："""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# ================= 5. 提问测试 =================
question = "RAG的全称是什么？它有什么好处？"
print(f"正在思考：{question} \n")

result = qa_chain.invoke({"query": question})

print("================ 最终回答 ================")
print(result['result'])
print("==========================================\n")

# 打印引用出处
print("📚 参考文献出处：")
# ================= 提取引用出处（修复连体标号问题） =================
import re
cited_ids = set()

# 提取所有数字，无视是否有括号，例如把 [1][2] 变成 1 和 2
# 匹配格式为 [数字] 或 [数字][数字]
matches = re.findall(r'\[(\d+)\]', result['result'])
for match in matches:
    # 有时候大模型会把 [1][2] 写在一起，这个匹配能把里面的数字都找出来
    for num in re.findall(r'\d+', match):
        cited_ids.add(num)

# 兼容处理中文数字（防止大模型抽风）
chinese_nums = {'一':'1', '二':'2', '三':'3', '四':'4', '五':'5', '六':'6', '七':'7', '八':'8', '九':'9', '十':'10'}
for cn in re.findall(r'\[([一二三四五六七八九十]+)\]', result['result']):
    cited_ids.add(chinese_nums.get(cn, cn))

# 打印引用出处
print("📚 参考文献出处：")
if not cited_ids:
    print("未检测到明确引用。")
else:
    # 打印一下抓取到的引用编号，方便调试
    print(f"（模型引用了编号：{cited_ids}）")
    
    # 直接遍历所有检索到的文档，不比对 metadata，直接打印出来
    # 因为检索到的 source_documents 通常就是模型参考的最相关的几个，直接展示给用户看
    for i, doc in enumerate(result['source_documents']):
        content_preview = doc.page_content.replace("\n", " ")[:150]
        print(f"片段 {i+1}：{content_preview}...")