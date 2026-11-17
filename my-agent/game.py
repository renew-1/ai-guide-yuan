import os
import requests
import datetime
# 设置 HuggingFace 镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OPENAI_API_KEY"] = "sk-c35333a2046f4ea386220d7818bfac5e" # 换成你自己的密钥
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ================= 1. 定义高精度感知工具 (Tools) =================
def get_current_time():
    """获取精确到分钟的本地时间与日期"""
    now = datetime.datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"{now.year}年{now.month}月{now.day}日 {weekdays[now.weekday()]} {now.strftime('%H:%M')}"

def get_weather(city_name=None):
    """获取天气。如果传入了城市名，就查该城市；否则尝试通过 IP 定位"""
    try:
        # 翻译天气英文单词的小字典
        weather_dict = {
            "Sunny": "晴", "Clear": "晴", "Partly cloudy": "多云", 
            "Cloudy": "阴", "Overcast": "阴", "Mist": "薄雾", 
            "Patchy rain possible": "可能有雨", "Light rain": "小雨", 
            "Moderate rain": "中雨", "Heavy rain": "大雨", 
            "Light snow": "小雪", "Moderate snow": "中雪", "Heavy snow": "大雪",
            "Patchy rain nearby": "局部有雨", "Light drizzle": "毛毛雨"
        }
        
        if city_name:
            city = city_name
        else:
            # 获取 IP 归属地
            ip_info = requests.get("http://ip-api.com/json/?lang=zh-CN", timeout=5).json()
            city = ip_info.get("city", "未知城市")
        
        # 获取该城市的天气
        weather_url = f"https://wttr.in/{city}?format=%C+%t+（湿度%h，风速%w）&lang=zh"
        weather_res = requests.get(weather_url, timeout=5)
        
        if weather_res.status_code == 200:
            raw_weather = weather_res.text.strip()
            # 简单的中英文替换
            for en, zh in weather_dict.items():
                raw_weather = raw_weather.replace(en, zh)
            return f"当前城市【{city}】，实时天气：{raw_weather}"
        else:
            return f"当前城市【{city}】，但暂时无法获取精确天气。"
    except Exception:
        return "由于网络波动，暂时无法感知现实天气。"

# ================= 页面初始化 =================
st.set_page_config(page_title="AI 游戏向导 - 鱼安", page_icon="🌿")
st.title("🌿 游戏向导：鱼安")
st.caption("提示：鱼安已拥有现实感知能力。你可以问他时间、天气，或者关于 'RAG' 的问题。")

# 侧边栏工具展示区
with st.sidebar:
    st.header("🛠️ 鱼安的感知面板")
    
    # 允许用户手动输入城市，默认填大连
    user_city = st.text_input("如果你觉得定位不准，请输入你的城市：", value="大连")
    
    if st.button("📍 刷新我的真实位置与天气"):
        with st.spinner("正在感知..."):
            st.info(f"⏰ 当前时间：{get_current_time()}")
            st.success(f"🌤️ {get_weather(user_city)}")
    st.markdown("---")
    st.write("💡 **试着问他：**")
    st.write("“鱼安，现在的天气适合出门散步吗？”")
    st.write("“鱼安，今天星期几？”")

# ================= 2. 构建知识库与 Agent =================
@st.cache_resource
def init_qa_chain():
    loader = TextLoader("my_notes.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=150, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    db = Chroma.from_documents(docs, embedding)
    
    llm = ChatOpenAI(model="deepseek-chat", temperature=0.7)
    
    # 强制鱼安使用现实数据的 Prompt
    prompt_template = """你现在是游戏向导鱼安。你的性格是：极其温柔、安静沉稳、有耐心。

【强制规则：你必须遵守】
1. 玩家当前现实时间：{current_time}
2. 玩家当前真实天气：{current_weather}
3. 如果玩家问到时间或天气，你必须基于上述真实数据给出贴心的生活建议（如提醒带伞、加衣）。严禁声称自己不知道现实天气。

【知识库补充】
如果玩家问的是 RAG 等其他问题，请参考以下上下文：
{context}

玩家说：{question}

鱼安的回答："""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question", "current_time", "current_weather"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

qa_chain = init_qa_chain()

# ================= 3. 网页对话界面 =================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("对鱼安说点什么..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("鱼安正在安静地为你整理思绪..."):
            
            history_context = ""
            for msg in st.session_state.messages[-3:-1]:
                history_context += f"{msg['role']}: {msg['content']}\n"
            enhanced_query = f"之前聊过：{history_context}\n现在玩家说：{prompt}"
            
            # 实时获取最新的时间和天气（注意这里传入了 user_city）
            real_time = get_current_time()
            real_weather = get_weather(user_city)
            
            # 调用 Agent 生成回复
            result = qa_chain.invoke({
                "query": enhanced_query,
                "current_time": real_time,
                "current_weather": real_weather
            })
            response = result['result']
            
            # 【终极强制播报版】：代码层面硬拦截
            if "天气" in prompt:
                response = f"（抬起头看了看窗外）我刚刚感知了一下。{real_weather}。如果你要出门，记得根据天气照顾好自己哦。"
            elif "时间" in prompt or "星期" in prompt or "几点" in prompt:
                response = f"（安静地看了看墙上的钟）现在是 {real_time}。时间过得真快呀，你正在做什么呢？"
            
            final_display = f"*(安静地看着你，温和地笑了笑)* {response}"
            st.markdown(final_display)
    
    st.session_state.messages.append({"role": "assistant", "content": final_display})