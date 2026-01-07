import streamlit as st
import time
import os

# --- 环境变量设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)             

if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.path.join(project_root, 'hf_cache')

# --- 导入模块 ---
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, MILVUS_LITE_DATA_PATH, COLLECTION_NAME,
    id_to_doc_map
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
from milvus_utils import get_milvus_client, setup_milvus_collection, index_data_if_needed, search_similar_documents
from rag_core import generate_answer, load_reranker_model, rerank_documents 

# --- Streamlit 页面配置 ---
st.set_page_config(page_title="医疗 RAG 助手", layout="wide")

# --- 侧边栏：系统状态与控制 ---
with st.sidebar:
    st.header("⚙️ 系统控制")
    
    # 清除历史按钮
    if st.button("🗑️ 清除对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
# 核心信息直接显示
    st.markdown("### 核心配置")
    st.info(f"**LLM:** {GENERATION_MODEL_NAME}")
    st.info(f"**Embedding:** {EMBEDDING_MODEL_NAME}")
    
    # 详细参数放入折叠面板，点击才会展开
    with st.expander("📊 查看详细参数", expanded=False):
        st.markdown(f"**知识库集合:** `{COLLECTION_NAME}`")
        st.markdown(f"**向量库路径:** `{MILVUS_LITE_DATA_PATH}`") 
        st.markdown(f"**数据源文件:** `{os.path.basename(DATA_FILE)}`")
        st.markdown("---")
        st.markdown(f"**最大索引文档:** `{MAX_ARTICLES_TO_INDEX}`")
        st.markdown(f"**检索 Top-K:** `{TOP_K}`")             

# --- 主标题 ---
st.title("🩺 智能医疗问答系统")
st.caption(f"基于 Milvus Lite + {GENERATION_MODEL_NAME} + BGE-Reranker 构建")

# --- 核心初始化逻辑 (使用 st.status 美化加载过程) ---
if "init_done" not in st.session_state:
    st.session_state.init_done = False

# 初始化 Session State 用于存储对话
if "messages" not in st.session_state:
    st.session_state.messages = []

# 定义全局变量占位符
milvus_client = None
embedding_model = None
generation_model = None
tokenizer = None
reranker_model = None

with st.status("正在初始化系统核心组件...", expanded=not st.session_state.init_done) as status:
    # 1. 初始化 Milvus
    st.write("🔌 连接向量数据库 (Milvus Lite)...")
    milvus_client = get_milvus_client()
    
    if milvus_client:
        setup_milvus_collection(milvus_client)
        
        # 2. 加载模型
        st.write("🧠 加载 Embedding 模型...")
        embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
        
        st.write("🚀 加载生成模型 (LLM)...")
        generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)
        
        st.write("⚖️ 加载 Re-ranker 重排序模型...")
        reranker_model = load_reranker_model("BAAI/bge-reranker-base")
        
        # 3. 处理数据
        st.write("📚 检查并索引知识库...")
        pubmed_data = load_data(DATA_FILE)
        if pubmed_data and embedding_model:
            index_data_if_needed(milvus_client, pubmed_data, embedding_model)
        
        st.session_state.init_done = True
        status.update(label="✅ 系统初始化完成！", state="complete", expanded=False)
    else:
        status.update(label="❌ Milvus 初始化失败", state="error")
        st.stop()

# --- 聊天界面逻辑 ---

# [cite_start]1.不仅显示历史消息，还要确保每次 rerun 都渲染出来 [cite: 26]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. 接收用户输入
if prompt := st.chat_input("请输入关于血液疾病的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 生成助手回复
    with st.chat_message("assistant"):
        # 创建占位符，用于动态更新状态
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        try:
            # A. 检索 (Retrieval)
            status_placeholder.markdown("🔍 *正在检索知识库...*")
            start_time = time.time()
            
            # 初筛 Top-K (建议 Config 中设为 10 或 20，给 Rerank 留空间)
            retrieved_ids, distances = search_similar_documents(milvus_client, prompt, embedding_model)
            
            if not retrieved_ids:
                full_response = "抱歉，知识库中没有找到相关信息。"
                final_docs = []
            else:
                # 映射 ID 到文本
                initial_docs = [id_to_doc_map[doc_id] for doc_id in retrieved_ids if doc_id in id_to_doc_map]
                
                # B. 重排序 (Re-ranking)
                if reranker_model:
                    status_placeholder.markdown("⚖️ *正在进行语义重排序...*")
                    # 取重排序后的 Top-3
                    final_docs = rerank_documents(prompt, initial_docs, reranker_model, top_k=3)
                else:
                    final_docs = initial_docs[:3] # 降级处理

                # C. 生成 (Generation)
                status_placeholder.markdown("✍️ *正在生成回答...*")
                answer = generate_answer(prompt, final_docs, generation_model, tokenizer)
                
                # 计算耗时
                cost_time = time.time() - start_time
                
                # 拼接最终回复 (包含引用源)
                full_response = answer + "\n\n---\n**参考文档:**"
                for idx, doc in enumerate(final_docs):
                    score_info = f"(Score: {doc.get('rerank_score', 0):.2f})" if 'rerank_score' in doc else ""
                    full_response += f"\n{idx+1}. **{doc['title']}** {score_info}"
                
                full_response += f"\n\n*(耗时: {cost_time:.2f}s)*"

            # 显示最终结果
            status_placeholder.empty() # 清除状态提示
            response_placeholder.markdown(full_response)
            
            # 保存到历史记录
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"发生错误: {str(e)}")