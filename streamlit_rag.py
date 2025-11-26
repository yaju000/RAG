"""
多模態 RAG 系統 - Streamlit 網頁介面
可以直接顯示最相關的 PDF 頁面圖片
"""
import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
import tempfile
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel, Qwen2VLForConditionalGeneration
from typing import List
import fitz
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image   
from transformers import CLIPProcessor, CLIPModel
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

# =====================================================
# 配置
# =====================================================
st.set_page_config(page_title="PDF 問答系統", layout="wide")

# =====================================================
# 快取模型載入
# =====================================================
@st.cache_resource
def load_models():
    """載入模型（只執行一次）"""
    import sys
    
    print("="*60, file=sys.stderr)
    print("開始載入模型...", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    # CLIP 模型
    print("1/2 正在載入 CLIP 模型...", file=sys.stderr)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("   ✓ CLIP Processor 載入完成", file=sys.stderr)
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    print("   ✓ CLIP Model 載入完成", file=sys.stderr)
    
    # 文字 embedding 模型
    print("2/2 正在載入文字 Embedding 模型...", file=sys.stderr)
    text_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print("   ✓ 文字 Embedding 模型載入完成", file=sys.stderr)
    
    print("="*60, file=sys.stderr)
    print("所有模型載入完成！", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    return clip_processor, clip_model, text_embeddings

@st.cache_data
def process_pdf(pdf_path, chunk_size=500, chunk_overlap=50):
    """處理 PDF：提取文字和圖片"""
    doc = fitz.open(pdf_path)
    
    # 1. 提取文字並分塊
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    text_chunks = []
    image_data = []
    
    clip_processor, clip_model, _ = load_models()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 提取文字
        text = page.get_text()
        if text.strip():
            chunks = text_splitter.create_documents(
                texts=[text],
                metadatas=[{"page": page_num + 1, "type": "text"}]
            )
            text_chunks.extend(chunks)
        
        # 轉換為圖片
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        # 生成 CLIP embedding
        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            image_embedding = clip_model.get_image_features(**inputs).squeeze(0).numpy()
        
        # 儲存圖片資料
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        image_data.append({
            "page": page_num + 1,
            "embedding": image_embedding,
            "image_base64": img_base64
        })
        
        # 更新進度
        progress = (page_num + 1) / len(doc)
        progress_bar.progress(progress)
        status_text.text(f"處理中... {page_num + 1}/{len(doc)} 頁")
    
    doc.close()
    progress_bar.empty()
    status_text.empty()
    
    return text_chunks, image_data

def build_vectorstore(text_chunks, text_embeddings):
    """建立文字向量資料庫（不使用快取，每次都重新建立）"""
    # 使用臨時的記憶體資料庫，不持久化
    return Chroma.from_documents(
        documents=text_chunks,
        embedding=text_embeddings,
        collection_name=f"temp_collection_{hash(str(text_chunks[0].page_content[:100]))}"
    )

# =====================================================
# 查詢函式
# =====================================================
def query_multimodal(question, text_vectorstore, image_data, top_k_text=3, top_k_image=2):
    """多模態查詢"""
    clip_processor, clip_model, _ = load_models()
    
    # 1. 搜尋文字
    text_results = text_vectorstore.similarity_search(question, k=top_k_text)
    
    # 2. 使用 CLIP 搜尋圖片
    inputs = clip_processor(text=[question], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = clip_model.get_text_features(**inputs).squeeze(0).numpy()
    
    # 計算相似度
    image_embeddings = np.array([img["embedding"] for img in image_data])
    
    # 正規化
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    image_embeddings_norm = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    
    # 餘弦相似度
    similarities = np.dot(image_embeddings_norm, query_embedding)
    top_k_indices = np.argsort(similarities)[::-1][:top_k_image]
    
    image_results = [image_data[idx] for idx in top_k_indices]
    
    return text_results, image_results

def generate_answer_from_chunks(question, text_results):
    """根據檢索到的文字塊生成答案 - 使用 Ollama Llama 3.1"""
    # 組合參考資料
    context = "\n\n".join([
        f"【第 {doc.metadata.get('page')} 頁】\n{doc.page_content}" 
        for doc in text_results
    ])
    
    # 使用 Ollama Llama 3.1
    try:
        
        # 初始化 Ollama 模型
        model = ChatOllama(model="llama3.1")
        
        prompt = f"""根據以下 PDF 文件內容，用繁體中文回答問題。

                    問題：{question}

                    參考資料：
                    {context}

                    請提供詳細的回答，並在回答最後列出引用的頁碼。"""

        # 呼叫模型生成答案
        response = model.invoke(prompt)
        
        return response.content
        
    except Exception as e:
        st.warning(f"⚠️ Ollama 呼叫失敗：{str(e)}，使用候補摘要")
        st.info("💡 請確認 Ollama 已安裝並執行，且已下載 llama3.1 模型")
    
    # 候補方案：提供摘要式回答
    summary_parts = []
    for i, doc in enumerate(text_results, 1):
        page = doc.metadata.get('page')
        content = doc.page_content.strip()[:300]  # 取前300字
        summary_parts.append(f"**參考 {i}（第 {page} 頁）：**\n{content}...")
    
    fallback_answer = f"""**根據文件內容的摘要回答：**
                    {chr(10).join(summary_parts)}

                    💡 提示：若要使用 LLM 生成完整答案，請確認：
                    1. 已安裝 Ollama (https://ollama.ai)
                    2. 已下載模型：ollama pull llama3.1
                    3. Ollama 服務正在執行"""
    
    return fallback_answer

# =====================================================
# Streamlit 介面
# =====================================================
def main():
    st.title("🔍 多模態 PDF 問答系統")
    
    # 在頁面頂部顯示載入狀態
    with st.spinner("正在初始化系統..."):
        try:
            # 預先載入模型（會顯示在終端機）
            load_models()
            st.success("✅ 系統初始化完成！")
        except Exception as e:
            st.error(f"❌ 模型載入失敗: {str(e)}")
            st.stop()
    
    st.markdown("---")
    
    # 側邊欄：上傳 PDF
    with st.sidebar:
        st.header("📁 上傳 PDF")
        uploaded_file = st.file_uploader("選擇 PDF 檔案", type=["pdf"])
        
        # 顯示當前載入的文件
        if "current_pdf_name" in st.session_state:
            st.success(f"✅ 已載入：{st.session_state.current_pdf_name}")
            if st.button("🔄 換新檔案"):
                # 清除所有資料，允許上傳新檔案
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        chunk_size = st.slider("文字分塊大小", 100, 1000, 500)
        chunk_overlap = st.slider("重疊大小", 0, 200, 50)
        
        st.markdown("---")
        st.header("⚙️ 查詢設定")
        top_k_text = st.slider("返回文字塊數量", 1, 10, 3)
        top_k_image = st.slider("返回頁面數量", 1, 5, 2)

        st.markdown("---")
        st.header("🦙 LLM 模型")
        st.info("使用 Ollama Llama 3.1")
        st.caption("請確認 Ollama 服務正在執行")
    
    # 主介面
    if uploaded_file is not None:
        # 儲存上傳的檔案
        pdf_path = f"temp_{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 檢查是否為新文件（比較檔名）
        current_file_name = uploaded_file.name
        if "current_pdf_name" not in st.session_state or st.session_state.current_pdf_name != current_file_name:
            # 新文件！清除舊資料
            st.info(f"📄 偵測到新文件：{current_file_name}，正在處理...")
            
            # 清除舊的 session state
            for key in ["text_chunks", "image_data", "text_vectorstore"]:
                if key in st.session_state:
                    del st.session_state[key]
            
            # 處理新 PDF
            text_chunks, image_data = process_pdf(pdf_path, chunk_size, chunk_overlap)
            
            _, _, text_embeddings = load_models()
            text_vectorstore = build_vectorstore(text_chunks, text_embeddings)
            
            # 儲存到 session state
            st.session_state.text_chunks = text_chunks
            st.session_state.image_data = image_data
            st.session_state.text_vectorstore = text_vectorstore
            st.session_state.current_pdf_name = current_file_name  # 記錄當前檔名
            
            st.success(f"✅ 處理完成！文字塊: {len(text_chunks)}, 頁面: {len(image_data)}")
        else:
            # 同一個文件，顯示已載入的資訊
            st.info(f"📄 當前文件：{current_file_name} （已載入）")
        
        # 問答介面
        st.header("💬 提問")
        question = st.text_input("請輸入您的問題：", placeholder="例如：培訓計劃的主要內容是什麼？")
        
        if st.button("🔍 查詢", type="primary") and question:
            with st.spinner("查詢中..."):
                text_results, image_results = query_multimodal(
                    question,
                    st.session_state.text_vectorstore,
                    st.session_state.image_data,
                    top_k_text=top_k_text,
                    top_k_image=top_k_image
                )
            
            # 顯示結果
            st.markdown("---")

             # 🤖 LLM 回答區塊（置於最上方）
            st.subheader("🤖 AI 回答")
            with st.spinner("生成答案中..."):
                answer = generate_answer_from_chunks(question, text_results)
            
            # 使用醒目的容器顯示答案
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                        {answer.replace(chr(10), '<br>')}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            st.markdown("### 📚 參考資料")
            
            # 左右分欄
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📝 相關文字內容")
                for i, doc in enumerate(text_results, 1):
                    with st.expander(f"文字塊 {i} - 第 {doc.metadata.get('page')} 頁"):
                        st.write(doc.page_content)
            
            with col2:
                st.subheader("🖼️ 最相關的頁面")
                for i, img_data in enumerate(image_results, 1):
                    st.write(f"**第 {img_data['page']} 頁**")
                    img_html = f'<img src="data:image/png;base64,{img_data["image_base64"]}" style="width:100%">'
                    st.markdown(img_html, unsafe_allow_html=True)
                    st.markdown("---")
        
        # 清理臨時檔案
        if st.sidebar.button("🗑️ 清除資料"):
            # 刪除臨時 PDF
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            
            # 刪除所有 temp_ 開頭的檔案
            import glob
            for temp_file in glob.glob("temp_*.pdf"):
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            # 刪除 Chroma 資料庫目錄（如果存在）
            import shutil
            if os.path.exists("./chroma"):
                shutil.rmtree("./chroma")
            
            # 清除所有 session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.success("✅ 已清除所有資料")
            st.rerun()
    
    else:
        st.info("👈 請在側邊欄上傳 PDF 檔案")


if __name__ == "__main__":
    main()
