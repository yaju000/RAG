# 多模態 PDF 問答系統 (Multimodal RAG)

這是一個基於 **多模態 RAG（Retrieval-Augmented Generation）** 的 PDF 問答系統，結合文字和圖片兩種檢索方式，並使用 **Ollama Llama 3.1** 本地大型語言模型生成答案。

## ✨ 主要功能

- 📄 **雙模態檢索**：同時搜尋文字內容和 PDF 頁面圖片
- 🤖 **本地 LLM 生成答案**：使用 Ollama Llama 3.1，無需 API 金鑰
- 🖼️ **視覺化結果**：直接顯示最相關的 PDF 頁面
- 🔄 **智慧檔案切換**：自動偵測新文件並清除舊資料
- 🧠 **向量資料庫**：使用 Chroma 進行高效相似度搜尋

## 🏗️ 系統架構

```
PDF 文件
    ↓
┌───────────────────────────────────┐
│  1️⃣ 文字提取 + 分塊                │
│     RecursiveCharacterTextSplitter │
│     ↓                              │
│     HuggingFace Embeddings         │
│     (multilingual-MiniLM-L12-v2)   │
│     ↓                              │
│     Chroma 向量資料庫              │
└───────────────────────────────────┘
    ↓
    ↓
┌───────────────────────────────────┐
│  2️⃣ 頁面轉圖片                     │
│     PyMuPDF (fitz)                 │
│     ↓                              │
│     CLIP Embeddings                │
│     (openai/clip-vit-base-patch32) │
│     ↓                              │
│     NumPy 向量儲存                 │
└───────────────────────────────────┘
    ↓
    ↓
┌───────────────────────────────────┐
│  3️⃣ 使用者提問                     │
│     ↓                              │
│  文字向量 ←→ Chroma 搜尋          │
│  CLIP 向量 ←→ 餘弦相似度搜尋      │
│     ↓                              │
│  檢索最相關的文字塊 + 頁面圖片    │
└───────────────────────────────────┘
    ↓
    ↓
┌───────────────────────────────────┐
│  4️⃣ LLM 生成答案                   │
│     ChatOllama (llama3.1)          │
│     ↓                              │
│  顯示：AI 回答 + 參考資料         │
└───────────────────────────────────┘
```

## 🚀 安裝步驟

### 1. 安裝 Python 依賴

```bash
pip install -r requirements_rag.txt
```

### 2. 安裝 Ollama

前往 [Ollama 官網](https://ollama.ai) 下載並安裝。

### 3. 下載 Llama 3.1 模型

```bash
ollama pull llama3.1
```

### 4. 啟動 Ollama 服務

```bash
ollama serve
```

**注意**：Ollama 需要持續執行才能使用 LLM 功能。

## 📦 依賴套件

主要套件及用途：

| 套件 | 版本 | 用途 |
|------|------|------|
| `streamlit` | ≥1.30.0 | 網頁介面框架 |
| `langchain` | latest | RAG 框架 |
| `langchain-ollama` | latest | Ollama 整合 |
| `chromadb` | latest | 向量資料庫 |
| `PyMuPDF` | latest | PDF 解析與圖片轉換 |
| `transformers` | latest | CLIP 模型載入 |
| `torch` | latest | PyTorch 深度學習框架 |
| `sentence-transformers` | latest | 文字 Embedding 模型 |
| `pillow` | latest | 圖片處理 |

## 🎯 使用方法

### 啟動系統

```bash
streamlit run streamlit_rag.py
```

### 操作流程

1. **上傳 PDF**：在左側邊欄選擇 PDF 檔案
2. **調整參數**：
   - 文字分塊大小（chunk_size）：建議 500
   - 重疊大小（chunk_overlap）：建議 50
   - 返回文字塊數量：建議 3
   - 返回頁面數量：建議 2
3. **提問**：在主介面輸入問題
4. **查看結果**：
   - 🤖 **AI 回答**：頂部顯示 LLM 生成的完整答案
   - 📝 **相關文字內容**：左側顯示檢索到的文字塊
   - 🖼️ **最相關的頁面**：右側顯示原始 PDF 頁面圖片

### 檔案切換

- **自動偵測**：上傳新檔案時自動清除舊資料
- **手動切換**：點擊側邊欄「🔄 換新檔案」按鈕
- **完整清除**：點擊側邊欄「🗑️ 清除資料」按鈕

## 🔧 技術細節

### 文字處理流程

1. **提取文字**：使用 PyMuPDF 從 PDF 提取文字
2. **分塊**：`RecursiveCharacterTextSplitter` 切分文字
3. **向量化**：HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` 生成 Embeddings
4. **儲存**：Chroma 向量資料庫建立索引

### 圖片處理流程

1. **轉換**：PyMuPDF 將每頁轉為 PNG 圖片（2x 解析度）
2. **編碼**：CLIP `openai/clip-vit-base-patch32` 生成視覺 Embeddings
3. **儲存**：Base64 編碼儲存於記憶體

### 檢索機制

- **文字檢索**：Chroma 相似度搜尋（餘弦相似度）
- **圖片檢索**：CLIP 文字-圖片跨模態搜尋
  ```python
  similarity = cosine_similarity(query_embedding, image_embeddings)
  ```

### LLM 生成

- **模型**：Ollama Llama 3.1（本地執行）
- **Prompt 結構**：
  ```
  根據以下 PDF 文件內容，用繁體中文回答問題。
  
  問題：{question}
  
  參考資料：
  {context}
  
  請提供詳細的回答，並在回答最後列出引用的頁碼。
  ```
- **候補方案**：若 Ollama 無法連線，顯示摘要式參考資料

## 📁 檔案說明

```
NotOusterAvoidance-collision_box_dev/
├── multimodal_rag_streamlit.py    # 主程式（Streamlit 網頁介面）
├── requirements_rag.txt            # Python 依賴清單
├── README_RAG.md                   # 本說明文件
└── temp_*.pdf                      # 暫存上傳的 PDF（自動清除）
```

## 🔒 隱私與安全

- ✅ **完全本地執行**：無需上傳資料到雲端
- ✅ **無需 API 金鑰**：使用 Ollama 本地模型
- ✅ **自動清除**：臨時檔案可手動刪除
- ✅ **記憶體資料庫**：Chroma 不持久化到硬碟

## 📊 效能建議

### 硬體需求

- **RAM**：至少 8GB（建議 16GB）
- **GPU**：非必要，但可加速 CLIP Embedding（支援 CUDA）
- **硬碟**：約 3GB 空間（模型快取）

## 🛠️ 開發與自訂

### 更換 LLM 模型

修改 `generate_answer_from_chunks()` 函式：

```python
# 使用其他 Ollama 模型
model = ChatOllama(model="mistral")  # 或 "codellama", "llama2" 等

# 或使用 OpenAI
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4", api_key="your-key")
```

### 調整 Embedding 模型

```python
# 更換文字 Embedding
text_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # 英文優化
)

# 更換 CLIP 模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
```

## 📚 參考資源

- [LangChain 文檔](https://python.langchain.com/)
- [Ollama 官網](https://ollama.ai)
- [Chroma 向量資料庫](https://www.trychroma.com/)
- [CLIP 論文](https://arxiv.org/abs/2103.00020)
- [Streamlit 文檔](https://docs.streamlit.io/)


**最後更新**：2025-11-26
