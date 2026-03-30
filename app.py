
import streamlit as st
import requests
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import PyPDF2
import glob

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

st.set_page_config(
    page_title="ResearchAdmin Guard | 연구비 규정 검토 AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background-color: #F8F9FB;
    }
    
    /* 헤더 */
    .main-header {
        background: linear-gradient(135deg, #1B2A4A 0%, #2E4080 100%);
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .main-header h1 {
        color: white;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #A8B8D8;
        font-size: 13px;
        margin: 4px 0 0 0;
    }
    .header-badge {
        background: rgba(255,255,255,0.15);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* FAQ 버튼 */
    .stButton > button {
        background-color: white;
        color: #1B2A4A;
        border: 1.5px solid #E2E8F0;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        padding: 8px 14px;
        width: 100%;
        text-align: left;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #EEF2FF;
        border-color: #2E4080;
        color: #2E4080;
    }
    
    /* 채팅 메시지 */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 12px;
    }
    
    /* 사용자 메시지 */
    [data-testid="stChatMessageContent"] {
        font-size: 14px;
        line-height: 1.7;
    }
    
    /* 채팅 입력창 */
    .stChatInput > div {
        border-radius: 12px;
        border: 2px solid #E2E8F0;
        background: white;
    }
    .stChatInput > div:focus-within {
        border-color: #2E4080;
        box-shadow: 0 0 0 3px rgba(46,64,128,0.1);
    }
    
    /* 사이드바 */
    [data-testid="stSidebar"] {
        background-color: #1B2A4A;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background-color: rgba(255,255,255,0.1);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(255,255,255,0.2);
    }
    
    /* 법령 뱃지 */
    .law-badge {
        background: rgba(46,64,128,0.1);
        color: #2E4080;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 6px;
        display: block;
        border-left: 3px solid #2E4080;
    }
    
    /* FAQ 섹션 타이틀 */
    .section-title {
        font-size: 12px;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    
    /* 구분선 */
    hr {
        border-color: #E2E8F0;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

def extract_pdf_text(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

@st.cache_resource
def load_laws():
    pdf_files = glob.glob("laws/*.pdf")
    if not pdf_files:
        return None, []
    full_text = ""
    loaded_files = []
    for pdf_path in pdf_files:
        text = extract_pdf_text(pdf_path)
        full_text += f"[출처: {os.path.basename(pdf_path)}]\n" + text
        loaded_files.append(os.path.basename(pdf_path))
    embeddings = load_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore, loaded_files

def ask_ai(question, context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    system_prompt = """당신은 대한민국 국가연구개발혁신법 전문가입니다.
반드시 아래 규칙을 지켜주세요:
1. 오직 한국어(한글)로만 답변하세요
2. 한자, 일본어, 중국어, 베트남어를 절대 사용하지 마세요
3. 줄바꿈을 사용해서 읽기 쉽게 작성하세요
4. 금액, 한도, 비율 등 구체적인 수치가 법령에 있으면 반드시 명시하세요
5. 답변 마지막에 반드시 각주를 아래 형식으로 추가하세요:

---
**[참고 법령]**
[1] 법령명 제X조 (조항 제목)
[2] 법령명 제X조 (조항 제목)

6. 법령에 없는 경우 아래 순서로 답변하세요:
   - 유사한 공공기관 사례나 일반적인 관행 안내
   - 관련 담당 부서 안내 (예: 연구처, 과기정통부, 한국연구재단 등)
   - 일반적인 절차적 관점에서 설명
   - 추가 문의처 안내"""

    user_prompt = f"""아래 법령 내용을 참고해서 질문에 답변해주세요.

[법령 내용]
{context}

[질문]
{question}"""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.1
    }
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    return f"오류: {result}"

def do_search(question):
    if not question:
        return
    vectorstore = st.session_state.get("vectorstore")
    if not vectorstore:
        st.error("법령 데이터 로딩 중입니다. 잠시 후 다시 시도해주세요.")
        return
    with st.spinner("법령 검토 중..."):
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        answer = ask_ai(question, context)
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        st.session_state["chat_history"].append({
            "question": question,
            "answer": answer
        })

# 벡터스토어 로드
if "vectorstore" not in st.session_state:
    with st.spinner("법령 데이터 로딩 중..."):
        vectorstore, loaded_files = load_laws()
        st.session_state["vectorstore"] = vectorstore
        st.session_state["loaded_files"] = loaded_files

# 사이드바
with st.sidebar:
    st.markdown("### 🛡️ ResearchAdmin Guard")
    st.markdown("---")
    st.markdown("**📚 학습된 법령**")
    for f in st.session_state.get("loaded_files", []):
        name = f[:25] + "..." if len(f) > 25 else f
        st.markdown(f"✅ {name}")
    st.markdown("---")
    st.markdown(f"**총 {len(st.session_state.get('loaded_files', []))}개 법령 학습 완료**")
    st.markdown("---")
    if st.button("🗑️ 대화 초기화"):
        st.session_state["chat_history"] = []
        st.rerun()
    st.markdown("---")
    st.markdown("**문의처**")
    st.markdown("과기정통부: 044-202-4900")
    st.markdown("한국연구재단: 042-869-6114")

# 메인 헤더
st.markdown("""
<div class="main-header">
    <div>
        <h1>🛡️ ResearchAdmin Guard</h1>
        <p>국가연구개발혁신법 기반 연구비 규정 자동 검토 시스템</p>
    </div>
    <span class="header-badge">AI POWERED</span>
</div>
""", unsafe_allow_html=True)

# FAQ 섹션
st.markdown('<p class="section-title">자주 묻는 질문</p>', unsafe_allow_html=True)

faqs = [
    "연구개발비로 식비를 사용할 수 있나요?",
    "연구비로 노트북을 구매할 수 있나요?",
    "간접비 계상 기준이 무엇인가요?",
    "출장비 일비 한도가 얼마인가요?",
    "연구장비 구매 한도는 얼마인가요?",
    "연구비 불용 처리 방법은?"
]

cols = st.columns(3)
for i, faq in enumerate(faqs):
    with cols[i % 3]:
        if st.button(faq, key=f"faq_{i}"):
            do_search(faq)
            st.rerun()

st.markdown("---")

# 채팅 기록
if "chat_history" in st.session_state and st.session_state["chat_history"]:
    st.markdown('<p class="section-title">대화 기록</p>', unsafe_allow_html=True)
    for chat in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
    st.markdown("---")

# 질문 입력창
question = st.chat_input("규정 관련 질문을 입력하고 Enter를 누르세요...")
if question:
    do_search(question)
    st.rerun()
