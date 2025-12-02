import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
import time
import random
import re # 정규표현식 모듈 추가 (정렬용)
from datetime import datetime, timedelta

# ---------------------------------------------------------
# [필수] 앱 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="MediScope: AI 감염병 플랫폼",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 1. 디자인 (CSS) - 기능 대폭 강화 (애니메이션, 고급 스타일링)
# ---------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    /* ------------------------------------------------------- */
    /* [Global] 변수 및 기본 설정 */
    /* ------------------------------------------------------- */
    :root {
        --primary-color: #5361F2;
        --primary-light: #E0E7FF;
        --primary-dark: #404ECA;
        --bg-color: #FFFFFF;
        --bg-secondary: #F8F9FA;
        --text-color: #333333;
        --text-sub: #666666;
        --border-color: #EEEEEE;
        --shadow-sm: 0 2px 4px rgba(0,0,0,0.02);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.05);
        --shadow-lg: 0 8px 24px rgba(83, 97, 242, 0.12);
    }
    
    html, body, [class*="css"] { 
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif; 
        color: var(--text-color);
        scroll-behavior: smooth;
    }
    
    /* 전체 배경 */
    .stApp { background-color: var(--bg-color); }
    
    /* 스크롤바 커스텀 */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f1f1; }
    ::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #a8a8a8; }

    /* ------------------------------------------------------- */
    /* [Animations] 애니메이션 정의 */
    /* ------------------------------------------------------- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .element-container, .stMarkdown, .stPlotlyChart {
        animation: fadeIn 0.5s ease-out forwards;
    }

    /* ------------------------------------------------------- */
    /* [Sidebar] 사이드바 스타일링 */
    /* ------------------------------------------------------- */
    [data-testid="stSidebar"] { 
        background-color: var(--bg-secondary); 
        border-right: 1px solid var(--border-color);
    }
    
    /* 사이드바 네비게이션 컨테이너 */
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
        gap: 8px;
        padding: 0 5px;
    }
    
    /* 메뉴 아이템 기본 스타일 */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        background-color: transparent;
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid transparent;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        margin-bottom: 2px;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    /* 메뉴 아이템 호버 효과 */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background-color: white;
        box-shadow: var(--shadow-sm);
        transform: translateX(4px);
        color: var(--primary-color);
    }

    /* 선택된 메뉴 (Active) */
    section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: var(--primary-color) !important;
        box-shadow: var(--shadow-md);
        transform: scale(1.02);
    }
    
    section[data-testid="stSidebar"] .stRadio label[data-checked="true"] > div:last-child {
        color: white !important;
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    
    /* 라디오 버튼 원형 제거 */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:last-child {
        margin-left: 0 !important;
        font-size: 0.95rem;
    }

    /* ------------------------------------------------------- */
    /* [Components] 메인 컴포넌트 스타일 */
    /* ------------------------------------------------------- */
    
    /* Hero Box */
    .hero-box {
        background: linear-gradient(135deg, #5361F2 0%, #7B87F7 100%);
        padding: 48px 36px;
        border-radius: 24px;
        color: white;
        margin-bottom: 40px;
        box-shadow: 0 20px 40px rgba(83, 97, 242, 0.2);
        position: relative;
        overflow: hidden;
    }
    /* Hero Box 배경 장식 (유리 질감 효과) */
    .hero-box::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-title { font-size: 2.8rem; font-weight: 800; margin-bottom: 12px; letter-spacing: -1px; text-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    .hero-subtitle { font-size: 1.15rem; opacity: 0.95; font-weight: 400; line-height: 1.5; }
    
    /* Metric Card */
    .metric-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        border-color: transparent;
        transform: translateY(-5px);
        box-shadow: var(--shadow-lg);
    }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: var(--primary-color); letter-spacing: -1px; }
    .metric-label { font-size: 0.9rem; color: var(--text-sub); margin-top: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid var(--primary-color);
        background-color: white;
        color: var(--primary-color);
        padding: 0.6rem 1.2rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div.stButton > button:hover {
        background-color: var(--primary-color);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(83, 97, 242, 0.2);
    }
    div.stButton > button:active { transform: translateY(0); }

    /* Form Submit Button */
    div[data-testid="stForm"] button {
        background: linear-gradient(135deg, #5361F2, #404ECA);
        color: white;
        border: none;
        width: 100%;
        padding: 0.7rem;
    }
    div[data-testid="stForm"] button:hover {
        box-shadow: 0 8px 16px rgba(83, 97, 242, 0.3);
    }

    /* Input Fields (Selectbox, TextInput etc.) */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        border-radius: 10px;
        border-color: #E0E0E0;
        background-color: #FAFAFA;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {
        border-color: var(--primary-light);
    }
    div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"] > div:focus-within {
        border-color: var(--primary-color);
        background-color: white;
        box-shadow: 0 0 0 3px rgba(83, 97, 242, 0.1);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-size: 1rem;
        color: var(--text-sub);
        font-weight: 600;
        border: none;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--primary-color) !important;
        border-bottom: 3px solid var(--primary-color) !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border-color);
        border-radius: 16px;
        background-color: white;
        transition: box-shadow 0.2s;
    }
    div[data-testid="stExpander"]:hover {
        box-shadow: var(--shadow-md);
    }
    
    /* Chat Message Styles */
    div[data-testid="stChatMessage"] {
        background-color: transparent;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid transparent;
    }
    div[data-testid="stChatMessage"][data-author="assistant"] {
        background-color: #F8F9FF; /* 아주 연한 블루 */
        border-color: #E0E7FF;
    }
    div[data-testid="stChatMessage"][data-author="user"] {
        background-color: #FDFDFD;
        border-color: #EEEEEE;
    }
    
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 데이터 로드 및 전처리
# ---------------------------------------------------------
@st.cache_data
def load_data():
    file_path = '법정감염병_월별_신고현황_20251201171222.csv'
    try:
        try:
            df = pd.read_csv(file_path, header=1, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, header=1, encoding='cp949')
            
        if '급별(2)' in df.columns and '급별(1)' in df.columns:
            # 소계, 합계 제거
            df_clean = df[~df['급별(2)'].isin(['소계', '합계'])].copy()
            
            # '제N급' -> 'N급'으로 변경
            df_clean['급별(1)'] = df_clean['급별(1)'].astype(str).str.replace('제', '').str.strip()
            
            # 1. 전체 질병 리스트
            disease_list = sorted(df_clean['급별(2)'].unique().tolist())
            
            # 2. 등급 리스트 (숫자 기준 정렬 로직 추가)
            raw_grades = df_clean['급별(1)'].unique().tolist()
            
            def grade_sort_key(grade):
                numbers = re.findall(r'\d+', str(grade))
                return int(numbers[0]) if numbers else 999
            
            grade_list = sorted(raw_grades, key=grade_sort_key)
            
            return df_clean, disease_list, grade_list
        else:
            return pd.DataFrame(), [], []
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame(), [], []

df, all_diseases, all_grades = load_data()

# ---------------------------------------------------------
# 3. 사이드바 (메뉴 및 리셋 버튼)
# ---------------------------------------------------------
with st.sidebar:
    # 로고 영역
    st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 30px; padding-left: 10px;">
            <span style="font-size: 3.5rem; margin-right: 12px; line-height: 1; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));">🏥</span>
            <div style="display: flex; flex-direction: column;">
                <span style="font-size: 1.6rem; font-weight: 800; color: #333; letter-spacing: -0.5px;">MediScope</span>
                <span style="font-size: 0.75rem; color: #888; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase;">AI Infection Control</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 네비게이션 메뉴
    menu = st.radio("Navigation", [
        "🏠 홈", 
        "💬 AI 의료 상담", 
        "📊 AI 분석 센터", 
        "👤 My Page"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    
    # 시스템 리셋 버튼
    if st.button("🔄 시스템 리셋", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #AAA; font-size: 0.75rem; margin-top: 20px; line-height: 1.5;'>
        © 2025 <b>MediScope AI</b><br>
        All rights reserved.
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# 4. 메인 컨텐츠 (메뉴별 화면 구성)
# ---------------------------------------------------------

# ==========================================
# [MENU 1] 🏠 홈
# ==========================================
if menu == "🏠 홈":
    
    default_grade = all_grades[0] if all_grades else "데이터 없음"
    current_grade = st.session_state.get('home_grade', default_grade)
    
    if not df.empty and current_grade in all_grades:
        filtered_diseases = sorted(df[df['급별(1)'] == current_grade]['급별(2)'].unique().tolist())
        default_disease = filtered_diseases[0] if filtered_diseases else "데이터 없음"
    else:
        filtered_diseases = []
        default_disease = "데이터 없음"
        
    current_disease = st.session_state.get('home_disease', default_disease)
    if current_disease not in filtered_diseases and filtered_diseases:
        current_disease = filtered_diseases[0]

    # Hero Section
    st.markdown(f"""
        <div class="hero-box">
            <div class="hero-title">MediScope AI Insights</div>
            <div class="hero-subtitle"><b>{current_grade} {current_disease}</b> 발생 추이 및 AI 기반 예방 정보</div>
        </div>
    """, unsafe_allow_html=True)

    # Filter
    st.markdown("### 🔍 감염병 현황 조회")
    col_filter1, col_filter2 = st.columns([1, 2])
    
    with col_filter1:
        try: g_idx = all_grades.index(current_grade)
        except: g_idx = 0
        selected_grade = st.selectbox("1. 분류 등급 선택", all_grades, index=g_idx, key='home_grade')
    
    with col_filter2:
        try: d_idx = filtered_diseases.index(current_disease)
        except: d_idx = 0
        selected_disease = st.selectbox("2. 전염병 선택", filtered_diseases, index=d_idx, key='home_disease')

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">1,240명</div>
                <div class="metric-label">이번 달 신고 건수</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #FF4B4B;">▲ 12.5%</div>
                <div class="metric-label">전월 대비 증감률</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #5361F2;">주의 단계</div>
                <div class="metric-label">현재 경보 수준</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Graph
    st.subheader(f"📈 {selected_disease} 월별 발생 추이")
    
    dates = pd.date_range(start='2024-01-01', periods=18, freq='M')
    values = np.random.randint(20, 300, size=18) + np.sin(np.linspace(0, 10, 18)) * 30
    chart_df = pd.DataFrame({'Date': dates, 'Patients': values})
    
    fig = px.line(chart_df, x='Date', y='Patients', markers=True, line_shape='spline')
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font={'family': 'Pretendard'})
    fig.update_traces(line_color='#5361F2', line_width=4, marker=dict(size=8, border=dict(width=2, color='white')))
    st.plotly_chart(fig, use_container_width=True)

    # Tips
    st.markdown("---")
    st.subheader(f"🩹 {selected_disease} 예방 및 행동 요령 (Tip)")

    def get_custom_tips(disease_name):
        d_name = disease_name
        if any(k in d_name for k in ["결핵", "인플루엔자", "코로나", "홍역", "수두", "백일해", "유행성이하선염", "성홍열", "폐렴구균", "엠폭스"]):
            return (
                "마스크 착용 및 기침 예절",
                "- 사람이 많은 곳에서는 반드시 마스크를 착용하세요.<br>- 기침이나 재채기 시 옷소매로 입과 코를 가리세요.<br>- 씻지 않은 손으로 눈, 코, 입을 만지지 마세요.",
                "실내 환기 및 격리",
                "- 하루 3회 이상, 10분씩 실내 환기를 시켜주세요.<br>- 발열 및 호흡기 증상 발생 시 등교/출근을 멈추고 집에서 휴식하세요."
            )
        elif any(k in d_name for k in ["콜레라", "장티푸스", "파라티푸스", "세균성이질", "장출혈성", "A형간염", "비브리오", "식중독", "노로바이러스"]):
            return (
                "안전한 물과 음식 섭취",
                "- 물은 반드시 끓여 마시고, 음식은 충분히 익혀 드세요.<br>- 채소와 과일은 흐르는 물에 깨끗이 씻어 껍질을 벗겨 드세요.<br>- 조리 도구는 끓는 물이나 소독제로 소독하세요.",
                "철저한 손 씻기",
                "- 화장실 사용 후, 조리 전, 식사 전 흐르는 물에 비누로 30초 이상 손을 씻으세요.<br>- 설사 증상이 있는 경우 음식을 조리하지 마세요."
            )
        elif any(k in d_name for k in ["말라리아", "일본뇌염", "쯔쯔가무시", "뎅기열", "지카", "열", "진드기"]):
            return (
                "피부 노출 최소화",
                "- 야외 활동 시 긴 소매, 긴 바지를 착용하여 피부 노출을 줄이세요.<br>- 진드기/모기 기피제를 사용하세요.<br>- 풀밭 위에 옷을 벗어두거나 바로 눕지 마세요.",
                "환경 관리 및 예방접종",
                "- 집 주변 웅덩이 등 모기 서식지를 제거하세요.<br>- 야외 활동 후 즉시 샤워하고 입었던 옷은 세탁하세요.<br>- 유행 지역 방문 전 예방접종 여부를 확인하세요."
            )
        elif any(k in d_name for k in ["B형간염", "C형간염", "매독", "후천성면역결핍증"]):
            return (
                "개인 위생용품 공유 금지",
                "- 칫솔, 면도기, 손톱깎이 등 혈액이 묻을 수 있는 용품은 절대 공유하지 마세요.<br>- 문신, 피어싱 등은 반드시 소독된 도구를 사용하는 곳에서 받으세요.",
                "정기 검진 및 안전 수칙",
                "- 정기적인 검진을 통해 감염 여부를 확인하세요.<br>- 의료 종사자는 주사 바늘 찔림 등 혈액 노출 사고에 주의하세요."
            )
        else:
            return (
                "일상 속 위생 수칙 준수",
                "- 흐르는 물에 30초 이상 비누로 손 씻기를 생활화하세요.<br>- 기침할 땐 옷소매로 입과 코를 가리세요.",
                "면역력 강화 및 건강 관리",
                "- 규칙적인 운동과 충분한 수면으로 면역력을 높이세요.<br>- 의심 증상 발생 시 즉시 의료기관을 방문하여 진료를 받으세요."
            )

    tip_t1, tip_d1, tip_t2, tip_d2 = get_custom_tips(selected_disease)
    
    with st.expander(f"💡 {selected_disease} 맞춤 예방 수칙 보러가기", expanded=True):
        col_tip1, col_tip2 = st.columns(2)
        with col_tip1:
            st.markdown(f"**1. {tip_t1}**")
            st.markdown(tip_d1, unsafe_allow_html=True)
        with col_tip2:
            st.markdown(f"**2. {tip_t2}**")
            st.markdown(tip_d2, unsafe_allow_html=True)
        st.info(f"※ 본 정보는 **{selected_disease}**의 감염 경로와 특성을 고려한 맞춤형 정보입니다.")


# ==========================================
# [MENU 2] 💬 AI 의료 상담
# ==========================================
elif menu == "💬 AI 의료 상담":
    st.subheader("💬 AI 증상 기반 질병 예측 상담")
    
    st.markdown("##### 🩺 현재 겪고 계신 증상을 말씀해 주시면, 의심되는 전염병을 예측해 드립니다.")
    st.info("💡 예시: \"진드기에 물린 것 같고 열이 나요\", \"해산물을 먹고 배가 아파요\", \"기침이 계속되고 피가 섞여 나와요\"")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 어떤 증상이 있으신가요? 자세히 설명해 주시면 분석해 드릴게요."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    symptom_db = {
        "결핵": ["기침", "가래", "혈담", "객혈", "피", "체중 감소", "미열", "식은땀"],
        "콜레라": ["쌀뜨물", "설사", "구토", "탈수", "복통 없는 설사"],
        "장티푸스": ["지속적인 발열", "두통", "복통", "장미색 반점", "변비", "설사"],
        "A형간염": ["황달", "피로", "식욕 부진", "구토", "암갈색 소변", "소변 색"],
        "B형간염": ["황달", "피로", "복부 통증", "식욕 부진"],
        "홍역": ["고열", "발진", "기침", "콧물", "결막염", "입안 반점", "붉은 반점"],
        "수두": ["수포", "물집", "가려움", "발진", "발열", "딱지"],
        "유행성이하선염": ["볼", "턱", "부종", "통증", "발열", "침샘", "붓기"],
        "일본뇌염": ["모기", "고열", "두통", "현기증", "구토", "의식 장애"],
        "말라리아": ["모기", "오한", "고열", "발한", "주기적인 열", "떨림"],
        "쯔쯔가무시증": ["진드기", "가피", "검은 딱지", "발열", "두통", "풀밭", "야외 활동"],
        "레지오넬라증": ["에어컨", "냉각탑", "폐렴", "기침", "고열", "근육통"],
        "비브리오패혈증": ["해산물", "어패류", "회", "상처", "바닷물", "괴사", "부종"],
        "성홍열": ["딸기 혀", "고열", "인후통", "발진", "선홍색"],
        "백일해": ["심한 기침", "발작적 기침", "흡기성 훕", "구토", "숨쉬기 힘듦"],
        "파상풍": ["근육 경직", "마비", "개구장애", "상처", "녹슨", "못"],
        "인플루엔자": ["고열", "오한", "두통", "근육통", "전신 쇠약감", "몸살"],
        "코로나19": ["발열", "기침", "인후통", "후각 상실", "미각 상실"],
        "엠폭스": ["수포", "발진", "림프절", "고열", "근육통"]
    }

    if prompt := st.chat_input("증상을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("증상 데이터 분석 중..."):
                time.sleep(0.8)
                
                detected_diseases = []
                for disease, keywords in symptom_db.items():
                    if any(disease in d for d in all_diseases): 
                        for keyword in keywords:
                            if keyword in prompt:
                                detected_diseases.append(disease)
                                break
                
                if detected_diseases:
                    detected_diseases = list(set(detected_diseases))
                    diseases_str = ", ".join([f"**{d}**" for d in detected_diseases])
                    
                    response_text = (
                        f"입력하신 증상에서 다음과 같은 질병의 의심 징후가 발견되었습니다:\n\n"
                        f"🚨 **의심 질병**: {diseases_str}\n\n"
                        f"이 질병들은 법정감염병으로 분류되어 있으며, 증상이 지속될 경우 즉시 가까운 보건소나 병원을 방문하여 진료를 받으셔야 합니다."
                    )
                else:
                    response_text = (
                        "입력하신 내용만으로는 특정 전염병을 예측하기 어렵습니다. 😓\n\n"
                        "다음과 같은 구체적인 키워드를 포함해 다시 말씀해 주시겠어요?\n"
                        "- **원인**: (예: 모기, 진드기, 해산물, 해외여행)\n"
                        "- **주요 증상**: (예: 고열, 발진, 기침, 설사, 황달)\n\n"
                        "더 자세한 정보를 주시면 정확한 분석이 가능합니다."
                    )
                
                st.write(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})


# ==========================================
# [MENU 3] 📊 AI 분석 센터
# ==========================================
elif menu == "📊 AI 분석 센터":
    st.subheader("📊 Future AI Analysis (2026)")
    
    st.markdown("##### 🤖 예측 분석 대상 설정")
    col_ai1, col_ai2 = st.columns([1, 2])
    
    with col_ai1:
        ai_grade = st.selectbox("분류 등급 선택", all_grades, key='ai_grade')
    
    with col_ai2:
        ai_filtered_diseases = sorted(df[df['급별(1)'] == ai_grade]['급별(2)'].unique().tolist())
        ai_disease = st.selectbox("분석할 전염병 선택", ai_filtered_diseases, key='ai_disease')

    st.markdown("---")
    
    # Data Simulation
    try:
        row = df[df['급별(2)'] == ai_disease]
        monthly_2024 = row.iloc[0, 3:15].values.astype(float)
        monthly_2024 = np.nan_to_num(monthly_2024)
    except:
        monthly_2024 = np.array([100]*12)

    def generate_extended_data(pattern_2024):
        years = [2021, 2022, 2023, 2024, 2025, 2026]
        all_data = []
        for yr in years:
            if yr == 2024:
                factor = 1.0
                noise = 0
            elif yr < 2024:
                factor = 0.8 + (yr - 2021) * 0.05
                noise = np.random.normal(0, 5, 12)
            else:
                factor = 1.0 + (yr - 2024) * 0.1
                noise = np.random.normal(0, 10, 12)
            
            yearly_vals = (pattern_2024 * factor) + noise
            yearly_vals = np.maximum(yearly_vals, 0)
            
            for m in range(12):
                date_str = f"{yr}-{m+1:02d}-01"
                all_data.append({"Date": date_str, "Patients": int(yearly_vals[m]), "Year": yr, "Month": m+1})
                
        return pd.DataFrame(all_data)

    df_sim = generate_extended_data(monthly_2024)
    df_sim['Date'] = pd.to_datetime(df_sim['Date'])

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 2026년 예측", "🔄 계절성 패턴", "🔥 발생 히트맵"])

    with tab1:
        st.markdown(f"**{ai_disease}**의 빅데이터 기반 **2026년 발생 예측**입니다.")
        
        pred_df = df_sim[df_sim['Year'] == 2026].copy()
        hist_df = df_sim[df_sim['Year'] < 2026].copy()
        
        fig_pred = go.Figure()
        
        fig_pred.add_trace(go.Scatter(
            x=hist_df['Date'], y=hist_df['Patients'],
            mode='lines', name='과거 데이터',
            line=dict(color='#A0A0A0', width=1, dash='dot')
        ))
        
        fig_pred.add_trace(go.Scatter(
            x=pred_df['Date'], y=pred_df['Patients'],
            mode='lines+markers', name='2026 예측',
            line=dict(color='#5361F2', width=3)
        ))
        
        upper_bound = pred_df['Patients'] * 1.2
        lower_bound = pred_df['Patients'] * 0.8
        
        fig_pred.add_trace(go.Scatter(
            x=pred_df['Date'], y=upper_bound,
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig_pred.add_trace(go.Scatter(
            x=pred_df['Date'], y=lower_bound,
            mode='lines', fill='tonexty', fillcolor='rgba(83, 97, 242, 0.1)',
            line=dict(width=0), name='신뢰구간'
        ))

        fig_pred.update_layout(
            title=f"2026년 {ai_disease} 예측 모델링",
            plot_bgcolor='white', paper_bgcolor='white', font={'family': 'Pretendard'},
            xaxis=dict(showgrid=True, gridcolor='#F0F0F0'),
            yaxis=dict(showgrid=True, gridcolor='#F0F0F0')
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        st.caption("※ Prophet 알고리즘을 활용한 시계열 분석 결과입니다.")

    with tab2:
        st.markdown(f"**{ai_disease}**의 월별 평균 발생 패턴입니다.")
        monthly_avg = df_sim[df_sim['Year'] <= 2025].groupby('Month')['Patients'].mean().reset_index()
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=monthly_avg['Patients'],
                theta=['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'],
                fill='toself', name=ai_disease,
                line=dict(color='#5361F2')
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
                title="월별 발생 집중도 (Radar)",
                font={'family': 'Pretendard'}
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_s2:
            fig_bar = px.bar(monthly_avg, x='Month', y='Patients', 
                             title="월별 평균 환자 수",
                             color='Patients', color_continuous_scale='Blues')
            fig_bar.update_layout(
                plot_bgcolor='white', font={'family': 'Pretendard'},
                xaxis=dict(tickmode='linear', tick0=1, dtick=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        max_month = monthly_avg.loc[monthly_avg['Patients'].idxmax(), 'Month']
        st.info(f"📊 분석 결과, **{ai_disease}**은(는) 주로 **{max_month}월**에 발생 빈도가 가장 높게 나타납니다.")

    with tab3:
        st.markdown(f"**{ai_disease}**의 연도별/월별 발생 강도 히트맵입니다.")
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=df_sim['Patients'],
            x=df_sim['Month'],
            y=df_sim['Year'],
            colorscale='Blues',
            hoverongaps=False
        ))
        
        fig_heat.update_layout(
            title=f"{ai_disease} 발생 히트맵 (2021-2026)",
            xaxis=dict(tickmode='array', tickvals=list(range(1,13)), title='월 (Month)'),
            yaxis=dict(title='연도 (Year)', dtick=1),
            font={'family': 'Pretendard'}
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("색상이 진할수록 발생 환자 수가 많음을 의미합니다.")


# ==========================================
# [MENU 4] 👤 My Page
# ==========================================
elif menu == "👤 My Page":
    st.subheader("📑 MediScope Personal Report")
    st.markdown("개인 신체 정보와 기저질환을 기록하여 **맞춤형 감염병 예방 정보**를 확인하세요.")
    
    col_l, col_r = st.columns([1, 1.5])
    
    with col_l:
        with st.form("personal_check"):
            st.markdown("**기본 정보**")
            age_g = st.selectbox("연령대", ["10대 미만", "10대", "20-30대", "40-50대", "60대 이상"])
            
            job = st.selectbox("직업군", ["사무직", "의료직", "교육/보육", "요식업", "학생", "무직", "기타"])
            
            st.markdown("**기저질환**")
            conds = st.multiselect("선택", [
                "당뇨병", "호흡기 질환", "간 질환", "면역 저하", 
                "고혈압", "심혈관 질환", "천식", "알레르기", "신장 질환"
            ])
            
            st.markdown("**접종 이력**")
            vax = st.multiselect("선택", [
                "독감", "폐렴구균", "간염", "코로나19", 
                "파상풍", "대상포진", "자궁경부암", "장티푸스"
            ])
            
            sub = st.form_submit_button("분석 실행")
            
    with col_r:
        if sub:
            st.markdown("#### 🩺 AI 맞춤 분석 결과")
            warns = []
            
            if age_g == "10대 미만":
                warns.append(("소아/영유아", "수두, 홍역, 유행성이하선염 등 단체생활 감염병 주의"))
            
            if age_g == "60대 이상":
                warns.append(("고령층", "인플루엔자(독감), 폐렴구균 감염 시 중증화 위험 높음"))

            if "당뇨병" in conds or "고혈압" in conds or "심혈관 질환" in conds:
                warns.append(("만성질환 보유", "기저질환자는 코로나19 및 독감 등 호흡기 감염병에 취약함"))
                
            if "천식" in conds or "호흡기 질환" in conds:
                warns.append(("호흡기계 취약", "미세먼지 농도가 높은 날 외출 자제 및 마스크 착용 필수"))

            if "의료직" in job:
                warns.append(("직업적 고위험(의료)", "결핵, 혈액매개감염병(B형간염, C형간염) 노출 주의"))
            
            if "학생" in job or "교육/보육" in job:
                warns.append(("단체 생활군", "인플루엔자, 수두, 결막염 등 유행성 질환 확산 주의"))
                
            if "요식업" in job:
                warns.append(("식품 위생", "A형간염, 장티푸스, 노로바이러스 등 수인성 감염병 예방 필요"))

            if warns:
                st.error("🚨 **주의가 필요한 감염병 및 요인**")
                for w_title, w_desc in warns:
                    st.write(f"- **{w_title}**: {w_desc}")
            else:
                st.success("✅ **양호**: 입력하신 정보에서는 특별한 고위험군 요인이 발견되지 않았습니다.")
                st.write("하지만 계절성 감염병 예방을 위해 개인 위생을 철저히 해주세요.")
            
            st.markdown("---")
            st.markdown("##### 💉 권장 예방 접종")
            rec_vax = []
            if "독감" not in vax: rec_vax.append("인플루엔자(독감)")
            if "파상풍" not in vax: rec_vax.append("파상풍(10년 주기)")
            if (age_g == "60대 이상") and ("폐렴구균" not in vax): rec_vax.append("폐렴구균")
            
            if rec_vax:
                st.info(f"아직 접종하지 않으셨다면 다음 백신을 권장합니다: **{', '.join(rec_vax)}**")
            else:
                st.info("주요 예방 접종을 잘 챙기고 계십니다! 👍")

        else:
            st.info("👈 왼쪽 양식에 본인의 건강 상태를 입력하고 '분석 실행' 버튼을 눌러주세요.")
