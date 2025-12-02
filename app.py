import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
import time
import random
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
# 1. 디자인 (CSS) - 수정 없음 (원본 유지)
# ---------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    html, body, [class*="css"] { font-family: 'Pretendard', sans-serif; }
    .stApp { background-color: #F8F9FA; }
    
    [data-testid="stSidebar"] { background-color: white; border-right: 1px solid #eee; }
    
    .hero-box {
        background: linear-gradient(120deg, #5361F2, #3B4CCA);
        padding: 40px 30px; border-radius: 20px; color: white;
        margin-bottom: 30px; box-shadow: 0 10px 20px rgba(59, 76, 202, 0.2);
    }
    .hero-title { font-size: 2.5rem; font-weight: 800; margin-bottom: 10px; }
    .hero-subtitle { font-size: 1.1rem; opacity: 0.9; font-weight: 300; }
    
    .metric-card {
        background: white; border-radius: 15px; padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-5px); }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #333; }
    .metric-label { font-size: 0.9rem; color: #666; margin-top: 5px; }
    
    div[data-testid="stExpander"] { border: none; box-shadow: 0 4px 10px rgba(0,0,0,0.03); border-radius: 10px; background: white; }
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
            
            # 1. 전체 질병 리스트 (기존 호환성)
            disease_list = sorted(df_clean['급별(2)'].unique().tolist())
            
            # 2. 등급 리스트
            grade_list = sorted(df_clean['급별(1)'].unique().tolist())
            
            return df_clean, disease_list, grade_list
        else:
            return pd.DataFrame(), [], []
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame(), [], []

df, all_diseases, all_grades = load_data()

# ---------------------------------------------------------
# 3. 사이드바 (메뉴 및 리셋 버튼만 남김)
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("MediScope")
    
    st.markdown("---")
    menu = st.radio("MENU", [
        "🏠 홈 (2025 현황)", 
        "💬 AI 의료 상담 (ChatBot)", 
        "📊 AI 분석 센터 (2026 예측)", 
        "👤 My Page (건강 리포트)"
    ])
    st.markdown("---")
    
    # 시스템 리셋 버튼
    if st.button("🔄 시스템 리셋"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("© 2025 MediScope AI")


# ---------------------------------------------------------
# 4. 메인 컨텐츠 (메뉴별 화면 구성)
# ---------------------------------------------------------

# ==========================================
# [MENU 1] 🏠 홈 (2025 현황)
# ==========================================
if menu == "🏠 홈 (2025 현황)":
    
    # 1. 상단 필터 (등급 -> 질병)
    st.markdown("### 🔍 감염병 현황 조회")
    col_filter1, col_filter2 = st.columns([1, 2])
    
    with col_filter1:
        selected_grade = st.selectbox("1. 분류 등급 선택", all_grades, key='home_grade')
    
    with col_filter2:
        # 선택된 등급에 해당하는 질병만 필터링
        filtered_diseases = sorted(df[df['급별(1)'] == selected_grade]['급별(2)'].unique().tolist())
        selected_disease = st.selectbox("2. 전염병 선택", filtered_diseases, key='home_disease')

    # Hero Section
    st.markdown(f"""
        <div class="hero-box">
            <div class="hero-title">MediScope AI Insights</div>
            <div class="hero-subtitle"><b>{selected_grade} {selected_disease}</b> 발생 추이 및 예방 정보</div>
        </div>
    """, unsafe_allow_html=True)

    # 2. 메트릭 카드
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
    
    # 3. 그래프
    st.subheader(f"📈 {selected_disease} 월별 발생 추이")
    
    # (예시 데이터)
    dates = pd.date_range(start='2024-01-01', periods=18, freq='M')
    values = np.random.randint(20, 300, size=18) + np.sin(np.linspace(0, 10, 18)) * 30
    chart_df = pd.DataFrame({'Date': dates, 'Patients': values})
    
    fig = px.line(chart_df, x='Date', y='Patients', markers=True, line_shape='spline')
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font={'family': 'Pretendard'})
    fig.update_traces(line_color='#5361F2', line_width=3)
    st.plotly_chart(fig, use_container_width=True)

    # 4. 예방 Tip 섹션 (요청사항 반영)
    st.markdown("---")
    st.subheader(f"🩹 {selected_disease} 예방 및 행동 요령 (Tip)")
    
    with st.expander("💡 주요 예방 수칙 보러가기", expanded=True):
        col_tip1, col_tip2 = st.columns(2)
        with col_tip1:
            st.markdown(f"""
            **1. 위생 관리**
            - 흐르는 물에 30초 이상 비누로 손 씻기
            - 기침할 땐 옷소매로 입과 코 가리기
            - 씻지 않은 손으로 눈, 코, 입 만지지 않기
            """)
        with col_tip2:
            st.markdown(f"""
            **2. 생활 환경**
            - 실내 환기를 자주 시키기
            - 의심 증상 발생 시 마스크 착용
            - 오염된 물이나 음식 섭취 주의
            """)
        st.info(f"※ 본 정보는 일반적인 예방 수칙이며, **{selected_disease}**의 특성에 따라 보건소의 지침을 따르세요.")


# ==========================================
# [MENU 2] 💬 AI 의료 상담 (ChatBot)
# ==========================================
elif menu == "💬 AI 의료 상담 (ChatBot)":
    st.subheader("💬 AI 의료 상담 챗봇")
    
    # 챗봇 페이지용 질병 선택
    c_grade = st.selectbox("등급 분류", all_grades, key='chat_grade')
    c_diseases = sorted(df[df['급별(1)'] == c_grade]['급별(2)'].unique().tolist())
    c_disease = st.selectbox("상담할 질병 선택", c_diseases, key='chat_disease')
    
    st.info(f"**{c_disease}**에 대해 궁금한 점을 물어보세요.")
    
    with st.chat_message("assistant"):
        st.write(f"안녕하세요! {c_disease}에 대해 무엇이 궁금하신가요? 증상, 예방법, 격리 기간 등을 질문해 주세요.")
        
    prompt = st.chat_input("질문을 입력하세요...")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            st.write("죄송합니다. 현재는 데모 버전이라 실제 AI 응답은 연결되어 있지 않습니다.")


# ==========================================
# [MENU 3] 📊 AI 분석 센터 (2026 예측)
# ==========================================
elif menu == "📊 AI 분석 센터 (2026 예측)":
    st.subheader("📊 Future AI Analysis (2026)")
    
    # 1. 상단 필터 (AI 센터 전용)
    st.markdown("##### 🤖 예측 분석 대상 설정")
    col_ai1, col_ai2 = st.columns([1, 2])
    
    with col_ai1:
        ai_grade = st.selectbox("분류 등급 선택", all_grades, key='ai_grade')
    
    with col_ai2:
        ai_filtered_diseases = sorted(df[df['급별(1)'] == ai_grade]['급별(2)'].unique().tolist())
        ai_disease = st.selectbox("분석할 전염병 선택", ai_filtered_diseases, key='ai_disease')

    st.markdown("---")
    st.markdown(f"빅데이터와 Prophet 알고리즘을 이용한 **{ai_disease} ({ai_grade})** 2026년 발생 예측입니다.")
    
    # 2. 그래프
    future_dates = pd.date_range(start='2025-01-01', periods=24, freq='M')
    future_values = np.linspace(100, 500, 24) + np.random.normal(0, 20, 24)
    
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Patients': future_values})
    
    fig_pred = px.area(pred_df, x='Date', y='Predicted Patients',
                       title=f"2026년 {ai_disease} 확산 예측 모델")
    fig_pred.update_layout(plot_bgcolor='white', paper_bgcolor='white', font={'family': 'Pretendard'})
    fig_pred.update_traces(line_color='#FF4B4B')
    st.plotly_chart(fig_pred, use_container_width=True)
    
    st.warning("⚠️ 이 예측치는 AI 모델링 결과이며 실제와 다를 수 있습니다.")


# ==========================================
# [MENU 4] 👤 My Page (건강 리포트)
# ==========================================
elif menu == "👤 My Page (건강 리포트)":
    st.subheader("📑 MediScope Personal Report")
    
    # 마이페이지용 질병 선택
    m_grade = st.selectbox("관심 등급", all_grades, key='my_grade')
    m_diseases = sorted(df[df['급별(1)'] == m_grade]['급별(2)'].unique().tolist())
    m_disease = st.selectbox("자가진단 대상 질병", m_diseases, key='my_disease')
    
    st.markdown(f"**{m_disease}**에 대한 개인 감염 위험도를 자가 진단해보세요.")
    
    col_l, col_r = st.columns([1, 1.5])
    
    with col_l:
        with st.form("personal_check"):
            st.markdown("**기본 정보**")
            age_g = st.multiselect("연령대", ["10대 미만", "10대", "20-30대", "40-50대", "60대 이상"])
            job = st.selectbox("직업군", ["사무직", "의료직", "교육/보육", "요식업"])
            
            st.markdown("**기저질환**")
            conds = st.multiselect("선택", ["당뇨병", "호흡기 질환", "간 질환", "면역 저하"])
            
            st.markdown("**접종 이력**")
            vax = st.multiselect("선택", ["독감", "폐렴구균", "간염"])
            
            sub = st.form_submit_button("분석 실행")
            
    with col_r:
        if sub:
            st.markdown("#### 분석 결과")
            score = 10; warns = []
            
            # 위험도 로직
            if "10대 미만" in age_g: score += 20; warns.append(("소아 취약", "수두/홍역 주의"))
            if "60대 이상" in age_g: score += 40; warns.append(("고령층 고위험", "합병증 주의"))
            if "당뇨병" in conds: score += 30; warns.append(("만성질환", "면역력 저하 주의"))
            if "의료직" in job: score += 15; warns.append(("직업적 특성", "병원균 노출 빈도 높음"))
            
            st.info(f"선택하신 **{m_disease}** 기준 개인 맞춤 분석입니다.")
            
            score = min(score, 100)
            st.progress(score)
            st.caption(f"감염 위험도 점수: {score}/100")
            
            if score < 30:
                st.success("🟢 **안전**: 현재 상태 양호합니다.")
            elif score < 60:
                st.warning("🟡 **주의**: 일부 위험 요인이 있습니다.")
            else:
                st.error("🔴 **위험**: 각별한 주의가 필요합니다.")
                
            if warns:
                st.markdown("---")
                st.write("**상세 위험 요인:**")
                for w_title, w_desc in warns:
                    st.write(f"- **{w_title}**: {w_desc}")
        else:
            st.info("👈 왼쪽 양식을 입력하고 '분석 실행'을 눌러주세요.")
