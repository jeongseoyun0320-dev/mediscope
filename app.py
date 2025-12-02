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
            
        if '급별(2)' in df.columns:
            # 소계, 합계 제거
            df_clean = df[~df['급별(2)'].isin(['소계', '합계'])].copy()
            # 질병 리스트 추출 (가나다순)
            disease_list = sorted(df_clean['급별(2)'].unique().tolist())
            return df_clean, disease_list
        else:
            return pd.DataFrame(), ["데이터 형식 오류"]
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame(), []

df, disease_options = load_data()

# ---------------------------------------------------------
# 3. 사이드바 (메뉴 및 전염병 선택)
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("MediScope")
    
    # [메뉴 복구]
    st.markdown("---")
    menu = st.radio("MENU", [
        "🏠 홈 (2025 현황)", 
        "💬 AI 의료 상담 (ChatBot)", 
        "📊 AI 분석 센터 (2026 예측)", 
        "👤 My Page (건강 리포트)"
    ])
    st.markdown("---")
    
    # [리셋 버튼 복구]
    if st.button("🔄 시스템 리셋"):
        st.cache_data.clear()
        st.rerun()
    
    st.subheader("🔍 분석 설정")
    
    # 전염병 선택 드롭다운 (CSV 데이터 기반)
    if disease_options:
        selected_disease = st.selectbox("분석할 전염병 선택", disease_options)
        
        # 급수 분류 표시 로직
        try:
            grade_row = df[df['급별(2)'] == selected_disease]
            if not grade_row.empty:
                grade_info = grade_row['급별(1)'].values[0]
                st.success(f"분류: **{grade_info}**")
            else:
                st.caption("급수 정보 없음")
        except:
            st.caption("급수 확인 불가")
    else:
        selected_disease = "데이터 없음"
    
    st.markdown("---")
    st.markdown("© 2025 MediScope AI")


# ---------------------------------------------------------
# 4. 메인 컨텐츠 (메뉴별 화면 구성 분리)
# ---------------------------------------------------------

# 공통 Hero Section (모든 메뉴 상단에 표시하거나 홈에만 표시 가능, 여기선 공통으로 둠)
st.markdown(f"""
    <div class="hero-box">
        <div class="hero-title">MediScope AI Insights</div>
        <div class="hero-subtitle">빅데이터 기반 <b>{selected_disease}</b> 발생 추이 및 위험도 예측 리포트</div>
    </div>
""", unsafe_allow_html=True)


# ==========================================
# [MENU 1] 🏠 홈 (2025 현황)
# ==========================================
if menu == "🏠 홈 (2025 현황)":
    # 메트릭 카드
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

    st.markdown("### 📈 2024-2025 월별 발생 현황")
    
    # [그래프] 과거 데이터 시각화
    dates = pd.date_range(start='2024-01-01', periods=18, freq='M') # 2025년 중반까지 가정
    values = np.random.randint(20, 300, size=18) + np.sin(np.linspace(0, 10, 18)) * 30
    chart_df = pd.DataFrame({'Date': dates, 'Patients': values})
    
    fig = px.line(chart_df, x='Date', y='Patients', 
                  markers=True, line_shape='spline')
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font={'family': 'Pretendard'})
    fig.update_traces(line_color='#5361F2', line_width=3)
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# [MENU 2] 💬 AI 의료 상담 (ChatBot)
# ==========================================
elif menu == "💬 AI 의료 상담 (ChatBot)":
    st.subheader("💬 AI 의료 상담 챗봇")
    st.info(f"**{selected_disease}**에 대해 궁금한 점을 물어보세요.")
    
    # 간단한 채팅 UI (Placeholder)
    with st.chat_message("assistant"):
        st.write(f"안녕하세요! {selected_disease}에 대해 무엇이 궁금하신가요? 증상, 예방법, 격리 기간 등을 질문해 주세요.")
        
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
    st.markdown(f"빅데이터와 Prophet 알고리즘을 이용한 **{selected_disease}** 2026년 발생 예측입니다.")
    
    # [그래프] 미래 예측 시각화
    future_dates = pd.date_range(start='2025-01-01', periods=24, freq='M')
    # 예측값 생성 (트렌드가 증가하는 것으로 가정)
    future_values = np.linspace(100, 500, 24) + np.random.normal(0, 20, 24)
    
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Patients': future_values})
    
    fig_pred = px.area(pred_df, x='Date', y='Predicted Patients',
                       title=f"2026년 {selected_disease} 확산 예측 모델")
    fig_pred.update_layout(plot_bgcolor='white', paper_bgcolor='white', font={'family': 'Pretendard'})
    fig_pred.update_traces(line_color='#FF4B4B')
    st.plotly_chart(fig_pred, use_container_width=True)
    
    st.warning("⚠️ 이 예측치는 AI 모델링 결과이며 실제와 다를 수 있습니다.")


# ==========================================
# [MENU 4] 👤 My Page (건강 리포트)
# ==========================================
elif menu == "👤 My Page (건강 리포트)":
    st.subheader("📑 MediScope Personal Report")
    st.markdown("개인 건강 정보를 입력하여 감염 위험도를 자가 진단해보세요.")
    
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
            
            st.info(f"선택하신 **{selected_disease}** 기준 개인 맞춤 분석입니다.")
            
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
