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
# 1. 디자인 (CSS) - 수정 없음
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
# 2. 데이터 로드 및 전처리 (수정됨: 안전한 로딩)
# ---------------------------------------------------------
@st.cache_data
def load_disease_data():
    csv_file = '법정감염병_월별_신고현황_20251201171222.csv'
    try:
        # CSV 구조상 두 번째 줄(index 1)이 실제 헤더(급별, 계, 1월...)입니다.
        # 인코딩 오류 방지를 위해 utf-8 시도 후 실패 시 cp949로 재시도합니다.
        try:
            df = pd.read_csv(csv_file, header=1, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, header=1, encoding='cp949')
        
        # '급별(2)' 컬럼이 실제 질병명을 담고 있습니다. '소계'나 '합계'는 제외합니다.
        if '급별(2)' in df.columns:
            # 소계 행 제외 및 유니크한 질병명 추출
            disease_list = df[~df['급별(2)'].isin(['소계', '합계'])]['급별(2)'].unique().tolist()
            # 가나다순 정렬
            disease_list.sort()
            return df, disease_list
        else:
            # 컬럼을 못 찾을 경우 기본값
            return df, ["A형간염", "결핵", "수두"]
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return pd.DataFrame(), ["데이터 로드 실패"]

# 데이터 불러오기
df_raw, disease_options = load_disease_data()


# ---------------------------------------------------------
# 3. 사이드바 및 메인 헤더
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("MediScope")
    st.markdown("---")
    
    st.subheader("🔍 분석 설정")
    
    # [수정됨] CSV 파일에 있는 모든 전염병 리스트를 드롭다운에 적용
    selected_disease = st.selectbox(
        "분석할 전염병 선택",
        options=disease_options
    )
    
    st.info(f"선택됨: **{selected_disease}**")
    st.markdown("---")
    st.markdown("© 2025 MediScope AI")

# 메인 헤더 (Hero Section)
st.markdown(f"""
    <div class="hero-box">
        <div class="hero-title">MediScope AI Insights</div>
        <div class="hero-subtitle">빅데이터 기반 <b>{selected_disease}</b> 발생 추이 및 위험도 예측 리포트</div>
    </div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 4. 메인 컨텐츠 (대시보드)
# ---------------------------------------------------------

# (예시 데이터 생성 로직 - 실제 CSV 데이터가 연동되면 이 부분은 df_raw를 필터링하여 사용하도록 고도화 가능)
# 현재는 UI 구조 유지를 위해 랜덤 데이터 생성 부분은 유지하되, 선택된 질병명을 반영합니다.

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-value">1,240명</div>
            <div class="metric-label">이번 달 예상 환자 수</div>
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

st.markdown("### 📈 발생 추이 및 AI 예측")

# 탭 구성
tab1, tab2 = st.tabs(["📊 시계열 분석", "📑 개인화 리포트"])

with tab1:
    # Prophet 예측 시뮬레이션 (간단한 예시)
    dates = pd.date_range(start='2024-01-01', periods=24, freq='M')
    values = np.random.randint(50, 500, size=24) + np.sin(np.linspace(0, 10, 24)) * 50
    
    chart_df = pd.DataFrame({'Date': dates, 'Patients': values})
    
    fig = px.line(chart_df, x='Date', y='Patients', 
                  title=f"{selected_disease} 월별 환자 수 추이",
                  markers=True, line_shape='spline')
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font={'family': 'Pretendard'})
    fig.update_traces(line_color='#5361F2', line_width=3)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("#### 🩺 개인별 감염 위험도 자가진단")
    col_l, col_r = st.columns([1, 2])
    
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
            st.subheader("📑 MediScope Personal Report")
            score = 10; warns = []
            
            # 간단한 로직 예시
            if "10대 미만" in age_g: score += 20; warns.append(("소아 취약", "수두 주의"))
            if "60대 이상" in age_g: score += 40; warns.append(("고령층 고위험", "폐렴구균/독감 주의"))
            if "당뇨병" in conds: score += 30; warns.append(("당뇨 고위험", "합병증 주의"))
            if "의료직" in job: score += 15; warns.append(("의료인", "감염 노출 주의"))
            
            # 선택된 질병에 대한 더미 코멘트 추가
            st.info(f"선택하신 **{selected_disease}**에 대한 개인 맞춤 분석 결과입니다.")
            
            # 점수에 따른 시각화
            score = min(score, 100)
            st.progress(score)
            st.caption(f"감염 위험도 점수: {score}/100")
            
            if warns:
                for w_title, w_desc in warns:
                    st.warning(f"**{w_title}**: {w_desc}")
            else:
                st.success("현재 입력하신 정보로는 고위험 요인이 발견되지 않았습니다.")
        else:
            st.info("왼쪽 양식을 입력하고 '분석 실행' 버튼을 눌러주세요.")
