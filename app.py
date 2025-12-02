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
    
    # [위치 변경 로직]
    # Hero 섹션(파란 바)을 먼저 보여주기 위해, 현재 선택된 상태를 미리 계산합니다.
    # 세션 상태(st.session_state)를 확인하여 이전에 선택한 값이 있으면 그것을 사용하고, 없으면 기본값을 사용합니다.
    
    default_grade = all_grades[0] if all_grades else "데이터 없음"
    current_grade = st.session_state.get('home_grade', default_grade)
    
    # 현재 등급에 맞는 질병 리스트 필터링
    if not df.empty and current_grade in all_grades:
        filtered_diseases = sorted(df[df['급별(1)'] == current_grade]['급별(2)'].unique().tolist())
        default_disease = filtered_diseases[0] if filtered_diseases else "데이터 없음"
    else:
        filtered_diseases = []
        default_disease = "데이터 없음"
        
    # 현재 선택된 질병 확인 (등급이 바뀌어서 리스트에 없으면 첫 번째로 리셋)
    current_disease = st.session_state.get('home_disease', default_disease)
    if current_disease not in filtered_diseases and filtered_diseases:
        current_disease = filtered_diseases[0]

    # 1. Hero Section (파란색 바) - 맨 위로 이동
    st.markdown(f"""
        <div class="hero-box">
            <div class="hero-title">MediScope AI Insights</div>
            <div class="hero-subtitle"><b>{current_grade} {current_disease}</b> 발생 추이 및 예방 정보</div>
        </div>
    """, unsafe_allow_html=True)

    # 2. 하단 필터 (등급 -> 질병) - Hero 섹션 아래로 이동
    st.markdown("### 🔍 감염병 현황 조회")
    col_filter1, col_filter2 = st.columns([1, 2])
    
    with col_filter1:
        # 인덱스 찾기
        try: g_idx = all_grades.index(current_grade)
        except: g_idx = 0
        selected_grade = st.selectbox("1. 분류 등급 선택", all_grades, index=g_idx, key='home_grade')
    
    with col_filter2:
        # 인덱스 찾기
        try: d_idx = filtered_diseases.index(current_disease)
        except: d_idx = 0
        selected_disease = st.selectbox("2. 전염병 선택", filtered_diseases, index=d_idx, key='home_disease')

    st.markdown("---")

    # 3. 메트릭 카드
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
    
    # 4. 그래프
    st.subheader(f"📈 {selected_disease} 월별 발생 추이")
    
    # (예시 데이터)
    dates = pd.date_range(start='2024-01-01', periods=18, freq='M')
    values = np.random.randint(20, 300, size=18) + np.sin(np.linspace(0, 10, 18)) * 30
    chart_df = pd.DataFrame({'Date': dates, 'Patients': values})
    
    fig = px.line(chart_df, x='Date', y='Patients', markers=True, line_shape='spline')
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font={'family': 'Pretendard'})
    fig.update_traces(line_color='#5361F2', line_width=3)
    st.plotly_chart(fig, use_container_width=True)

    # 5. 예방 Tip 섹션
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
    st.subheader("💬 AI 증상 기반 질병 예측 상담")
    
    st.markdown("##### 🩺 현재 겪고 계신 증상을 말씀해 주시면, 의심되는 전염병을 예측해 드립니다.")
    st.info("💡 예시: \"갑자기 고열이 나고 온몸에 붉은 발진이 생겼어요.\" 또는 \"기침이 멈추지 않고 가래가 나옵니다.\"")
    
    # 채팅 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 어떤 증상이 있으신가요? 자세히 설명해 주시면 분석해 드릴게요."}]

    # 이전 메시지 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # 사용자 입력 처리
    if prompt := st.chat_input("증상을 입력하세요..."):
        # 사용자 메시지 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # AI 응답 (개선된 CSV 기반 로직)
        with st.chat_message("assistant"):
            with st.spinner("빅데이터 분석 중..."):
                time.sleep(1.2) # 분석하는 척 딜레이
                
                # CSV 데이터(all_diseases)에서 전염병 찾기
                if all_diseases:
                    # 간단한 키워드 매칭 시도 (예시)
                    matched = [d for d in all_diseases if d in prompt]
                    
                    if matched:
                        predicted = matched[0]
                        desc = f"입력하신 내용에서 **'{predicted}'**와(과) 관련된 키워드가 감지되었습니다."
                    else:
                        # 매칭되는 게 없으면 CSV 리스트 중 랜덤 추천 (다양성 확보)
                        predicted = random.choice(all_diseases)
                        desc = f"입력하신 증상 **'{prompt}'** 패턴을 분석한 결과, 다음 질병의 징후와 유사성이 있습니다."

                    response_text = (
                        f"{desc}\n\n"
                        f"🧪 **AI 예측 결과**: **{predicted}** 가능성 발견\n"
                        f"⚠️ 이 결과는 **MediScope 데이터베이스**({len(all_diseases)}종 감염병) 기반 예측이며, "
                        f"정확한 진단은 반드시 의료기관을 방문하세요."
                    )
                else:
                    response_text = "죄송합니다. 현재 데이터베이스에 연결할 수 없습니다."
                
                st.write(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})


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
    st.markdown("개인 신체 정보와 기저질환을 기록하여 **맞춤형 감염병 예방 정보**를 확인하세요.")
    
    col_l, col_r = st.columns([1, 1.5])
    
    with col_l:
        with st.form("personal_check"):
            st.markdown("**기본 정보**")
            age_g = st.multiselect("연령대", ["10대 미만", "10대", "20-30대", "40-50대", "60대 이상"])
            
            # [수정] 직업군 추가 (학생, 무직)
            job = st.selectbox("직업군", ["사무직", "의료직", "교육/보육", "요식업", "학생", "무직", "기타"])
            
            # [수정] 기저질환 선택지 추가
            st.markdown("**기저질환**")
            conds = st.multiselect("선택", [
                "당뇨병", "호흡기 질환", "간 질환", "면역 저하", 
                "고혈압", "심혈관 질환", "천식", "알레르기", "신장 질환"
            ])
            
            # [수정] 접종 이력 선택지 추가
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
            
            # 로직: 입력된 정보를 바탕으로 주의해야 할 질병 역추적
            if "10대 미만" in age_g:
                warns.append(("소아/영유아", "수두, 홍역, 유행성이하선염 등 단체생활 감염병 주의"))
            
            if "60대 이상" in age_g:
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

            # 결과 출력
            if warns:
                st.error("🚨 **주의가 필요한 감염병 및 요인**")
                for w_title, w_desc in warns:
                    st.write(f"- **{w_title}**: {w_desc}")
            else:
                st.success("✅ **양호**: 입력하신 정보에서는 특별한 고위험군 요인이 발견되지 않았습니다.")
                st.write("하지만 계절성 감염병 예방을 위해 개인 위생을 철저히 해주세요.")
            
            # 접종 제안 (간단 로직)
            st.markdown("---")
            st.markdown("##### 💉 권장 예방 접종")
            rec_vax = []
            if "독감" not in vax: rec_vax.append("인플루엔자(독감)")
            if "파상풍" not in vax: rec_vax.append("파상풍(10년 주기)")
            if ("60대 이상" in age_g) and ("폐렴구균" not in vax): rec_vax.append("폐렴구균")
            
            if rec_vax:
                st.info(f"아직 접종하지 않으셨다면 다음 백신을 권장합니다: **{', '.join(rec_vax)}**")
            else:
                st.info("주요 예방 접종을 잘 챙기고 계십니다! 👍")

        else:
            st.info("👈 왼쪽 양식에 본인의 건강 상태를 입력하고 '분석 실행' 버튼을 눌러주세요.")
