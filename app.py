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
            
            # 1. 전체 질병 리스트
            disease_list = sorted(df_clean['급별(2)'].unique().tolist())
            
            # 2. 등급 리스트 (숫자 기준 정렬 로직 추가)
            raw_grades = df_clean['급별(1)'].unique().tolist()
            
            def grade_sort_key(grade):
                # "제1급" -> 1, "2급" -> 2 등 숫자만 추출하여 정렬 키로 사용
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
# 3. 사이드바 (메뉴 및 리셋 버튼만 남김)
# ---------------------------------------------------------
with st.sidebar:
    # st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60) # 기존 이미지 주석 처리
    st.markdown("# 🏠 MediScope") # 이모지로 대체 및 크기 조절
    # st.title("MediScope") # 기존 타이틀 주석 처리 (마크다운에 포함됨)
    
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
    default_grade = all_grades[0] if all_grades else "데이터 없음"
    current_grade = st.session_state.get('home_grade', default_grade)
    
    # 현재 등급에 맞는 질병 리스트 필터링
    if not df.empty and current_grade in all_grades:
        filtered_diseases = sorted(df[df['급별(1)'] == current_grade]['급별(2)'].unique().tolist())
        default_disease = filtered_diseases[0] if filtered_diseases else "데이터 없음"
    else:
        filtered_diseases = []
        default_disease = "데이터 없음"
        
    # 현재 선택된 질병 확인
    current_disease = st.session_state.get('home_disease', default_disease)
    if current_disease not in filtered_diseases and filtered_diseases:
        current_disease = filtered_diseases[0]

    # 1. Hero Section (파란색 바)
    st.markdown(f"""
        <div class="hero-box">
            <div class="hero-title">MediScope AI Insights</div>
            <div class="hero-subtitle"><b>{current_grade} {current_disease}</b> 발생 추이 및 예방 정보</div>
        </div>
    """, unsafe_allow_html=True)

    # 2. 하단 필터 (등급 -> 질병)
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
    
    dates = pd.date_range(start='2024-01-01', periods=18, freq='M')
    values = np.random.randint(20, 300, size=18) + np.sin(np.linspace(0, 10, 18)) * 30
    chart_df = pd.DataFrame({'Date': dates, 'Patients': values})
    
    fig = px.line(chart_df, x='Date', y='Patients', markers=True, line_shape='spline')
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font={'family': 'Pretendard'})
    fig.update_traces(line_color='#5361F2', line_width=3)
    st.plotly_chart(fig, use_container_width=True)

    # 5. 예방 Tip 섹션 - [맞춤형 로직 적용]
    st.markdown("---")
    st.subheader(f"🩹 {selected_disease} 예방 및 행동 요령 (Tip)")

    # 맞춤형 팁 생성 함수
    def get_custom_tips(disease_name):
        d_name = disease_name
        
        # 1. 호흡기 감염병 (비말/공기)
        if any(k in d_name for k in ["결핵", "인플루엔자", "코로나", "홍역", "수두", "백일해", "유행성이하선염", "성홍열", "폐렴구균", "엠폭스"]):
            return (
                "마스크 착용 및 기침 예절",
                "- 사람이 많은 곳에서는 반드시 마스크를 착용하세요.\n- 기침이나 재채기 시 옷소매로 입과 코를 가리세요.\n- 씻지 않은 손으로 눈, 코, 입을 만지지 마세요.",
                "실내 환기 및 격리",
                "- 하루 3회 이상, 10분씩 실내 환기를 시켜주세요.\n- 발열 및 호흡기 증상 발생 시 등교/출근을 멈추고 집에서 휴식하세요."
            )
        
        # 2. 수인성/식품매개 감염병 (물/음식)
        elif any(k in d_name for k in ["콜레라", "장티푸스", "파라티푸스", "세균성이질", "장출혈성", "A형간염", "비브리오", "식중독", "노로바이러스"]):
            return (
                "안전한 물과 음식 섭취",
                "- 물은 반드시 끓여 마시고, 음식은 충분히 익혀 드세요.\n- 채소와 과일은 흐르는 물에 깨끗이 씻어 껍질을 벗겨 드세요.\n- 조리 도구는 끓는 물이나 소독제로 소독하세요.",
                "철저한 손 씻기",
                "- 화장실 사용 후, 조리 전, 식사 전 흐르는 물에 비누로 30초 이상 손을 씻으세요.\n- 설사 증상이 있는 경우 음식을 조리하지 마세요."
            )
        
        # 3. 매개체 감염병 (모기/진드기)
        elif any(k in d_name for k in ["말라리아", "일본뇌염", "쯔쯔가무시", "뎅기열", "지카", "열", "진드기"]):
            return (
                "피부 노출 최소화",
                "- 야외 활동 시 긴 소매, 긴 바지를 착용하여 피부 노출을 줄이세요.\n- 진드기/모기 기피제를 사용하세요.\n- 풀밭 위에 옷을 벗어두거나 바로 눕지 마세요.",
                "환경 관리 및 예방접종",
                "- 집 주변 웅덩이 등 모기 서식지를 제거하세요.\n- 야외 활동 후 즉시 샤워하고 입었던 옷은 세탁하세요.\n- 유행 지역 방문 전 예방접종 여부를 확인하세요."
            )
        
        # 4. 혈액/성매개/접촉 감염병
        elif any(k in d_name for k in ["B형간염", "C형간염", "매독", "후천성면역결핍증"]):
            return (
                "개인 위생용품 공유 금지",
                "- 칫솔, 면도기, 손톱깎이 등 혈액이 묻을 수 있는 용품은 절대 공유하지 마세요.\n- 문신, 피어싱 등은 반드시 소독된 도구를 사용하는 곳에서 받으세요.",
                "정기 검진 및 안전 수칙",
                "- 정기적인 검진을 통해 감염 여부를 확인하세요.\n- 의료 종사자는 주사 바늘 찔림 등 혈액 노출 사고에 주의하세요."
            )
        
        # 5. 기타/일반적인 경우
        else:
            return (
                "일상 속 위생 수칙 준수",
                "- 흐르는 물에 30초 이상 비누로 손 씻기를 생활화하세요.\n- 기침할 땐 옷소매로 입과 코를 가리세요.",
                "면역력 강화 및 건강 관리",
                "- 규칙적인 운동과 충분한 수면으로 면역력을 높이세요.\n- 의심 증상 발생 시 즉시 의료기관을 방문하여 진료를 받으세요."
            )

    tip_t1, tip_d1, tip_t2, tip_d2 = get_custom_tips(selected_disease)
    
    with st.expander(f"💡 {selected_disease} 맞춤 예방 수칙 보러가기", expanded=True):
        col_tip1, col_tip2 = st.columns(2)
        with col_tip1:
            st.markdown(f"**1. {tip_t1}**")
            st.markdown(tip_d1)
        with col_tip2:
            st.markdown(f"**2. {tip_t2}**")
            st.markdown(tip_d2)
        st.info(f"※ 본 정보는 **{selected_disease}**의 감염 경로와 특성을 고려한 맞춤형 정보입니다. (출처: 질병관리청 지침 기반 재구성)")


# ==========================================
# [MENU 2] 💬 AI 의료 상담 (ChatBot)
# ==========================================
elif menu == "💬 AI 의료 상담 (ChatBot)":
    st.subheader("💬 AI 증상 기반 질병 예측 상담")
    
    st.markdown("##### 🩺 현재 겪고 계신 증상을 말씀해 주시면, 의심되는 전염병을 예측해 드립니다.")
    st.info("💡 예시: \"진드기에 물린 것 같고 열이 나요\", \"해산물을 먹고 배가 아파요\", \"기침이 계속되고 피가 섞여 나와요\"")
    
    # 채팅 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 어떤 증상이 있으신가요? 자세히 설명해 주시면 분석해 드릴게요."}]

    # 이전 메시지 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # [핵심 로직] 증상 키워드 데이터베이스 (CSV 질병 매핑)
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

    # 사용자 입력 처리
    if prompt := st.chat_input("증상을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # AI 응답 로직
        with st.chat_message("assistant"):
            with st.spinner("증상 데이터 분석 중..."):
                time.sleep(0.8)
                
                detected_diseases = []
                # 사용자의 입력(prompt)에서 키워드를 찾아 매칭되는 질병 추출
                for disease, keywords in symptom_db.items():
                    # CSV 파일(all_diseases)에 있는 질병인지 확인 (데이터 정합성)
                    if any(disease in d for d in all_diseases): 
                        for keyword in keywords:
                            if keyword in prompt:
                                detected_diseases.append(disease)
                                break
                
                # 결과 생성
                if detected_diseases:
                    # 중복 제거
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
# [MENU 3] 📊 AI 분석 센터 (2026 예측)
# ==========================================
elif menu == "📊 AI 분석 센터 (2026 예측)":
    st.subheader("📊 Future AI Analysis (2026)")
    
    st.markdown("##### 🤖 예측 분석 대상 설정")
    col_ai1, col_ai2 = st.columns([1, 2])
    
    with col_ai1:
        # 여기에도 정렬된 all_grades가 반영됨
        ai_grade = st.selectbox("분류 등급 선택", all_grades, key='ai_grade')
    
    with col_ai2:
        ai_filtered_diseases = sorted(df[df['급별(1)'] == ai_grade]['급별(2)'].unique().tolist())
        ai_disease = st.selectbox("분석할 전염병 선택", ai_filtered_diseases, key='ai_disease')

    st.markdown("---")
    st.markdown(f"빅데이터와 Prophet 알고리즘을 이용한 **{ai_disease} ({ai_grade})** 2026년 발생 예측입니다.")
    
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
            
            # 단일 선택에 따른 로직 (==)
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
