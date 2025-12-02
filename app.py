# 1. 초기화 (기존 프로세스 정리)
import os
from pyngrok import ngrok
import streamlit as st

!pkill -f ngrok
!pkill -f streamlit
ngrok.kill()

# 2. 필수 패키지 설치
!pip install streamlit prophet plotly pyngrok

# 3. 앱 코드 작성
code = """
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
import time

# ---------------------------------------------------------
# [필수] 앱 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="MediScope: AI 감염병 플랫폼",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 캐시 초기화 (오류 해결용)
if "clear_cache" not in st.session_state:
    st.cache_data.clear()
    st.session_state.clear_cache = True

# ---------------------------------------------------------
# 1. 디자인 (CSS)
# ---------------------------------------------------------
st.markdown(\"\"\"
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    html, body, [class*="css"] { font-family: 'Pretendard', sans-serif; }
    .stApp { background-color: #F8F9FA; }
    
    /* 사이드바 디자인 */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #EAEAEA;
    }
    
    /* 로고 영역 */
    .logo-box {
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
        border-radius: 15px;
    }
    .brand-title {
        color: #5361F2;
        font-weight: 900;
        font-size: 1.8rem;
        margin: 10px 0 0 0;
        letter-spacing: -1px;
    }
    .brand-sub {
        color: #7F8C8D;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* 히어로 배너 */
    .hero-box {
        background: linear-gradient(120deg, #5361F2, #3B4CCA);
        padding: 40px 30px; border-radius: 20px; color: white;
        margin-bottom: 30px; box-shadow: 0 10px 25px rgba(83, 97, 242, 0.3); text-align: center;
    }
    .hero-title { font-size: 2.5rem; font-weight: 800; margin-bottom: 5px; }
    
    /* 카드 스타일 */
    .stat-card {
        background-color: white; border-radius: 18px; padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #EAEAEA;
        height: 100%; transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.1); }
    
    /* 버튼 */
    .stButton > button {
        background-color: #5361F2; color: white; border-radius: 12px;
        height: 50px; font-weight: bold; border: none; width: 100%;
    }
    .stButton > button:hover { background-color: #3845b5; }
    </style>
    \"\"\", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 데이터 로드 (2025년 강제 변환 & 안전 로딩)
# ---------------------------------------------------------
@st.cache_data
def get_disease_data():
    file_path = "법정감염병_월별_신고현황_20251201171522.csv"
    empty_df = pd.DataFrame(columns=['ds', 'Class', 'Disease', 'y'])
    
    try:
        # 헤더 없이 읽어서 강제 지정 (오류 원천 차단)
        df = pd.read_csv(file_path, header=None, encoding='cp949')
        
        # 2행부터 데이터
        df_body = df.iloc[2:].copy()
        
        # 컬럼 15개만 사용 (등급, 병명, 계, 1~12월)
        df_body = df_body.iloc[:, :15]
        col_names = ['Class', 'Disease', 'Total'] + [str(i) for i in range(1, 13)]
        df_body.columns = col_names
        
        df_body = df_body[df_body['Disease'] != '소계']
        
        # Melt
        df_melted = df_body.melt(id_vars=['Class', 'Disease'], value_vars=[str(i) for i in range(1,13)], var_name='Month', value_name='Count')
        
        # [중요] 2025년으로 날짜 생성
        df_melted['ds'] = pd.to_datetime('2025-' + df_melted['Month'].astype(str) + '-01', errors='coerce')
        
        def clean_count(x):
            if str(x).strip() in ['-', '', 'nan']: return 0
            try: return int(str(x).replace(',', ''))
            except: return 0
            
        df_melted['y'] = df_melted['Count'].apply(clean_count)
        df_final = df_melted.dropna(subset=['ds'])
        
        return df_final[['ds', 'Class', 'Disease', 'y']]

    except Exception as e:
        return empty_df

data = get_disease_data()

# ---------------------------------------------------------
# 3. 사이드바 (디자인 개선)
# ---------------------------------------------------------
with st.sidebar:
    # 깔끔한 의료 아이콘 (이미지 깨짐 방지 위해 온라인 아이콘 사용)
    st.markdown(\"\"\"
    <div class="logo-box">
        <img src="https://cdn-icons-png.flaticon.com/512/2966/2966334.png" width="80">
        <div class="brand-title">MediScope</div>
        <div class="brand-sub">AI Bio-Surveillance</div>
    </div>
    \"\"\", unsafe_allow_html=True)
    
    st.markdown("### 📌 Navigation")
    menu = st.radio("Go to", [
        "🏠 홈 (2025 현황)", 
        "💬 AI 의료 상담 (ChatBot)", 
        "📊 AI 분석 센터 (2026 예측)", 
        "👤 My Page (건강 리포트)"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    st.caption("Data Updated: 2025.12.01")
    if st.button("🔄 시스템 리셋"):
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------
# 4. 기능 페이지
# ---------------------------------------------------------

# [PAGE 1] 홈 (2025년 현황)
if menu == "🏠 홈 (2025 현황)":
    st.markdown(\"\"\"
        <div class="hero-box">
            <div class="hero-title">MediScope Dashboard</div>
            <div class="hero-desc"><b>2025년</b> 대한민국 감염병 발생 현황 실시간 모니터링</div>
        </div>
    \"\"\", unsafe_allow_html=True)
    
    if not data.empty:
        st.subheader("🔥 Monthly Hot Issue (2025년 12월 기준)")
        latest = data['ds'].max()
        prev = latest - pd.DateOffset(months=1)
        top3 = data[data['ds'] == latest].sort_values('y', ascending=False).head(3)
        
        cols = st.columns(3)
        for idx, (i, row) in enumerate(top3.iterrows()):
            prev_row = data[(data['Disease'] == row['Disease']) & (data['ds'] == prev)]
            diff = row['y'] - prev_row['y'].values[0] if not prev_row.empty else 0
            diff_str = f"▲ {diff:,}" if diff > 0 else f"▼ {abs(diff):,}"
            trend_col = "#E74C3C" if diff > 0 else "#27AE60"
            
            with cols[idx]:
                st.markdown(f\"\"\"<div class="stat-card">
                    <div style="font-weight:bold; color:#E74C3C;">🚨 {row['Class']} 경보</div>
                    <div style="font-size:1.3rem; font-weight:800; margin:10px 0;">{row['Disease']}</div>
                    <div style="font-size:2rem; font-weight:900; color:#5361F2;">{row['y']:,}<span style="font-size:1rem; color:#aaa;">명</span></div>
                    <div style="color:#666; font-size:0.9rem;">전월 대비 <span style="color:{trend_col}; font-weight:bold;">{diff_str}</span></div>
                </div>\"\"\", unsafe_allow_html=True)
    else:
        st.error("데이터 로드 실패")

    st.write(""); st.subheader("🛡️ AI 예방 브리핑")
    c1, c2 = st.columns(2)
    with c1: st.info("**🫧 손 씻기:** 감염병의 70%는 손 씻기로 예방 가능합니다.")
    with c2: st.info("**💉 백신 접종:** 독감, 폐렴구균 등 주요 백신 접종을 확인하세요.")
    
    st.markdown("### 🔍 감염병 정밀 분석")
    if not data.empty:
        with st.container():
            c1, c2, c3 = st.columns([1, 2, 0.5])
            with c1: s_class = st.selectbox("등급 분류", sorted(data['Class'].unique()))
            with c2: s_dis = st.selectbox("질병명 검색", data[data['Class'] == s_class]['Disease'].unique())
            with c3: st.write(""); st.write(""); btn = st.button("분석 🚀")
        
        if btn or s_dis:
            st.divider(); target = data[data['Disease'] == s_dis].sort_values('ds')
            c_l, c_r = st.columns([1, 2])
            with c_l: 
                st.markdown(f"#### 🩺 **{s_dis}** 요약")
                st.write(f"**분류:** {s_class}")
                st.write(f"**2025 누적:** {target['y'].sum():,}명")
                st.write(f"**최근 월:** {target.iloc[-1]['y']:,}명")
            with c_r: 
                fig = px.area(target, x='ds', y='y', color_discrete_sequence=['#5361F2'])
                fig.update_layout(plot_bgcolor='white', height=300, xaxis_title=None, yaxis_title="발생 수")
                st.plotly_chart(fig, use_container_width=True)

# [PAGE 2] AI 챗봇 (로직 보강)
elif menu == "💬 AI 의료 상담 (ChatBot)":
    st.title("💬 Medi-Bot: Intelligent Triage")
    st.markdown('<div style="background:#FFF3CD; padding:10px; border-radius:5px; color:#856404; font-size:0.9rem; margin-bottom:20px;">⚠️ 본 서비스는 정보 제공 목적이며 의사의 진단을 대신할 수 없습니다.</div>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요. 증상을 말씀해주시면(예: 열이 나요, 배가 아파요) 2025년 데이터와 대조하여 분석해 드립니다."}]
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
    if prompt := st.chat_input("증상을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # 증상 매칭 로직
        symptom_db = {
            "호흡기 감염": {"kwd": ["열", "기침", "가래", "콧물", "인후통", "목", "오한", "근육통", "숨"], "cand": ["인플루엔자", "백일해", "폐렴구균"], "dept": "내과/이비인후과"},
            "소화기(장염)": {"kwd": ["복통", "설사", "구토", "메스꺼움", "속", "체한", "배가"], "cand": ["A형간염", "노로바이러스", "장티푸스"], "dept": "내과"},
            "피부 질환": {"kwd": ["발진", "두드러기", "수포", "물집", "가려움", "피부"], "cand": ["수두", "홍역", "수족구병"], "dept": "피부과"}
        }
        
        best_cat = None; max_score = 0
        for cat, info in symptom_db.items():
            score = sum(1 for k in info["kwd"] if k in prompt)
            if score > max_score: max_score = score; best_cat = cat
            
        if best_cat:
            info = symptom_db[best_cat]
            top_dis = info["cand"][0]
            if not data.empty:
                latest = data['ds'].max()
                for c in info["cand"]:
                    if not data[(data['ds'] == latest) & (data['Disease'].str.contains(c))].empty:
                        top_dis = c; break
            
            resp = f"분석 결과 **[{best_cat}]** 의심됩니다.\\n데이터상 **{top_dis}** 유행 가능성이 높으니 **{info['dept']}** 진료를 권장합니다."
        else:
            resp = "증상이 명확하지 않습니다. 구체적인 증상(열, 복통 등)을 입력해 주세요."
            
        with st.chat_message("assistant"):
            with st.spinner("분석 중..."): time.sleep(1); st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})

# [PAGE 3] AI 분석 센터 (2026 예측 - 에러 수정완료)
elif menu == "📊 AI 분석 센터 (2026 예측)":
    st.title("📊 AI Analytics Center")
    st.markdown("2025년 데이터를 기반으로 **2026년(Future)** 확산 패턴을 예측합니다.")
    
    if not data.empty:
        c1, c2 = st.columns([1, 2])
        with c1: 
            s_class = st.selectbox("분류", sorted(data['Class'].unique()), key='aic')
            s_dis = st.selectbox("질병 선택", data[data['Class'] == s_class]['Disease'].unique(), key='aid')
        with c2: 
            st.info(f"💡 **{s_dis}**의 2026년 시뮬레이션 모델 가동")
        
        df_t = data[data['Disease'] == s_dis].sort_values('ds')
        
        if len(df_t) > 0:
            tab1, tab2, tab3 = st.tabs(["📉 2026년 예측", "🔄 계절성 패턴", "🔥 발생 히트맵"])
            
            with tab1:
                with st.spinner("2026년 예측 중..."):
                    # [핵심] 1년치 데이터라도 계절성 분석 강제 활성화
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                    m.fit(df_t[['ds', 'y']])
                    
                    # 2026년 예측을 위해 12개월 추가 (월 단위)
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    fcst = m.predict(future)
                    
                    # 2026년 데이터 필터링
                    fcst_2026 = fcst[fcst['ds'] >= '2026-01-01']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_t['ds'], y=df_t['y'], mode='markers+lines', name='2025 실측값', marker=dict(color='gray')))
                    fig.add_trace(go.Scatter(x=fcst_2026['ds'], y=fcst_2026['yhat'], mode='lines', name='2026 AI 예측', line=dict(color='#5361F2', width=3)))
                    fig.update_layout(height=450, plot_bgcolor='white', title=f"2026년 {s_dis} 확산 시뮬레이션")
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # [핵심] 계절성 그래프 복구
                if 'yearly' in fcst.columns:
                    fig_s = px.line(fcst.iloc[:12], x='ds', y='yearly', title='연간 유행 주기 (Seasonality)', color_discrete_sequence=['#27AE60'])
                    fig_s.update_xaxes(tickformat="%b") # 월만 표시
                    fig_s.update_layout(plot_bgcolor='white', height=300, xaxis_title="월 (Month)")
                    st.plotly_chart(fig_s, use_container_width=True)
                else:
                    st.warning("데이터 부족으로 계절성 분석 불가")

            with tab3:
                class_df = data[data['Class'] == s_class].copy()
                class_df['MonthStr'] = class_df['ds'].dt.strftime('%m월')
                piv = class_df.groupby(['Disease', 'MonthStr'])['y'].sum().reset_index()
                fig_h = px.density_heatmap(piv, x='MonthStr', y='Disease', z='y', color_continuous_scale='Redor', title="질병별 발생 강도")
                st.plotly_chart(fig_h, use_container_width=True)

# [PAGE 4] My Page
elif menu == "👤 My Page (건강 리포트)":
    st.title("👤 My Health Profile")
    col_p, col_r = st.columns([1, 2])
    with col_p:
        with st.form("mf"):
            st.subheader("내 정보 입력")
            age_g = st.selectbox("연령대", ["10대 미만", "10대", "20-30대", "40-50대", "60대 이상"])
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
            if "10대 미만" in age_g: score += 20; warns.append(("소아 취약", "수두, 유행성 이하선염 주의"))
            if "60대 이상" in age_g: score += 40; warns.append(("고령층 고위험", "폐렴구균/독감 주의"))
            if "당뇨병" in conds: score += 30; warns.append(("당뇨 고위험", "감염병 합병증 주의"))
            
            score = min(100, score)
            c_val = "green" if score < 40 else "orange" if score < 70 else "red"
            st.markdown(f"#### 취약 지수: <span style='color:{c_val}'>{score}점</span>", unsafe_allow_html=True)
            st.progress(score)
            
            for t, m in warns:
                st.warning(f"**{t}**: {m}")
            if not warns: st.success("현재 특별한 위험 요인은 없습니다.")
"""

# 4. 파일 저장
with open("app.py", "w", encoding='utf-8') as f:
    f.write(code)

# 5. 실행
ngrok.set_auth_token("36Em29EIy3iP3cdFQ20xLYyBudI_27VKZL4nbwuKBhfZCpcJ")
print("MediScope Final Version 실행 중...")
!streamlit run app.py &>/dev/null&

try:
    public_url = ngrok.connect(8501).public_url
    print(f"\n✨ 접속 링크 (그래프/로고/데이터 완벽 해결) ✨\n{public_url}")
except Exception as e:
    print(f"오류: {e}")
