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
# 1. 디자인 (CSS) - 깔끔하고 세련된 스타일
# ---------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    html, body, [class*="css"] { font-family: 'Pretendard', sans-serif; }
    .stApp { background-color: #F8F9FA; }
    
    /* 사이드바 */
    [data-testid="stSidebar"] { background-color: white; border-right: 1px solid #eee; }
    
    /* 히어로 배너 */
    .hero-box {
        background: linear-gradient(120deg, #5361F2, #3B4CCA);
        padding: 45px 30px; border-radius: 20px; color: white;
        margin-bottom: 30px; box-shadow: 0 10px 25px rgba(83, 97, 242, 0.3); text-align: center;
    }
    .hero-title { font-size: 2.5rem; font-weight: 800; margin-bottom: 5px; }
    
    /* 카드 스타일 */
    .stat-card {
        background-color: white; border-radius: 18px; padding: 22px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #EAEAEA;
        height: 100%; transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.1); }
    
    /* 팁 & 경고 카드 */
    .tip-card { background-color: #FFFFFF; border-left: 5px solid #5361F2; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 10px; }
    .warning-card { background-color: #FFF5F5; border: 1px solid #FEB2B2; padding: 15px; border-radius: 10px; margin-top: 10px; }
    
    /* 버튼 */
    .stButton > button {
        background-color: #5361F2; color: white; border-radius: 12px;
        height: 52px; font-weight: bold; border: none; width: 100%;
    }
    .stButton > button:hover { background-color: #3845b5; }
    
    /* 채팅 메시지 */
    .chat-bubble { padding: 15px; border-radius: 15px; margin-bottom: 10px; font-size: 0.95rem; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 데이터 로드 (2025년 강제 변환 & 없는 급수 자동 제외)
# ---------------------------------------------------------
@st.cache_data
def get_disease_data():
    file_path = "법정감염병_월별_신고현황_20251201171522.csv"
    empty_df = pd.DataFrame(columns=['ds', 'Class', 'Disease', 'y'])
    
    try:
        # 헤더 무시하고 읽기
        df = pd.read_csv(file_path, header=None, encoding='cp949')
        
        # 데이터 본문 추출 (2행부터)
        df_body = df.iloc[2:].copy()
        
        # 컬럼명 강제 지정 (15개 컬럼 기준)
        df_body = df_body.iloc[:, :15]
        col_names = ['Class', 'Disease', 'Total'] + [str(i) for i in range(1, 13)]
        df_body.columns = col_names
        
        # 소계 제거
        df_body = df_body[df_body['Disease'] != '소계']
        
        # Melt (월별 데이터를 세로로 변환)
        df_melted = df_body.melt(id_vars=['Class', 'Disease'], value_vars=[str(i) for i in range(1,13)], var_name='Month', value_name='Count')
        
        # [핵심] 2023년 데이터를 2025년으로 날짜 변환
        df_melted['ds'] = pd.to_datetime('2025-' + df_melted['Month'].astype(str) + '-01', errors='coerce')
        
        # 숫자 정제 (콤마, 결측치 처리)
        def clean_count(x):
            x = str(x).strip()
            if x in ['-', '', 'nan']: return 0
            try: return int(x.replace(',', ''))
            except: return 0
            
        df_melted['y'] = df_melted['Count'].apply(clean_count)
        
        # 날짜가 제대로 생성된 데이터만 남김
        df_final = df_melted.dropna(subset=['ds'])
        
        return df_final[['ds', 'Class', 'Disease', 'y']]

    except Exception as e:
        return empty_df

data = get_disease_data()

# ---------------------------------------------------------
# 3. 사이드바 (메뉴명 괄호 제거 & 디자인)
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.markdown("""
    <h1 style='color:#5361F2; margin-top:-10px; font-size:24px; font-weight:800;'>MediScope</h1>
    <p style='color:gray; font-size:12px; margin-top:-15px; letter-spacing:1px;'>AI Bio-Surveillance</p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    # 메뉴명 깔끔하게 변경
    menu = st.radio("MENU", [
        "🏠 홈", 
        "💬 AI 의료 상담", 
        "📊 AI 분석 센터", 
        "👤 My Page"
    ])
    st.markdown("---")
    st.caption("Data: 2025.12.01 Updated")
    if st.button("🔄 시스템 리셋"):
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------
# 4. 기능 페이지
# ---------------------------------------------------------

# [PAGE 1] 홈 (2025년 현황)
if menu == "🏠 홈":
    st.markdown("""
        <div class="hero-box">
            <div class="hero-title">MediScope Dashboard</div>
            <div class="hero-desc"><b>2025년</b> 대한민국 감염병 발생 현황 실시간 모니터링</div>
        </div>
    """, unsafe_allow_html=True)
    
    if not data.empty:
        st.subheader("🔥 Monthly Hot Issue (12월 기준)")
        latest = data['ds'].max()
        prev = latest - pd.DateOffset(months=1)
        # 발생 수 0이 아닌 것 중에서 Top 3
        top3 = data[(data['ds'] == latest) & (data['y'] > 0)].sort_values('y', ascending=False).head(3)
        
        if top3.empty:
            st.info("현재 집계된 주요 감염병 데이터가 없습니다.")
        else:
            cols = st.columns(3)
            for idx, (i, row) in enumerate(top3.iterrows()):
                prev_row = data[(data['Disease'] == row['Disease']) & (data['ds'] == prev)]
                diff = row['y'] - prev_row['y'].values[0] if not prev_row.empty else 0
                diff_str = f"▲ {diff:,}" if diff > 0 else f"▼ {abs(diff):,}"
                trend_col = "#E74C3C" if diff > 0 else "#27AE60"
                
                with cols[idx]:
                    st.markdown(f"""<div class="stat-card">
                        <div style="font-weight:bold; color:#E74C3C; font-size:0.9rem;">🚨 {row['Class']} 경보</div>
                        <div style="font-size:1.35rem; font-weight:800; margin:10px 0; color:#2D3748; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{row['Disease']}</div>
                        <div style="font-size:2rem; font-weight:900; color:#5361F2;">{row['y']:,}<span style="font-size:1rem; color:#aaa; font-weight:500;">명</span></div>
                        <div style="color:#666; font-size:0.9rem; background:#F7FAFC; padding:8px; border-radius:8px;">
                            전월 대비 <span style="color:{trend_col}; font-weight:bold;">{diff_str}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.error("데이터 로드 실패")

    st.write(""); st.subheader("🛡️ AI 예방 브리핑")
    c1, c2 = st.columns(2)
    with c1: st.markdown('<div class="tip-card"><div class="tip-title">🫧 올바른 손 씻기</div><div>비누로 30초 이상 씻으면 감염병 70% 예방 가능합니다.</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="tip-card"><div class="tip-title">💉 백신 접종</div><div>독감, 폐렴구균, 대상포진 등 주요 백신 접종을 확인하세요.</div></div>', unsafe_allow_html=True)
    
    st.markdown("### 🔍 Disease Deep-Dive")
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

# [PAGE 2] AI 챗봇 (증상 DB 대폭 강화)
elif menu == "💬 AI 의료 상담":
    st.title("💬 Medi-Bot: Intelligent Triage")
    st.markdown('<div style="background:#FFF3CD; padding:10px; border-radius:5px; color:#856404; font-size:0.9rem; margin-bottom:20px;">⚠️ 본 서비스는 정보 제공 목적이며 의사의 진단을 대신할 수 없습니다.</div>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요. MediScope AI입니다. 증상을 자세히 말씀해주시면(예: 열나고 머리가 아파요, 상한 음식을 먹고 배가 아파요) 2025년 데이터와 대조하여 분석해 드립니다."}]
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
    if prompt := st.chat_input("증상을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # [업그레이드된 증상 키워드 DB]
        symptom_db = {
            "호흡기 감염": {
                "kwd": ["열", "고열", "기침", "가래", "콧물", "코막힘", "인후통", "목이", "오한", "근육통", "숨", "폐렴", "감기", "독감"],
                "cand": ["인플루엔자", "백일해", "폐렴구균", "성홍열", "코로나바이러스감염증-19"],
                "dept": "내과, 이비인후과"
            },
            "소화기(수인성)": {
                "kwd": ["복통", "배가", "설사", "구토", "토할", "메스꺼움", "속이", "체한", "장염", "물설사", "음식"],
                "cand": ["A형간염", "노로바이러스", "장티푸스", "세균성이질", "장출혈성대장균감염증"],
                "dept": "내과, 가정의학과"
            },
            "피부/발진": {
                "kwd": ["발진", "두드러기", "수포", "물집", "가려움", "피부", "따가움", "붉은", "반점"],
                "cand": ["수두", "홍역", "수족구병", "성홍열", "엠폭스"],
                "dept": "피부과, 소아청소년과"
            },
            "발열성/매개체": {
                "kwd": ["벌레", "물린", "산", "풀밭", "야외", "진드기", "모기", "고열", "두통", "오한"],
                "cand": ["쯔쯔가무시증", "말라리아", "일본뇌염", "뎅기열", "신증후군출혈열"],
                "dept": "감염내과, 내과"
            },
            "안과 질환": {
                "kwd": ["눈", "눈곱", "충혈", "따갑", "눈물", "시력"],
                "cand": ["유행성각결막염", "급성출혈성결막염"],
                "dept": "안과"
            }
        }
        
        best_cat = None; max_score = 0
        detected_kwd = []
        
        # 점수 계산 (매칭된 키워드 개수)
        for cat, info in symptom_db.items():
            score = 0
            for k in info["kwd"]:
                if k in prompt:
                    score += 1
                    if k not in detected_kwd: detected_kwd.append(k)
            
            # 카테고리별 가중치 (열은 흔하므로 가중치 낮음)
            if "열" in prompt and cat in ["호흡기 감염", "발열성/매개체"]: score += 0.5
            
            if score > max_score:
                max_score = score
                best_cat = cat
        
        response = ""
        if best_cat and max_score >= 1:
            info = symptom_db[best_cat]
            
            # 데이터 연동: 해당 카테고리 병 중 지금 제일 많이 걸리는 것 찾기
            top_dis = "정보 없음"
            max_val = 0
            
            if not data.empty:
                latest = data['ds'].max()
                for c in info["cand"]:
                    # 포함 검색
                    rows = data[(data['ds'] == latest) & (data['Disease'].str.contains(c))]
                    if not rows.empty:
                        val = rows['y'].sum()
                        if val > max_val:
                            max_val = val
                            top_dis = c
            
            # 만약 데이터에 없으면 후보군 중 첫 번째를 예시로
            if top_dis == "정보 없음": top_dis = info["cand"][0]
            
            response = f\"\"\"
            <div style="background-color:#F0F9FF; padding:15px; border-radius:10px; border-left:5px solid #0077B6;">
                <h4 style="margin:0; color:#0077B6;">📊 AI 증상 분석 리포트</h4>
            </div>
            <br>
            <b>1. 분석 결과:</b> <b>[{best_cat}]</b> 계열 질환이 의심됩니다.<br>
            (감지된 키워드: {', '.join(detected_kwd)})<br><br>
            <b>2. 데이터 역학 (2025 Data):</b><br>
            현재 데이터상 해당 증상군 내에서 <b><span style="color:#E53E3E;">'{top_dis}'</span></b>의 발생 빈도가 가장 높습니다.<br><br>
            <b>3. AI 권고 (Triage):</b><br>
            즉시 가까운 <b>{info['dept']}</b>를 방문하여 전문의의 진료를 받으십시오.
            \"\"\"
        else:
            response = "증상이 명확하지 않습니다. '열이 나고 기침해요', '상한 음식을 먹고 배가 아파요' 처럼 구체적인 상황을 말씀해 주세요."
            
        with st.chat_message("assistant"):
            with st.spinner("증상 데이터 대조 중..."): time.sleep(1); st.markdown(response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})

# [PAGE 3] AI 분석 센터 (2026 예측)
elif menu == "📊 AI 분석 센터":
    st.title("📊 AI Analytics Center (2026 Future)")
    st.markdown("2025년 데이터를 학습하여 **2026년**의 확산 패턴을 시뮬레이션합니다.")
    
    if not data.empty:
        # [수정] 안내 문구 위치 조정
        c1, c2 = st.columns([1, 2])
        with c1: 
            s_class = st.selectbox("분류", sorted(data['Class'].unique()), key='aic')
            s_dis = st.selectbox("질병 선택", data[data['Class'] == s_class]['Disease'].unique(), key='aid')
        with c2: 
            # 위치를 아래로 살짝 내림 (Spacer)
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            st.info(f"💡 **'{s_dis}'**의 2026년 유행 패턴 예측 모델 가동")
        
        df_t = data[data['Disease'] == s_dis].sort_values('ds')
        
        if len(df_t) > 0:
            tab1, tab2, tab3 = st.tabs(["📉 2026년 예측", "🔄 계절성 패턴", "🔥 발생 히트맵"])
            
            with tab1:
                with st.spinner("2026년 시뮬레이션 중..."):
                    m = Prophet(yearly_seasonality=True)
                    m.fit(df_t[['ds', 'y']])
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    fcst = m.predict(future)
                    fcst_2026 = fcst[fcst['ds'] >= '2026-01-01']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_t['ds'], y=df_t['y'], mode='markers+lines', name='2025 실측값', marker=dict(color='gray')))
                    fig.add_trace(go.Scatter(x=fcst_2026['ds'], y=fcst_2026['yhat'], mode='lines', name='2026 AI 예측', line=dict(color='#5361F2', width=3)))
                    fig.update_layout(height=400, plot_bgcolor='white', title=f"2026년 {s_dis} 확산 예측")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # [추가] 전문가 코멘트
                    peak = fcst_2026.loc[fcst_2026['yhat'].idxmax()]
                    st.markdown(f\"\"\"
                    <div style="background:#F8F9FA; padding:20px; border-radius:10px; border:1px solid #E9ECEF;">
                        <h5 style="color:#2C3E50; margin-top:0;">📝 AI Specialist Commentary</h5>
                        <ul style="color:#4A5568; font-size:0.95rem;">
                            <li><b>추세 분석:</b> Prophet 알고리즘 분석 결과, <b>{s_dis}</b>는 2026년 <b>{peak['ds'].strftime('%m월')}</b>에 유행 정점(Peak)에 도달할 것으로 예측됩니다.</li>
                            <li><b>대응 전략:</b> 해당 시기 1개월 전부터 예방 접종 캠페인 및 방역 물품 확보가 필요합니다.</li>
                        </ul>
                    </div>
                    \"\"\", unsafe_allow_html=True)

            with tab2:
                if 'yearly' in fcst.columns:
                    fig_s = px.line(fcst.iloc[:12], x='ds', y='yearly', title='연간 유행 주기 (Seasonality)', color_discrete_sequence=['#27AE60'])
                    fig_s.update_xaxes(tickformat="%b")
                    fig_s.update_layout(plot_bgcolor='white', height=300, xaxis_title="월 (Month)")
                    st.plotly_chart(fig_s, use_container_width=True)
                else: st.warning("계절성 데이터 부족")

            with tab3:
                class_df = data[data['Class'] == s_class].copy()
                class_df['MonthStr'] = class_df['ds'].dt.strftime('%m월')
                piv = class_df.groupby(['Disease', 'MonthStr'])['y'].sum().reset_index()
                fig_h = px.density_heatmap(piv, x='MonthStr', y='Disease', z='y', color_continuous_scale='Redor', title="질병별 발생 강도")
                st.plotly_chart(fig_h, use_container_width=True)

# [PAGE 4] My Page (직업 추가)
elif menu == "👤 My Page":
    st.title("👤 My Health Profile")
    col_p, col_r = st.columns([1, 2])
    with col_p:
        with st.form("mf"):
            st.subheader("내 정보 입력")
            age_g = st.selectbox("연령대", ["10대 미만", "10대", "20-30대", "40-50대", "60대 이상"])
            # [수정] 직업군 추가
            job = st.selectbox("직업군", ["학생", "무직/은퇴", "일반 사무직", "의료 종사자", "교육/보육 종사자", "요식업 종사자", "해외 출장 잦음"])
            st.markdown("**기저질환**")
            conds = st.multiselect("선택", ["당뇨병", "만성 호흡기 질환", "간 질환", "면역 저하", "심혈관 질환"])
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
            
            # 직업 로직
            if "의료" in job: score += 20; warns.append(("의료인", "병원 내 감염 노출 주의"))
            if "학생" in job: score += 10; warns.append(("단체 생활", "학교 내 인플루엔자/수두 유행 주의"))
            if "무직" in job and "60대 이상" in age_g: score += 10; warns.append(("가정 내 감염", "가족 구성원 전파 주의"))
            
            if "독감" in vax: score -= 10
            score = max(0, min(100, score))
            
            c_val = "green" if score < 40 else "orange" if score < 70 else "red"
            st.markdown(f"#### 취약 지수: <span style='color:{c_val}'>{score}점</span>", unsafe_allow_html=True)
            st.progress(score)
            
            for t, m in warns:
                bg = "#FFF5F5" if "고위험" in t else "#F0F9FF"
                icon = "🚨" if "고위험" in t else "💡"
                st.markdown(f'<div style="background:{bg}; padding:15px; margin-bottom:10px; border-radius:5px; border-left:4px solid #3182CE;"><b>{icon} {t}</b><br>{m}</div>', unsafe_allow_html=True)
            
            if not warns: st.success("현재 특별한 위험 요인은 없습니다.")
        else:
            st.info("👈 왼쪽 폼에 정보를 입력하면 AI가 맞춤형 리포트를 생성합니다.")
"""
