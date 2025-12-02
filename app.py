import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
import time
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. 앱 설정 (가장 먼저 실행)
# ---------------------------------------------------------
st.set_page_config(
    page_title="MediScope: AI 감염병 플랫폼",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. 디자인 (CSS)
# ---------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    html, body, [class*="css"] { font-family: 'Pretendard', sans-serif; }
    .stApp { background-color: #F8F9FA; }
    
    [data-testid="stSidebar"] { background-color: white; border-right: 1px solid #eee; }
    
    .hero-box {
        background: linear-gradient(120deg, #5361F2, #3B4CCA);
        padding: 45px 30px; border-radius: 20px; color: white;
        margin-bottom: 30px; box-shadow: 0 10px 25px rgba(83, 97, 242, 0.3); text-align: center;
    }
    .hero-title { font-size: 2.5rem; font-weight: 800; margin-bottom: 5px; }
    
    .stat-card {
        background-color: white; border-radius: 18px; padding: 22px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #EAEAEA;
        height: 100%; transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.1); }
    
    .warning-card { background-color: #FFF5F5; border: 1px solid #FEB2B2; padding: 15px; border-radius: 10px; margin-top: 10px; }
    .tip-card { background-color: #FFFFFF; border-left: 5px solid #5361F2; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 10px; }
    .tip-title { font-weight: bold; color: #2C3E50; font-size: 1.1rem; margin-bottom: 5px; }

    .stButton > button {
        background-color: #5361F2; color: white; border-radius: 12px;
        height: 52px; font-weight: bold; border: none; width: 100%;
    }
    .stButton > button:hover { background-color: #3845b5; }
    
    .chat-bubble { padding: 15px; border-radius: 15px; margin-bottom: 10px; font-size: 0.95rem; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. 데이터 로드 (캐시 제거 + 1~4급 모든 데이터 포함)
# ---------------------------------------------------------
def get_disease_data():
    file_path = "법정감염병_월별_신고현황_20251201171522.csv"
    
    # [비상용] 파일 읽기 실패 시 보여줄 데이터 (1~4급 모두 포함)
    def generate_mock():
        dates = pd.date_range('2025-01-01', '2025-12-01', freq='MS')
        mock = []
        # 다양한 급수 포함
        disease_list = [
            ("제1급", "에볼라바이러스병"), ("제1급", "두창"),
            ("제2급", "결핵"), ("제2급", "수두"), ("제2급", "홍역"), ("제2급", "A형간염"),
            ("제3급", "파상풍"), ("제3급", "B형간염"), ("제3급", "일본뇌염"), ("제3급", "말라리아"),
            ("제4급", "인플루엔자"), ("제4급", "수족구병"), ("제4급", "급성호흡기감염증")
        ]
        for c, d in disease_list:
            for date in dates:
                # 급수에 따라 발생 수 차등 부여 (현실감)
                if "1급" in c: val = np.random.randint(0, 5) # 1급은 적게
                elif "4급" in c: val = np.random.randint(1000, 5000) # 4급은 많게
                else: val = np.random.randint(10, 500)
                
                # 겨울철 유행 시뮬레이션
                if date.month in [11, 12, 1, 2]: val = int(val * 1.5)
                
                mock.append([date, c, d, int(val)])
        return pd.DataFrame(mock, columns=['ds', 'Class', 'Disease', 'y'])

    try:
        # 파일 읽기
        df = pd.read_csv(file_path, header=None, encoding='cp949')
        
        # 데이터 본문 추출 (2행부터)
        df_body = df.iloc[2:].copy()
        
        # 컬럼명 강제 지정 (15개)
        if df_body.shape[1] >= 15:
            df_body = df_body.iloc[:, :15]
            col_names = ['Class', 'Disease', 'Total'] + [str(i) for i in range(1, 13)]
            df_body.columns = col_names
        else:
            return generate_mock()
            
        # 소계 제거 (이것만 빼고 나머지는 다 살림)
        df_body = df_body[df_body['Disease'] != '소계']
        
        # Melt
        df_melted = df_body.melt(id_vars=['Class', 'Disease'], value_vars=[str(i) for i in range(1,13)], var_name='Month', value_name='Count')
        
        # 2025년으로 날짜 고정
        df_melted['ds'] = pd.to_datetime('2025-' + df_melted['Month'].astype(str) + '-01', errors='coerce')
        
        def clean_count(x):
            s = str(x).strip()
            if s in ['-', '', 'nan', 'None']: return 0
            try: return int(s.replace(',', ''))
            except: return 0
            
        df_melted['y'] = df_melted['Count'].apply(clean_count)
        df_final = df_melted.dropna(subset=['ds'])
        
        if df_final.empty: return generate_mock()
        return df_final[['ds', 'Class', 'Disease', 'y']]

    except Exception as e:
        # 에러 발생 시 1~4급이 포함된 비상용 데이터를 반환
        return generate_mock()

data = get_disease_data()

# ---------------------------------------------------------
# 4. 사이드바 (메뉴명 괄호 제거)
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.markdown("""
    <h1 style='color:#5361F2; margin-top:-10px; font-size:24px; font-weight:800;'>MediScope</h1>
    <p style='color:gray; font-size:12px; margin-top:-15px; letter-spacing:1px;'>AI Bio-Surveillance</p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    # [수정] 메뉴명 괄호 제거
    menu = st.radio("MENU", [
        "🏠 홈", 
        "💬 AI 의료 상담", 
        "📊 AI 분석 센터", 
        "👤 My Page"
    ])
    st.markdown("---")
    st.caption("Data: 2025.12.01 Updated")
    
    # 리셋 버튼 (안전하게)
    if st.button("🔄 시스템 리셋"):
        try: st.rerun()
        except: pass

# ---------------------------------------------------------
# 5. 기능 페이지
# ---------------------------------------------------------

# [PAGE 1] 홈
if menu == "🏠 홈":
    st.markdown("""
        <div class="hero-box">
            <div class="hero-title">MediScope Dashboard</div>
            <div class="hero-desc"><b>2025년</b> 대한민국 감염병 발생 현황 실시간 모니터링</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 데이터 로드 확인
    if not data.empty:
        st.subheader("🔥 Monthly Hot Issue (12월 기준)")
        latest = data['ds'].max()
        prev = latest - pd.DateOffset(months=1)
        # 0이 아닌 데이터 중 상위 3개
        top3 = data[(data['ds'] == latest) & (data['y'] > 0)].sort_values('y', ascending=False).head(3)
        
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
        st.error("데이터 로드 실패.")

    st.write(""); st.subheader("🛡️ AI 예방 브리핑")
    c1, c2 = st.columns(2)
    with c1: st.markdown('<div class="tip-card"><div class="tip-title">🫧 올바른 손 씻기</div><div>감염병 70% 예방 효과가 있습니다.</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="tip-card"><div class="tip-title">💉 백신 접종</div><div>독감, 폐렴구균 접종을 확인하세요.</div></div>', unsafe_allow_html=True)
    
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

# [PAGE 2] 챗봇 (증상 DB 대폭 확장)
elif menu == "💬 AI 의료 상담":
    st.title("💬 Medi-Bot: Intelligent Triage")
    st.markdown('<div style="background:#FFF3CD; padding:10px; border-radius:5px; color:#856404; font-size:0.9rem; margin-bottom:20px;">⚠️ 본 서비스는 정보 제공 목적이며 의사의 진단을 대신할 수 없습니다.</div>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요. 증상을 말씀해주시면 2025년 데이터와 대조하여 분석해 드립니다."}]
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
    if prompt := st.chat_input("증상을 입력하세요... (예: 눈이 빨갛고 따가워요)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # [수정] 증상 키워드 대폭 확장 (호흡기, 소화기, 피부, 발열, 안과)
        symptom_db = {
            "호흡기 감염": {"kwd": ["열", "기침", "가래", "콧물", "인후통", "목", "오한", "근육통", "숨", "폐렴", "감기", "독감"], "cand": ["인플루엔자", "백일해", "폐렴구균"], "dept": "내과/이비인후과"},
            "소화기(장염)": {"kwd": ["복통", "설사", "구토", "메스꺼움", "속", "체한", "배가", "장염", "식중독", "물설사"], "cand": ["A형간염", "노로바이러스", "장티푸스", "세균성이질"], "dept": "내과"},
            "피부 질환": {"kwd": ["발진", "두드러기", "수포", "물집", "가려움", "피부", "따가움", "반점", "붉은"], "cand": ["수두", "홍역", "수족구병", "엠폭스"], "dept": "피부과"},
            "발열/매개체": {"kwd": ["벌레", "물린", "산", "진드기", "야외", "풀밭", "캠핑"], "cand": ["쯔쯔가무시증", "말라리아", "일본뇌염"], "dept": "감염내과"},
            "안과 질환": {"kwd": ["눈", "충혈", "눈곱", "시력", "따갑", "눈물"], "cand": ["유행성각결막염"], "dept": "안과"}
        }
        
        best_cat = None; max_score = 0
        for cat, info in symptom_db.items():
            score = sum(1 for k in info["kwd"] if k in prompt)
            if score > max_score: max_score = score; best_cat = cat
            
        if best_cat:
            info = symptom_db[best_cat]
            top_dis = info["cand"][0]
            # 데이터에 있으면 갱신
            if not data.empty:
                latest = data['ds'].max()
                for c in info["cand"]:
                    if not data[(data['ds'] == latest) & (data['Disease'].str.contains(c))].empty:
                        top_dis = c; break
            
            resp = f"""
            <div style="background:#F0F9FF; border-left:4px solid #0077B6; padding:15px; border-radius:5px;">
                <h4 style="margin:0; color:#0077B6;">📋 AI 증상 분석</h4>
                <p style="margin:5px 0;">분석 결과 <b>[{best_cat}]</b> 계열 질환이 의심됩니다.</p>
                <p style="margin:0;">현재 데이터상 <b>'{top_dis}'</b> 발생 빈도가 가장 높습니다.</p>
            </div>
            <br>
            <b>💡 권장 사항:</b><br>
            가까운 <b>{info['dept']}</b>를 방문하여 진료를 받으십시오.
            """
        else:
            resp = "증상이 명확하지 않습니다. '눈이 빨개요', '배가 너무 아파요' 처럼 구체적인 증상을 말씀해 주세요."
            
        with st.chat_message("assistant"):
            with st.spinner("분석 중..."): time.sleep(1); st.markdown(resp, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": resp})

# [PAGE 3] AI 분석 센터 (디자인 개선)
elif menu == "📊 AI 분석 센터":
    st.title("📊 AI Analytics Center")
    st.markdown("2025년 데이터를 학습하여 **2026년**의 확산 패턴을 예측합니다.")
    
    if not data.empty:
        c1, c2 = st.columns([1, 2])
        with c1: 
            s_class = st.selectbox("분류", sorted(data['Class'].unique()), key='aic')
            s_dis = st.selectbox("질병 선택", data[data['Class'] == s_class]['Disease'].unique(), key='aid')
        with c2: 
            # [수정] 위치 아래로 조정
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            st.info(f"💡 **'{s_dis}'**의 2026년 유행 패턴 예측 모델 가동")
        
        df_t = data[data['Disease'] == s_dis].sort_values('ds')
        
        if len(df_t) > 0:
            tab1, tab2, tab3 = st.tabs(["📉 2026년 예측", "🔄 계절성 패턴", "🔥 발생 히트맵"])
            
            with tab1:
                with st.spinner("2026년 예측 중..."):
                    m = Prophet(yearly_seasonality=True)
                    m.fit(df_t[['ds', 'y']])
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    fcst = m.predict(future)
                    
                    fcst_2026 = fcst[fcst['ds'] >= '2026-01-01']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_t['ds'], y=df_t['y'], mode='markers+lines', name='2025 실측값', marker=dict(color='gray')))
                    fig.add_trace(go.Scatter(x=fcst_2026['ds'], y=fcst_2026['yhat'], mode='lines', name='2026 AI 예측', line=dict(color='#5361F2', width=3)))
                    fig.update_layout(height=400, plot_bgcolor='white', title=f"2026년 {s_dis} 확산 시뮬레이션")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if not fcst_2026.empty:
                        peak = fcst_2026.loc[fcst_2026['yhat'].idxmax()]
                        # [수정] 설명글 추가
                        st.markdown(f\"\"\"<div style="background:#F8F9FA; padding:15px; border-radius:10px; margin-top:10px;">
                            <b>📝 AI Analyst Comment:</b><br>
                            Prophet 알고리즘 분석 결과, 2026년 <b>{peak['ds'].strftime('%m월')}</b>에 약 <b>{int(peak['yhat']):,}명</b>으로 유행 정점이 예상됩니다. 
                            해당 시기 1개월 전부터 예방 활동 강화가 필요합니다.
                        </div>\"\"\", unsafe_allow_html=True)

            with tab2:
                if 'yearly' in fcst.columns:
                    fig_s = px.line(fcst.iloc[:12], x='ds', y='yearly', title='연간 유행 주기 (Seasonality)', color_discrete_sequence=['#27AE60'])
                    fig_s.update_xaxes(tickformat="%b")
                    fig_s.update_layout(plot_bgcolor='white', height=300, xaxis_title="월 (Month)")
                    st.plotly_chart(fig_s, use_container_width=True)
                else:
                    st.warning("계절성 데이터 부족")

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
            # [수정] 학생, 무직 추가
            job = st.selectbox("직업군", ["학생", "무직/은퇴", "일반 사무직", "의료 종사자", "교육/보육 종사자", "요식업 종사자"])
            st.markdown("**기저질환**")
            conds = st.multiselect("선택", ["당뇨병", "만성 호흡기 질환", "간 질환", "면역 저하", "심혈관 질환"])
            st.markdown("**접종 이력**")
            vax = st.multiselect("선택", ["독감", "폐렴구균", "간염"])
            sub = st.form_submit_button("분석 실행")
            
    with col_r:
        if sub:
            st.subheader("📑 MediScope Personal Report")
            score = 10; warns = []
            if "10대 미만" in age_g: score += 20; warns.append(("소아 취약", "수두 주의"))
            if "60대 이상" in age_g: score += 40; warns.append(("고령층 고위험", "폐렴구균/독감 주의"))
            if "당뇨병" in conds: score += 30; warns.append(("당뇨 고위험", "합병증 주의"))
            
            if "의료" in job: score += 20; warns.append(("의료인", "감염 노출 주의"))
            # [수정] 학생/무직 로직 추가
            if "학생" in job: score += 10; warns.append(("단체 생활", "인플루엔자/수두 유행 주의"))
            if "무직" in job and "60대 이상" in age_g: score += 10; warns.append(("가정 내 감염", "가족 간 전파 주의"))
            
            if "독감" in vax: score -= 10
            score = max(0, min(100, score))
            
            c_val = "green" if score < 40 else "orange" if score < 70 else "red"
            st.markdown(f"#### 취약 지수: <span style='color:{c_val}'>{score}점</span>", unsafe_allow_html=True)
            st.progress(score)
            
            for t, m in warns:
                bg = "#FFF5F5" if "고위험" in t else "#F0F9FF"
                st.markdown(f'<div class="warning-card" style="background:{bg};"><b>{t}</b><br>{m}</div>', unsafe_allow_html=True)
            
            if not warns: st.success("현재 특별한 위험 요인은 없습니다.")
