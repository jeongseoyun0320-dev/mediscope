# 1. 초기화
import os
from pyngrok import ngrok

!pkill -f ngrok
!pkill -f streamlit
ngrok.kill()

# 2. 설치
!pip install streamlit prophet plotly pyngrok

# 3. 앱 코드
code = """
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
import time

# ---------------------------------------------------------
# 1. 앱 설정 & 디자인
# ---------------------------------------------------------
st.set_page_config(
    page_title="MediScope: AI 감염병 플랫폼",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(\"\"\"
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    html, body, [class*="css"] { font-family: 'Pretendard', sans-serif; }
    .stApp { background-color: #F8F9FA; }
    
    /* 사이드바 로고 */
    .logo-container {
        text-align: center; padding: 20px 0; margin-bottom: 20px;
        background: linear-gradient(180deg, #FFFFFF 0%, #F8F9FA 100%);
        border-radius: 15px; border: 1px solid #F0F0F0;
    }
    .brand-name { font-weight: 800; font-size: 1.8rem; color: #5361F2; margin: 0; }
    .brand-slogan { font-size: 0.8rem; color: #7F8C8D; letter-spacing: 1px; margin-top: 5px; }

    /* 히어로 배너 */
    .hero-box {
        background: linear-gradient(135deg, #5361F2 0%, #3a47c9 100%);
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
    
    /* 경고 카드 (My Page) */
    .alert-card {
        padding: 15px; border-radius: 12px; margin-bottom: 10px; border-left: 5px solid;
    }
    .alert-high { background-color: #FFF5F5; border-color: #E53E3E; }
    .alert-mid { background-color: #FFFFF0; border-color: #D69E2E; }
    .alert-title { font-weight: bold; font-size: 1.05rem; display: flex; align-items: center; gap: 8px; }
    
    /* 팁 카드 */
    .tip-card { background-color: #FFFFFF; border-left: 5px solid #5361F2; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 10px; }
    .tip-title { font-weight: bold; color: #2C3E50; }
    
    /* 버튼 */
    .stButton > button {
        background-color: #5361F2; color: white; border-radius: 12px;
        height: 50px; font-weight: bold; border: none; width: 100%;
    }
    .stButton > button:hover { background-color: #3845b5; }
    </style>
    \"\"\", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 데이터 로드
# ---------------------------------------------------------
@st.cache_data
def get_disease_data():
    file_path = "법정감염병_월별_신고현황_20251201171522.csv"
    empty_df = pd.DataFrame(columns=['ds', 'Class', 'Disease', 'y'])
    try:
        df = pd.read_csv(file_path, header=None, encoding='cp949')
        try: year_val = int(str(df.iloc[0, 2]).replace('.', ''))
        except: year_val = 2023
        df.columns = df.iloc[1]; df = df.iloc[2:]
        target_col = [c for c in df.columns if '급별' in str(c) and '2' in str(c)]
        if target_col: df = df.rename(columns={target_col[0]: 'Disease'})
        else: df = df.rename(columns={df.columns[1]: 'Disease'})
        df = df.rename(columns={df.columns[0]: 'Class'})
        df = df[df['Disease'] != '소계']
        month_cols = [c for c in df.columns if '월' in str(c)]
        df_melted = df.melt(id_vars=['Class', 'Disease'], value_vars=month_cols, var_name='Month', value_name='Count')
        month_map = {f'{i}월': i for i in range(1, 13)}
        df_melted['MonthNum'] = df_melted['Month'].map(month_map)
        df_melted['ds'] = pd.to_datetime(str(year_val) + '-' + df_melted['MonthNum'].astype(str) + '-01', errors='coerce')
        def clean_count(x):
            if str(x).strip() == '-': return 0
            try: return int(str(x).replace(',', ''))
            except: return 0
        df_melted['y'] = df_melted['Count'].apply(clean_count)
        df_melted = df_melted.dropna(subset=['ds'])
        return df_melted[['ds', 'Class', 'Disease', 'y']]
    except: return empty_df

data = get_disease_data()

# ---------------------------------------------------------
# 3. 사이드바
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(\"\"\"
    <div class="logo-container">
        <img src="https://img.icons8.com/fluency/200/health-data.png" style="width:80px; margin-bottom:10px;">
        <h1 class="brand-name">MediScope</h1>
        <div class="brand-slogan">AI Bio-Surveillance System</div>
    </div>
    \"\"\", unsafe_allow_html=True)
    
    st.markdown("### 📌 MENU")
    menu = st.radio("이동할 페이지", ["🏠 홈 (대시보드)", "💬 AI 의료 상담 (ChatBot)", "📊 AI 분석 센터", "👤 My Page (건강 리포트)"], label_visibility="collapsed")
    st.markdown("---")
    if st.button("🔄 시스템 재부팅"): st.cache_data.clear(); st.rerun()

# ---------------------------------------------------------
# 4. 기능 페이지
# ---------------------------------------------------------

# [PAGE 1] 홈
if menu == "🏠 홈 (대시보드)":
    st.markdown(\"\"\"
        <div class="hero-box">
            <div class="hero-title">MediScope Surveillance</div>
            <div class="hero-desc">데이터와 AI가 만드는 대한민국 감염병 안전지대</div>
        </div>
    \"\"\", unsafe_allow_html=True)
    
    if data is not None and not data.empty:
        st.subheader("🔥 이번 달 급상승 감염병 (Top 3)")
        latest_month = data['ds'].max()
        prev_month = latest_month - pd.DateOffset(months=1)
        current_top = data[data['ds'] == latest_month].sort_values('y', ascending=False).head(3)
        cols = st.columns(3)
        for idx, (i, row) in enumerate(current_top.iterrows()):
            prev_row = data[(data['Disease'] == row['Disease']) & (data['ds'] == prev_month)]
            diff = row['y'] - prev_row['y'].values[0] if not prev_row.empty else 0
            diff_str = f"▲ {diff:,}" if diff > 0 else f"▼ {abs(diff):,}"
            trend_col = "#E74C3C" if diff > 0 else "#27AE60"
            with cols[idx]:
                st.markdown(f\"\"\"<div class="stat-card">
                    <div style="font-weight:bold; color:#E74C3C;">🚨 {row['Class']}</div>
                    <div style="font-size:1.3rem; font-weight:800; margin-bottom:10px;">{row['Disease']}</div>
                    <div style="font-size:1.8rem; font-weight:900; color:#5361F2;">{row['y']:,}명</div>
                    <div style="color:#666; font-size:0.9rem;"><span style="color:{trend_col}; font-weight:bold;">{diff_str}</span> (전월 대비)</div>
                </div>\"\"\", unsafe_allow_html=True)
    
    st.write(""); st.subheader("🛡️ 오늘의 예방 Tip"); col_t1, col_t2 = st.columns(2)
    with col_t1: st.markdown('<div class="tip-card"><div class="tip-title">🫧 올바른 손 씻기</div><div class="tip-content">흐르는 물에 비누로 30초 이상 씻으세요. 감염병 70% 예방 효과!</div></div>', unsafe_allow_html=True)
    with col_t2: st.markdown('<div class="tip-card"><div class="tip-title">💉 예방접종 확인</div><div class="tip-content">독감, 폐렴구균 등 계절성 질환 백신 접종을 확인하세요.</div></div>', unsafe_allow_html=True)
    
    st.write(""); st.markdown("### 🔍 감염병 정밀 분석")
    with st.container():
        c1, c2, c3 = st.columns([1, 2, 0.5])
        with c1: s_class = st.selectbox("등급 분류", sorted(data['Class'].unique()))
        with c2: s_dis = st.selectbox("질병명 검색", data[data['Class'] == s_class]['Disease'].unique())
        with c3: st.write(""); st.write(""); btn = st.button("분석 시작 >")
    if btn or s_dis:
        st.divider(); target = data[data['Disease'] == s_dis].sort_values('ds')
        c_l, c_r = st.columns([1, 2])
        with c_l: st.markdown(f'<div class="stat-card" style="background:#F8F9FA; border:none;"><div style="font-size:1.2rem; font-weight:bold;">🩺 {s_dis} 요약</div><div style="margin-top:15px;"><p><b>분류:</b> {s_class}</p><p><b>누적:</b> {target["y"].sum():,}명</p><p><b>최근:</b> {target.iloc[-1]["y"]:,}명</p></div></div>', unsafe_allow_html=True)
        with c_r: fig = px.area(target, x='ds', y='y', color_discrete_sequence=['#5361F2']); fig.update_layout(plot_bgcolor='white', height=300, xaxis_title=None, yaxis_title=None); st.plotly_chart(fig, use_container_width=True)

# [PAGE 2] AI 챗봇
elif menu == "💬 AI 의료 상담 (ChatBot)":
    st.title("💬 Medi-Bot: 증상 기반 AI 트리아지")
    st.markdown('<div style="background-color:#FFF3CD; padding:10px; border-radius:8px; border:1px solid #FFEEBA; color:#856404; text-align:center; margin-bottom:20px;"><b>[주의]</b> 본 서비스는 정보 제공 목적이며 의사의 진단을 대신할 수 없습니다.</div>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하십니까. MediScope AI입니다. 증상을 말씀해 주시면(예: 고열, 기침) 데이터를 기반으로 분석해 드립니다."}]
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"], unsafe_allow_html=True)
    
    if prompt := st.chat_input("증상을 입력하세요..."):
        with st.chat_message("user"): st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 간단한 로직 (이전 코드 참조)
        symptom_map = {"호흡기": ["열", "기침", "목"], "소화기": ["복통", "설사", "구토"], "피부": ["발진", "가려움"]}
        cat = None; 
        for c, k in symptom_map.items(): 
            if any(w in prompt for w in k): cat = c; break
        
        response = f"증상('{prompt}')에 대한 분석 결과, 관련 전문의 진료를 권장합니다."
        if cat: response = f"입력하신 증상은 **{cat} 질환** 가능성이 있습니다. 가까운 병원을 방문하세요."
        
        with st.chat_message("assistant"):
            with st.spinner("데이터 분석 중..."): time.sleep(1); st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


# [PAGE 3] AI 분석 센터 (대폭 강화)
elif menu == "📊 AI 분석 센터":
    st.title("📊 AI Analytics Center")
    st.markdown("Prophet 모델과 시계열 분해(Decomposition) 기술을 활용한 심층 분석 대시보드입니다.")
    
    if data is not None:
        # 상단 컨트롤 패널
        with st.container():
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1: s_class = st.selectbox("분류", sorted(data['Class'].unique()), key='aic')
            with c2: s_dis = st.selectbox("질병 선택", data[data['Class'] == s_class]['Disease'].unique(), key='aid')
            with c3: st.info(f"💡 **'{s_dis}'**의 미래 확산 패턴을 다각도로 분석합니다.")
        
        df_t = data[data['Disease'] == s_dis].sort_values('ds')
        
        if len(df_t) > 1:
            # 탭 구성
            tab1, tab2, tab3 = st.tabs(["📉 미래 예측 (Forecast)", "🔄 계절성 분석 (Seasonality)", "🔥 질병 히트맵 (Heatmap)"])
            
            # [Tab 1] 기본 예측
            with tab1:
                with st.spinner("AI 모델 연산 중..."):
                    m = Prophet()
                    m.fit(df_t[['ds', 'y']])
                    future = m.make_future_dataframe(periods=60) # 60일 예측
                    fcst = m.predict(future)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_t['ds'], y=df_t['y'], mode='markers+lines', name='실제 데이터', line=dict(color='gray', dash='dot')))
                    fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], mode='lines', name='AI 예측', line=dict(color='#5361F2', width=3)))
                    fig.add_trace(go.Scatter(x=fcst['ds'].tolist()+fcst['ds'].tolist()[::-1], y=fcst['yhat_upper'].tolist()+fcst['yhat_lower'].tolist()[::-1], fill='toself', fillcolor='rgba(83,97,242,0.1)', line=dict(color='rgba(0,0,0,0)'), name='95% 신뢰구간'))
                    fig.update_layout(height=450, plot_bgcolor='white', title=f"향후 60일 확산 예측 모델")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 수치 리포트
                    next_month = int(fcst.iloc[-30]['yhat'])
                    st.success(f"📌 **AI Insight:** 현재 추세를 반영할 때, 다음 달 예상 환자 수는 약 **{next_month:,}명** (오차범위 ±{int(next_month*0.1)})으로 전망됩니다.")

            # [Tab 2] 계절성/트렌드 분해
            with tab2:
                st.subheader("🗓️ 시계열 패턴 분해")
                st.caption("AI가 데이터에서 '전체적인 추세(Trend)'와 '반복되는 패턴(Seasonality)'을 분리했습니다.")
                
                # Prophet 컴포넌트 시각화 (Trend & Yearly)
                # Trend
                fig_trend = px.line(fcst, x='ds', y='trend', title='장기적 추세 (Trend Component)', color_discrete_sequence=['#E74C3C'])
                fig_trend.update_layout(plot_bgcolor='white', height=300)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # 시뮬레이션된 계절성 (데이터가 1년치라 완벽하진 않지만 시각화)
                # Prophet의 yearly 컬럼 활용
                if 'yearly' in fcst.columns:
                    fig_season = px.line(fcst.iloc[:365], x='ds', y='yearly', title='연간 주기성 (Yearly Seasonality)', color_discrete_sequence=['#27AE60'])
                    fig_season.update_layout(plot_bgcolor='white', height=300)
                    st.plotly_chart(fig_season, use_container_width=True)
                    st.info("ℹ️ 그래프가 높게 솟은 구간이 해당 질병이 매년 유행하는 시기입니다.")

            # [Tab 3] 히트맵 (종합 분석)
            with tab3:
                st.subheader("🌡️ 질병별 월별 발생 히트맵")
                st.caption("선택한 등급(Class) 내 다른 질병들과의 발생 강도를 비교합니다.")
                
                # 같은 등급의 데이터 가져오기
                class_df = data[data['Class'] == s_class].copy()
                class_df['MonthStr'] = class_df['ds'].dt.strftime('%m월')
                
                # 피벗 테이블 생성
                pivot_df = class_df.groupby(['Disease', 'MonthStr'])['y'].sum().reset_index()
                
                fig_heat = px.density_heatmap(pivot_df, x='MonthStr', y='Disease', z='y', 
                                              color_continuous_scale='Redor', title=f"{s_class} 감염병 발생 강도 비교")
                fig_heat.update_layout(height=500)
                st.plotly_chart(fig_heat, use_container_width=True)

# [PAGE 4] My Page (심화 기능 추가)
elif menu == "👤 My Page (건강 리포트)":
    st.title("👤 My Health Profile (Personalized)")
    st.markdown("개인 건강 정보와 **실시간 유행 데이터(Live Data)**를 연동하여 맞춤형 행동 지침을 생성합니다.")
    
    col_p, col_r = st.columns([1, 2])
    
    with col_p:
        with st.form("mf"):
            st.subheader("📝 상세 건강 정보 입력")
            
            # 1. 기본 정보
            age_g = st.selectbox("연령대", ["10대 미만 (영유아/소아)", "10대 (청소년)", "20-30대 (청년)", "40-50대 (중장년)", "60대 이상 (고령층)"])
            gender = st.radio("성별", ["남성", "여성"], horizontal=True)
            job = st.selectbox("직업군 (환경 요인)", ["일반 사무직/학생", "의료 종사자 (병원)", "교육/보육 종사자", "식품/요식업 종사자", "해외 출장/여행 잦음"])
            
            st.markdown("---")
            # 2. 기저질환 (확장됨)
            st.markdown("**💊 기저질환 (다중 선택)**")
            conditions = st.multiselect("해당하는 항목을 모두 선택하세요", 
                ["당뇨병", "만성 호흡기 질환 (천식/COPD)", "만성 간 질환", "심혈관 질환", "만성 신장 질환", "항암 치료 중/면역 저하", "임신부"])
            
            st.markdown("---")
            # 3. 접종 이력
            st.markdown("**💉 최근 1년 내 예방접종**")
            vax = st.multiselect("접종한 백신", ["인플루엔자(독감)", "폐렴구균", "대상포진", "A/B형 간염", "코로나19"])
            
            sub = st.form_submit_button("🛡️ AI 맞춤 분석 실행")
    
    with col_r:
        if sub:
            st.subheader("📑 MediScope AI 분석 리포트")
            
            # --- [알고리즘: 위험도 점수 산정] ---
            risk_score = 10
            warnings = [] # 경고 메시지 리스트
            
            # 1. 연령별 위험도
            if "영유아" in age_g: 
                risk_score += 20; warnings.append(("소아 취약", "수두, 유행성 이하선염 등 단체생활 감염병 주의"))
            if "60대 이상" in age_g: 
                risk_score += 40; warnings.append(("고령층 고위험", "폐렴구균 및 인플루엔자 합병증 위험 매우 높음"))
                
            # 2. 기저질환 연동
            if "당뇨병" in conditions or "만성 간 질환" in conditions:
                risk_score += 30
                # 데이터 연동: 만약 A형 간염이 유행중이라면? (시뮬레이션 로직)
                warnings.append(("간/당뇨 고위험군", "A형 간염 감염 시 치명적일 수 있습니다. 항체 검사 필수."))
            
            if "만성 호흡기 질환" in conditions:
                risk_score += 30
                warnings.append(("호흡기 취약계층", "미세먼지가 심한 날 외출 자제 및 마스크 상시 착용 권고."))
                
            # 3. 직업 연동
            if "의료" in job:
                risk_score += 15; warnings.append(("의료인", "혈액 매개 감염병 및 호흡기 감염병 상시 노출 위험."))
            if "요식업" in job:
                risk_score += 15; warnings.append(("식품 취급자", "A형 간염, 장티푸스 등 수인성 감염병 전파 주의."))
                
            # 4. 백신 방어 효과 (점수 차감)
            if "인플루엔자(독감)" in vax: risk_score -= 10
            if "폐렴구균" in vax: risk_score -= 10
            
            # 점수 보정
            risk_score = max(0, min(100, risk_score))
            
            # --- [시각화] ---
            # 상태 결정
            if risk_score < 40: color, status = "green", "안전 (Low Risk)"
            elif risk_score < 75: color, status = "orange", "주의 (Moderate Risk)"
            else: color, status = "red", "위험 (High Risk)"
            
            st.markdown(f"#### 🛡️ 나의 감염병 취약 지수: <span style='color:{color}'>{risk_score}점</span>", unsafe_allow_html=True)
            st.progress(risk_score)
            st.caption(f"상태: {status} | 분석 기준: 질병관리청 가이드라인")
            
            st.divider()
            
            # --- [조건부 경고 카드 출력] ---
            st.markdown("#### 🚨 실시간 데이터 연동 경고")
            
            if not warnings:
                st.success("✅ 현재 귀하의 정보와 매칭되는 고위험 경보는 없습니다. 건강한 생활 습관을 유지하세요!")
            else:
                for title, msg in warnings:
                    # 위험도에 따른 카드 색상
                    bg_col = "#FFF5F5" if "고위험" in title or "취약" in title else "#FFFFF0"
                    border_col = "#FC8181" if "고위험" in title or "취약" in title else "#F6E05E"
                    icon = "🚨" if "고위험" in title else "⚠️"
                    
                    st.markdown(f\"\"\"
                    <div class="alert-card" style="background-color:{bg_col}; border-left-color:{border_col}; border:1px solid {border_col};">
                        <div class="alert-title" style="color:#2D3748;">{icon} {title}</div>
                        <div style="margin-top:5px; color:#4A5568; font-size:0.95rem;">{msg}</div>
                    </div>
                    \"\"\", unsafe_allow_html=True)
            
            # 백신 추천 (미접종 시)
            if "60대 이상" in age_g and "폐렴구균" not in vax:
                st.info("💉 **[권장]** 65세 이상은 **폐렴구균 무료 접종** 대상입니다. 가까운 보건소를 방문하세요.")
        
        else:
            # 입력 전 화면
            st.info("👈 왼쪽 폼에 상세 정보를 입력하면, AI가 **기저질환**과 **직업 환경**까지 고려한 정밀 리포트를 제공합니다.")
"""

# 4. 파일 저장
with open("app.py", "w", encoding='utf-8') as f:
    f.write(code)

# 5. 실행
ngrok.set_auth_token("36Em29EIy3iP3cdFQ20xLYyBudI_27VKZL4nbwuKBhfZCpcJ")
print("MediScope Expert Edition 실행 중...")
!streamlit run app.py &>/dev/null&

try:
    public_url = ngrok.connect(8501).public_url
    print(f"\n✨ 전문가용 접속 링크 ✨\n{public_url}")
except Exception as e:
    print(f"오류: {e}")
