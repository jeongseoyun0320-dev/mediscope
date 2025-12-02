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
# 1. 디자인 (CSS) - 기존 유지
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
        margin-bottom: 30px; box-shadow: 0 10px 20px rgba(83, 97, 242, 0.2);
    }
    .hero-title { font-size: 42px; font-weight: 800; margin-bottom: 10px; }
    .hero-subtitle { font-size: 18px; opacity: 0.9; font-weight: 300; }
    
    .kpi-card {
        background: white; border-radius: 16px; padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-5px); }
    .kpi-value { font-size: 32px; font-weight: 800; color: #333; }
    .kpi-label { font-size: 14px; color: #888; margin-top: 5px; }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: white; border-radius: 10px; padding: 10px 20px;
        border: 1px solid #eee; color: #555; font-weight: 600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #5361F2; color: white; border: none;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 데이터 로드 (수정됨: 모든 급수 데이터 로드)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    file_path = "법정감염병_월별_신고현황_20251201171222.csv"
    
    try:
        # 헤더가 2번째 줄(Index 1)에 위치함 ("급별(1)", "급별(2)", "계", "1월"...)
        df = pd.read_csv(file_path, header=1)
        
        # 컬럼명 앞뒤 공백 제거
        df.columns = [c.strip() for c in df.columns]
        
        # 이름 변경 매핑
        rename_map = {}
        for c in df.columns:
            if "급별(1)" in c: rename_map[c] = "Class"
            elif "급별(2)" in c: rename_map[c] = "Disease"
        df = df.rename(columns=rename_map)
        
        # '소계' 행 제외 (개별 질병만 분석)
        df = df[df['Disease'] != '소계']
        
        # 데이터 정제 (콤마 제거 및 숫자 변환)
        month_cols = [f"{i}월" for i in range(1, 13)]
        for col in month_cols:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(",", "").astype(int)
                else:
                    df[col] = df[col].fillna(0).astype(int)
        
        # '계' 컬럼도 동일하게 처리
        if '계' in df.columns:
            if df['계'].dtype == object:
                df['계'] = df['계'].astype(str).str.replace(",", "").astype(int)
            else:
                df['계'] = df['계'].fillna(0).astype(int)
                    
        return df
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

df = load_data()

# ---------------------------------------------------------
# 3. 사이드바
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063823.png", width=80)
    st.title("MediScope")
    st.write("2024 법정감염병 통합 분석")
    st.markdown("---")
    if not df.empty:
        st.info(f"🧬 분석 가능한 질병 수: **{len(df['Disease'].unique())}개**")
        # 디버깅용: 로드된 급수 목록 표시 (사용자 확인용)
        loaded_classes = sorted(df['Class'].unique())
        st.caption(f"감지된 급수: {', '.join(loaded_classes)}")
    st.caption("Last Updated: 2025.12.02")

# ---------------------------------------------------------
# 4. 메인 헤더
# ---------------------------------------------------------
st.markdown("""
    <div class="hero-box">
        <div class="hero-title">MediScope Analytics</div>
        <div class="hero-subtitle">공공데이터 기반 AI 감염병 예측 및 개인화 리포트 솔루션</div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. 탭 구성
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Disease Deep-Dive", "🤖 AI Analytics Center", "📑 Personal Report"])

# =========================================================
# TAB 1: Disease Deep-Dive (수정됨: 모든 급수 선택 가능)
# =========================================================
with tab1:
    st.subheader("📊 질병별 상세 분석 (Disease Deep-Dive)")
    
    if not df.empty:
        # [수정] 데이터프레임에 있는 모든 Class를 가져와서 정렬
        all_classes = sorted(df['Class'].unique())
        
        c1, c2, c3 = st.columns([1, 2, 4])
        
        with c1:
            # 동적으로 가져온 급수 목록을 표시
            selected_class = st.selectbox("등급(Class) 선택", all_classes)
        
        # 선택된 급수에 해당하는 질병만 필터링
        filtered_by_class = df[df['Class'] == selected_class]
        disease_list = sorted(filtered_by_class['Disease'].unique())
        
        with c2:
            target_disease = st.selectbox("질병명(Disease) 선택", disease_list)
            
        # 선택된 데이터 추출
        if target_disease:
            row = filtered_by_class[filtered_by_class['Disease'] == target_disease].iloc[0]
            
            # 월별 데이터 추출
            month_cols = [f"{i}월" for i in range(1, 13)]
            values = [row[c] for c in month_cols]
            total_cnt = row["계"] if "계" in row else sum(values)
            
            st.markdown("---")
            
            # KPI 카드
            kc1, kc2, kc3 = st.columns(3)
            with kc1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{total_cnt:,}명</div>
                    <div class="kpi-label">2024년 총 신고 건수</div>
                </div>""", unsafe_allow_html=True)
            with kc2:
                max_val = max(values)
                max_month = month_cols[values.index(max_val)]
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{max_month}</div>
                    <div class="kpi-label">최다 발생 월 ({max_val:,}명)</div>
                </div>""", unsafe_allow_html=True)
            with kc3:
                avg_val = round(sum(values)/12, 1)
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{avg_val}명</div>
                    <div class="kpi-label">월 평균 발생</div>
                </div>""", unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 차트 영역
            chart_col1, chart_col2 = st.columns([2, 1])
            
            with chart_col1:
                # 라인 차트
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=month_cols, y=values, 
                    mode='lines+markers', 
                    name=target_disease,
                    line=dict(color='#5361F2', width=4),
                    marker=dict(size=10, color='white', line=dict(color='#5361F2', width=2))
                ))
                fig.update_layout(
                    title=f"📈 {target_disease} 월별 발생 추이",
                    paper_bgcolor='white', plot_bgcolor='white',
                    height=400,
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#eee')
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with chart_col2:
                # 히트맵 스타일 바 차트 (계절성 확인용)
                df_season = pd.DataFrame({'Month': month_cols, 'Count': values})
                fig2 = px.bar(df_season, x='Count', y='Month', orientation='h',
                              title="월별 비중", text='Count',
                              color='Count', color_continuous_scale='Bluyl')
                fig2.update_layout(height=400, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("해당 등급에 질병 데이터가 없습니다.")

    else:
        st.warning("데이터를 불러올 수 없습니다.")


# =========================================================
# TAB 2: AI Analytics Center (수정됨: 모든 급수 예측 가능)
# =========================================================
with tab2:
    st.subheader("🤖 AI 감염병 예측 센터")
    st.write("Prophet 모델을 활용하여 과거 패턴을 학습하고, 향후 발생 추이를 예측합니다.")
    
    if not df.empty:
        col_ai_1, col_ai_2 = st.columns([1, 3])
        
        with col_ai_1:
            st.markdown("#### 예측 설정")
            
            # [수정] 모든 급수 선택 가능하도록 변경
            ai_classes = sorted(df['Class'].unique())
            ai_class = st.selectbox("등급 선택", ai_classes, key='ai_class')
            
            # 선택된 등급의 질병 목록
            ai_diseases = sorted(df[df['Class'] == ai_class]['Disease'].unique())
            ai_target = st.selectbox("분석 대상 질병", ai_diseases, key='ai_disease')
            
            periods = st.slider("예측 기간 (개월)", 1, 6, 3)
            
            run_ai = st.button("AI 예측 실행 🚀", type="primary")

        with col_ai_2:
            if run_ai and ai_target:
                with st.spinner(f"AI가 '{ai_target}' 데이터를 분석 중입니다..."):
                    time.sleep(1.2) # 연출용 딜레이
                    
                    # 데이터 준비 (Prophet용 포맷: ds, y)
                    # 2024년 1월 ~ 12월 데이터로 가정
                    row = df[(df['Class'] == ai_class) & (df['Disease'] == ai_target)].iloc[0]
                    
                    dates = []
                    counts = []
                    for i in range(1, 13):
                        date_str = f"2024-{i:02d}-01"
                        val = row[f"{i}월"]
                        dates.append(date_str)
                        counts.append(val)
                    
                    df_prophet = pd.DataFrame({'ds': dates, 'y': counts})
                    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
                    
                    # 모델 학습 (데이터 포인트가 적으므로 예외처리/파라미터 조정 필요하지만 단순화)
                    try:
                        m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
                        m.fit(df_prophet)
                        
                        future = m.make_future_dataframe(periods=periods, freq='MS') # 월 단위
                        forecast = m.predict(future)
                        
                        # 시각화
                        fig_ai = go.Figure()
                        
                        # 실제 데이터 (2024)
                        fig_ai.add_trace(go.Scatter(
                            x=df_prophet['ds'], y=df_prophet['y'],
                            mode='lines+markers', name='실제 발생(2024)',
                            line=dict(color='#333', width=2)
                        ))
                        
                        # 예측 데이터
                        pred_data = forecast[forecast['ds'] > '2024-12-01']
                        fig_ai.add_trace(go.Scatter(
                            x=pred_data['ds'], y=pred_data['yhat'],
                            mode='lines+markers', name='AI 예측',
                            line=dict(color='#FF4B4B', dash='dot', width=3),
                            marker=dict(symbol='star', size=12, color='#FF4B4B')
                        ))
                        
                        fig_ai.update_layout(
                            title=f"🔍 {ai_target} 향후 {periods}개월 예측 시뮬레이션",
                            hovermode="x unified",
                            height=500
                        )
                        st.plotly_chart(fig_ai, use_container_width=True)
                        
                        # 인사이트 생성 (간단 로직)
                        last_real = df_prophet['y'].iloc[-1]
                        last_pred = pred_data['yhat'].iloc[-1] if not pred_data.empty else 0
                        diff = last_pred - last_real
                        
                        insight_color = "red" if diff > 0 else "blue"
                        insight_text = "증가" if diff > 0 else "감소"
                        
                        st.info(f"""
                        **💡 AI Insight**
                        
                        현재 추세를 분석했을 때, **{ai_target}**의 발생 빈도는 향후 **{insight_text}**할 가능성이 있습니다.
                        특히 계절적 요인을 고려할 때 선제적인 예방 조치가 필요할 수 있습니다.
                        """)
                        
                    except Exception as e:
                        st.error(f"데이터 포인트 부족으로 예측이 어렵습니다. (최소 2년치 데이터 권장): {e}")
            elif not run_ai:
                st.info("좌측 패널에서 질병을 선택하고 'AI 예측 실행' 버튼을 눌러주세요.")
    else:
        st.warning("데이터가 로드되지 않아 분석을 수행할 수 없습니다.")

# =========================================================
# TAB 3: Personal Report (기존 유지)
# =========================================================
with tab3:
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.subheader("📝 사용자 정보 입력")
        with st.form("user_info"):
            age_g = st.selectbox("연령대", ["10대 미만", "10대", "20-30대", "40-50대", "60대 이상"])
            job = st.selectbox("직업군", ["사무직", "의료직", "교육/보육", "요식업", "기타"])
            st.markdown("**기저질환**")
            conds = st.multiselect("선택", ["당뇨병", "호흡기 질환", "간 질환", "면역 저하"])
            st.markdown("**접종 이력**")
            vax = st.multiselect("선택", ["독감", "폐렴구균", "간염"])
            sub = st.form_submit_button("분석 실행")
            
    with col_r:
        if sub:
            st.subheader("📑 MediScope Personal Report")
            score = 10; warns = []
            
            # 간단 로직
            if "10대 미만" in age_g: score += 20; warns.append(("소아 취약", "수두, 유행성 이하선염 주의"))
            if "60대 이상" in age_g: score += 40; warns.append(("고령층 고위험", "폐렴구균/독감 주의"))
            if "당뇨병" in conds: score += 30; warns.append(("만성질환", "합병증 및 감염 취약"))
            if "의료직" in job: score += 15; warns.append(("직업적 노출", "혈액 매개 감염 주의"))
            
            if "독감" in vax: score -= 10
            if "폐렴구균" in vax: score -= 10
            
            # 점수 클리핑
            score = max(0, min(100, score))
            
            # 위험도 표시
            risk_color = "green"
            risk_level = "안전"
            if score >= 40: risk_color = "orange"; risk_level = "주의"
            if score >= 70: risk_color = "red"; risk_level = "위험"
            
            st.markdown(f"""
            <div style="padding:20px; border-radius:10px; border:2px solid {risk_color}; text-align:center;">
                <h2 style="color:{risk_color}; margin:0;">위험도: {risk_level} ({score}점)</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🩺 맞춤형 권고 사항")
            if not warns:
                st.write("- 특별한 위험 요인이 감지되지 않았습니다. 개인 위생을 철저히 하세요.")
            else:
                for w_title, w_desc in warns:
                    st.write(f"- **{w_title}**: {w_desc}")
            
            st.info("본 결과는 AI 모의 분석 결과이며, 의학적 진단을 대체할 수 없습니다.")
