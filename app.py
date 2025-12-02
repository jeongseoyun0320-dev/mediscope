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
# 1. 디자인 (CSS)
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
    .hero-title { font-size: 32px; font-weight: 800; margin-bottom: 10px; }
    .hero-subtitle { font-size: 16px; opacity: 0.9; }
    
    .metric-card {
        background: white; border-radius: 15px; padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03); border: 1px solid #eee;
        text-align: center; transition: all 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
    .metric-val { font-size: 28px; font-weight: bold; color: #333; }
    .metric-label { font-size: 14px; color: #666; margin-bottom: 5px; }
    
    .info-box {
        background-color: #E3F2FD; border-left: 5px solid #2196F3;
        padding: 15px; border-radius: 5px; margin-bottom: 20px;
        color: #0D47A1; font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 데이터 로드 및 전처리 (수정됨)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # CSV 파일 로드 (헤더가 2줄인 구조 처리)
    # 첫 번째 줄: 연도 등, 두 번째 줄: 월별 헤더
    try:
        df = pd.read_csv("법정감염병_월별_신고현황_20251201171222.csv", header=1, encoding='utf-8')
    except:
        # 인코딩 오류 시 cp949 시도
        df = pd.read_csv("법정감염병_월별_신고현황_20251201171222.csv", header=1, encoding='cp949')

    # 컬럼명 정리 (급별(1) -> Grade, 급별(2) -> Disease)
    df.rename(columns={df.columns[0]: 'Grade', df.columns[1]: 'Disease'}, inplace=True)
    
    # 데이터 정제
    # 1. '소계', '기타' 등 통계용 행 제거
    df = df[~df['Disease'].isin(['소계', '기타', '총계'])]
    
    # 2. 등급 명칭 통일 (제1급 -> 1급, 2급 -> 2급)
    df['Grade'] = df['Grade'].astype(str).str.replace('제', '').str.strip()
    
    # 3. 월별 데이터 컬럼 확인 (숫자형 변환)
    month_cols = [c for c in df.columns if '월' in c]
    for col in month_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
    return df, month_cols

# 데이터 로딩
try:
    raw_df, month_cols = load_data()
    
    # 질병별 등급 매핑 딕셔너리 생성 (CSV 기반 동적 생성)
    disease_map = dict(zip(raw_df['Disease'], raw_df['Grade']))
    
    # 모든 질병 리스트 추출
    all_diseases = sorted(raw_df['Disease'].unique())
    
except Exception as e:
    st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. 사이드바 (네비게이션 & 필터)
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2785/2785819.png", width=60)
    st.markdown("### MediScope AI")
    st.info(f"데이터 기준: 2024년\n등록된 감염병: {len(all_diseases)}개")
    
    menu = st.radio("메뉴 이동", ["대시보드", "AI 예측 분석", "개인 위험도 평가"], index=0)
    
    st.markdown("---")
    st.markdown("### 🔍 감염병 선택")
    
    # 동적으로 로드된 전체 질병 리스트 사용
    selected_disease = st.selectbox(
        "분석할 질병을 선택하세요",
        all_diseases,
        index=0 if "A형간염" not in all_diseases else all_diseases.index("A형간염")
    )
    
    # 선택된 질병의 데이터 추출
    target_row = raw_df[raw_df['Disease'] == selected_disease].iloc[0]
    current_grade = disease_map.get(selected_disease, "정보없음")
    
    st.success(f"현재 선택: **{selected_disease}** ({current_grade})")

# ---------------------------------------------------------
# 4. 메인 콘텐츠
# ---------------------------------------------------------

# 헤더 섹션
st.markdown(f"""
<div class="hero-box">
    <div class="hero-title">MediScope Analytics</div>
    <div class="hero-subtitle">공공데이터 기반 감염병 발생 현황 및 AI 예측 솔루션</div>
</div>
""", unsafe_allow_html=True)

# [페이지 1] 대시보드
if menu == "대시보드":
    st.title(f"📊 {selected_disease} 발생 현황")
    
    # 상단 요약 카드
    total_cases = target_row['계']
    avg_cases = int(total_cases / 12) if total_cases > 0 else 0
    max_month_val = 0
    max_month_name = "-"
    
    # 월별 최대 발생월 찾기
    monthly_data = target_row[month_cols]
    if total_cases > 0:
        max_month_val = monthly_data.max()
        max_month_name = monthly_data.idxmax()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">법정 감염병 등급</div>
            <div class="metric-val" style="color:#5361F2">{current_grade}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">2024 누적 확진</div>
            <div class="metric-val">{int(total_cases):,}명</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">월 평균 발생</div>
            <div class="metric-val">{avg_cases:,}명</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">최다 발생월</div>
            <div class="metric-val" style="color:#E91E63">{max_month_name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 등급별 안내 메시지 (동적 처리)
    grade_info = {
        "1급": "즉시 신고가 필요한 최고 위험 감염병입니다. (에볼라, 페스트 등)",
        "2급": "24시간 이내 신고 및 격리가 필요한 감염병입니다. (결핵, 홍역, 콜레라 등)",
        "3급": "24시간 이내 신고, 발생 감시가 필요한 감염병입니다. (파상풍, B형/C형 간염 등)",
        "4급": "표본감시 활동이 필요한 감염병입니다. (인플루엔자, 성매개감염병 등)"
    }
    
    info_msg = grade_info.get(current_grade, "등급 정보가 명확하지 않습니다.")
    st.markdown(f'<div class="info-box">ℹ️ <b>{current_grade} 감염병 가이드:</b> {info_msg}</div>', unsafe_allow_html=True)

    # 차트 영역
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.subheader("🗓️ 월별 발생 추이")
        # 데이터프레임 변환 for Plotly
        chart_df = pd.DataFrame({
            '월': month_cols,
            '환자수': monthly_data.values
        })
        
        fig = px.area(chart_df, x='월', y='환자수', markers=True, 
                      line_shape='spline', color_discrete_sequence=['#5361F2'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
    with col_chart2:
        st.subheader("📊 분기별 비중")
        # 분기 데이터 계산
        q1 = sum(monthly_data.values[0:3])
        q2 = sum(monthly_data.values[3:6])
        q3 = sum(monthly_data.values[6:9])
        q4 = sum(monthly_data.values[9:12])
        
        fig_pie = px.donut(values=[q1, q2, q3, q4], names=['1분기','2분기','3분기','4분기'],
                           color_discrete_sequence=px.colors.sequential.Bluyl)
        fig_pie.update_layout(showlegend=False, 
                              annotations=[dict(text='분기', x=0.5, y=0.5, font_size=20, showarrow=False)])
        st.plotly_chart(fig_pie, use_container_width=True)

# [페이지 2] AI 예측 (Prophet)
elif menu == "AI 예측 분석":
    st.title("🤖 AI Future Prediction")
    st.markdown("과거 데이터를 기반으로 **향후 3개월 간의 발생 추이**를 예측합니다.")
    
    if total_cases == 0:
        st.warning("⚠️ 데이터가 부족하여(0건) 예측을 수행할 수 없습니다.")
    else:
        with st.spinner('AI 모델이 데이터를 분석 중입니다...'):
            time.sleep(1) # UX용 딜레이
            
            # Prophet용 데이터셋 생성 (2024년 기준 가상 시계열 생성)
            # 실제로는 연도별 데이터가 더 필요하지만, 데모를 위해 2024년 데이터를 시계열로 변환
            dates = []
            vals = []
            base_date = datetime(2024, 1, 1)
            
            for idx, val in enumerate(monthly_data.values):
                # 각 월의 1일로 설정
                curr_date = base_date + timedelta(days=idx*30) 
                dates.append(curr_date)
                vals.append(val)
                
            prophet_df = pd.DataFrame({'ds': dates, 'y': vals})
            
            # 모델 학습
            m = Prophet()
            m.fit(prophet_df)
            
            # 향후 90일(3개월) 예측
            future = m.make_future_dataframe(periods=3, freq='M')
            forecast = m.predict(future)
            
            # 결과 시각화
            st.subheader(f"{selected_disease} 향후 예측 그래프")
            
            fig_pred = go.Figure()
            # 실제 데이터
            fig_pred.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='실제 발생',
                                        line=dict(color='#333', width=3)))
            # 예측 데이터
            fig_pred.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI 예측',
                                        line=dict(color='#5361F2', dash='dot')))
            
            fig_pred.update_layout(title="실제 vs 예측 비교", hovermode="x unified")
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # 예측 코멘트
            next_month_pred = int(forecast.iloc[-1]['yhat'])
            st.success(f"📈 분석 결과, 다음 달 예상 환자 수는 약 **{max(0, next_month_pred)}명** 입니다.")

# [페이지 3] 개인 위험도 평가
elif menu == "개인 위험도 평가":
    st.title("🩺 Personal Health Check")
    st.markdown("간단한 문진을 통해 감염병 위험도를 체크해보세요.")
    
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        with st.form("check_form"):
            st.markdown("**기본 정보**")
            age_g = st.radio("연령대", ["10대 미만", "10대", "20-30대", "40-50대", "60대 이상"])
            job = st.selectbox("직업군", ["사무직", "의료직", "교육/보육", "요식업", "기타"])
            
            st.markdown("**기저질환**")
            conds = st.multiselect("해당사항 선택", ["당뇨병", "호흡기 질환", "간 질환", "면역 저하", "없음"])
            
            st.markdown("**관심/접종 이력**")
            vax = st.multiselect("최근 접종 백신", ["독감", "폐렴구균", "간염", "코로나19", "없음"])
            
            sub = st.form_submit_button("분석 실행")
            
    with col_r:
        if sub:
            st.subheader("📑 MediScope Personal Report")
            score = 10
            warns = []
            
            # 간단한 룰 기반 로직
            if "10대 미만" in age_g: 
                score += 20
                warns.append(("소아 취약", "수두, 홍역 등 단체생활 감염 주의"))
            if "60대 이상" in age_g: 
                score += 40
                warns.append(("고령층 고위험", "폐렴구균 및 독감 합병증 주의"))
            if "당뇨병" in conds or "간 질환" in conds:
                score += 30
                warns.append(("만성질환자", "감염 시 중증 진행 위험 높음"))
            if "의료직" in job:
                score += 20
                warns.append(("의료 종사자", "혈액 매개 감염(B형/C형 간염) 및 호흡기 감염 주의"))
            if "요식업" in job:
                warns.append(("식품 매개", "A형 간염 및 수인성 전염병 주의"))
                
            # 백신 효과
            if "없음" not in vax and len(vax) > 0:
                score -= 10
                st.success(f"✅ {', '.join(vax)} 백신 접종으로 방어력이 형성되었습니다.")
            
            # 결과 표시
            st.divider()
            if score >= 60:
                st.error(f"🚨 위험도: 높음 ({score}점)")
                st.markdown("**전문가 상담 및 철저한 예방수칙 준수가 필요합니다.**")
            elif score >= 30:
                st.warning(f"⚠️ 위험도: 주의 ({score}점)")
                st.markdown("**유행 시기에는 사람이 많은 곳을 피하세요.**")
            else:
                st.success(f"🟢 위험도: 양호 ({score}점)")
                st.markdown("**현재 건강 상태를 잘 유지하세요!**")
                
            if warns:
                st.markdown("#### 💡 맞춤형 조언")
                for w_title, w_desc in warns:
                    st.info(f"**[{w_title}]** {w_desc}")
