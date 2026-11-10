# 🏥 MediScope: AI 감염병 예측 및 분석 플랫폼

> **MediScope**는 법정감염병 통계 데이터를 기반으로 미래 발생 추이를 예측하고, 사용자의 증상 및 개인 프로필에 맞춘 지능형 예방 정보를 제공하는 **AI 기반 감염병 모니터링 플랫폼**입니다.

---

## 🌟 주요 기능 (Key Features)

### 1. 🏠 홈 (Infection Status & Trends)
* **실시간 감염병 현황 조회**: 법정감염병 분류 등급(1급~4급 등) 및 질병별 월별 발생 추이를 직관적인 그래프로 제공합니다.
* **맞춤형 예방 수칙**: 조회 중인 질병의 특성(호흡기, 수인성/식중독, 매독/혈액매개 등)에 따른 구체적이고 실효성 있는 예방 행동 요령을 제공합니다.

### 2. 💬 AI 의료 상담 (Symptom-based Prediction)
* **대화형 증상 분석**: 사용자가 일상 언어로 겪고 있는 증상(예: "진드기에 물렸어요", "해산물을 먹고 배가 아파요")을 입력하면 AI가 실시간으로 분석합니다.
* **의심 질병 매칭**: 법정감염병 데이터베이스와 연동하여 연관성이 높은 의심 질병을 즉각 식별하고, 보건소 및 병원 방문 가이드를 안내합니다.

### 3. 📊 AI 분석 센터 (Future AI Analysis)
* **2026년 미래 예측 모델링**: 과거 데이터를 기반으로 Prophet 시계열 분석 및 머신러닝 패턴을 적용하여 향후 환자 발생 추이와 **신뢰구간(Confidence Interval)**을 시각화합니다.
* **계절성 패턴 분석**: 월별 발생 집중도를 파악할 수 있는 **Radar Chart**와 **Bar Chart**를 지원합니다.
* **다년도 발생 히트맵 (Heatmap)**: 연도별·월별 발생 강도를 한눈에 파악할 수 있는 다차원 히트맵을 제공합니다.

### 4. 👤 My Page (Personal Risk Report)
* **개인 맞춤형 위험도 진단**: 연령대, 직업군, 기저질환, 접종 이력을 입력받아 고위험 요인을 다각도로 분석합니다.
* **예방 가이드 및 추천 백신**: 사용자 특성에 최적화된 감염병 주의 경고와 미접종 필수 백신을 안내합니다.

---

## 🛠️ 기술 스택 (Tech Stack)

* **Language**: Python 3.8+
* **Framework**: Streamlit
* **Data Processing**: Pandas, NumPy
* **Data Visualization**: Plotly (Graph Objects, Express)
* **Machine Learning / Time Series**: Prophet (시계열 예측용)

---

## 📂 프로젝트 구조 (Project Structure)

```text
MediScope/
│
├── app.py                                   # 메인 Streamlit 애플리케이션 코드
├── 법정감염병_월별_신고현황_20251201171222.csv   # 법정감염병 통계 데이터셋
└── README.md                                # 프로젝트 설명서
