# 💳 RecoAI — 생성형 AI 카드 추천 서비스

<div align="center">
  <h3>"당신의 라이프스타일에 맞춘 최고의 카드 추천"</h3>
  <p>멋쟁이사자처럼 자연어처리 1팀 Reco | 5인 팀 프로젝트 (2024.12)</p>
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)

</div>

---

## 📋 프로젝트 개요

![image](https://github.com/OreOrca/AI_NLP_-Reco/blob/main/images/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202024-12-25%20205754.png?raw=true)

사용자의 소비 내역을 분석하거나 자연어 대화를 통해 라이프스타일을 파악하고, **최대 혜택 카드를 추천 및 예상 혜택 금액을 시뮬레이션**하는 AI 챗봇 서비스입니다.

### 💡 주요 기능

- 소비 내역 업로드 시 업종/가맹점별 혜택을 자동 계산하여 최적 카드 추천
- 자연어 대화를 통한 실시간 카드 혜택 문의 및 맞춤 추천
- 예상 혜택 금액 시뮬레이션 및 시각화

---

## 🙋 나의 기여 (My Contributions)

> 5인 팀 프로젝트에서 **자연어 분류 모델 개발, 카드 혜택 데이터 정규화, 클라우드 인프라 구축**을 담당했습니다.

### 1. 자연어 분류 모델 최적화 — 분류 정확도 60% → 99%

사용자 질의에서 **업종명과 가맹점명을 추출·분류**하는 AI 모델을 설계·개발했습니다.

**문제 인식 및 해결 과정:**

- **베이스 모델 선정**: 한국어 구어체 처리에 강점을 가진 `KcELECTRA`를 베이스 모델로 선정
- **Two-Tower 아키텍처 응용**: 단순 코사인 유사도 분류 방식의 한계를 극복하기 위해, 질문 벡터와 업종 벡터를 결합하여 학습시키는 **Two-Tower 구조**를 고안·적용
- **데이터 증강**: 학습 데이터 부족 및 클래스 불균형 문제 해결을 위해 GPT를 활용한 데이터 증강 수행
- **Longest Match First 알고리즘 적용**: 가맹점명 추출 시 포함 관계에 있는 단어의 오분류 문제를 `최장 일치 우선(Longest Match First)` 알고리즘으로 해결

| 항목 | 개선 전 | 개선 후 |
|------|--------|--------|
| 분류 정확도 (자체 검증) | ~60% | **99%** |
| 분류 방식 | 코사인 유사도 | Two-Tower 아키텍처 |
| 가맹점명 오분류 | 발생 | Longest Match First로 해결 |

---

### 2. 카드 혜택 데이터 정규화 (Data Structuring)

전월 실적, 한도, 할인 유형 등 **복잡한 비정형 카드 혜택 데이터**를 실시간 정량 계산이 가능한 구조로 정규화했습니다.

- 카드고릴라 상위 50개 카드의 혜택 조건을 계산 가능한 필드 단위로 분해·설계
- **혜택 선택형 카드** 처리: 사용자가 혜택을 직접 선택하는 카드의 경우, 각 선택지를 파생 상품으로 분리하여 DB에 적재함으로써 복잡한 경우의 수를 유연하게 수용

---

### 3. 클라우드 인프라 구축 (Infrastructure)

클라우드 인스턴스의 **로컬 스토리지 용량 부족**으로 인해 8B 규모의 LLM 구동이 불가능한 상황을 해결했습니다.

- 환경 변수(`OLLAMA_MODELS`) 설정으로 Ollama의 모델 저장 경로를 **NAS 마운트 스토리지로 우회**
- 로컬 스토리지 제약 없이 대용량 LLM 추론 환경을 안정적으로 구동

---

## 🏗️ 시스템 아키텍처

![image](https://github.com/OreOrca/AI_NLP_-Reco/blob/main/images/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202024-12-25%20214301.png?raw=true)

```
사용자 입력
    │
    ▼
[KoBERT 의도 분류기] ──────────────────────────────────┐
    │                                                   │
    ├── 카드 혜택 질의 ──► [Vector DB 검색 (FAISS)]      │
    │                                                   │
    ├── 최신 정보 질의 ──► [Web 검색 (Tavily)]           │
    │                                                   │
    └── 소비 내역 분석 ──► [업종/가맹점 분류 ★]          │
                              │                         │
                              ▼                         │
                     [혜택 계산 엔진]                   │
                              │                         │
                              └──────────────────────────┤
                                                         ▼
                                           [LLM (BCCard/LLaMA-3.1-8B)]
                                                         │
                                                         ▼
                                                   최종 응답 생성
```

> ★ 표시는 본인 담당 파트

---

## 🤖 사용 모델

| 모델 | 용도 |
|------|------|
| `BCCard/Llama-3.1-Kor-BCCard-Finance-8B` | 한국 금융 도메인 특화 LLM (답변 생성) |
| `intfloat/multilingual-e5-large-instruct` | 카드 혜택 벡터 임베딩 (RAG) |
| `monologg/KoBERT` | 사용자 의도 분류 |
| `snunlp/KcELECTRA` ★ | 업종/가맹점명 분류 (Two-Tower 구조 적용) |

---

## 📊 데이터셋

| 구분 | 수량 | 설명 |
|------|------|------|
| Vector DB | 2,740개 | 정답/불가 버전 각 1,370개 |
| Web 검색 DB | 3,000개 | 정답/불가 버전 각 1,500개 |
| **합계** | **5,740개** | |

- **출처**: 카드고릴라 신용카드 TOP 100 차트 상위 50개 카드
- 전월 실적, 한도, 할인 유형 등 혜택 조건을 정량 계산 가능한 형태로 정규화 ★

---

## 🗂️ 프로젝트 구조

```
├── app.py              # Flask 웹 서버 및 라우팅
├── Graph.py            # LangGraph 에이전트 파이프라인 설계
├── Models.py           # LLM / 분류기 / 임베딩 모델 로딩
├── Tools.py            # VectorDB 검색, Web 검색, 보상 함수 툴
├── Reward_function.py  # 카드 혜택 계산 엔진
├── settings.py         # 전역 설정 및 환경 변수
├── Runner.py           # 모델 실행 진입점
└── images/             # README 및 서비스 스크린샷
```

---

## 👥 팀 구성

멋쟁이사자처럼 자연어처리 1팀 Reco (5인, 약 1개월)

---

© 2024 생성형 AI 카드 추천 서비스. All Rights Reserved.
