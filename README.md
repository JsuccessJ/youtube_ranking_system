# YouTube Multitask Ranking System

YouTube 비디오 추천 시스템에서 사용되는 멀티태스크 랭킹 시스템의 PyTorch 구현입니다. 이 구현은 RecSys 2019에서 발표된 논문 ["Recommending What Video to Watch Next: A Multitask Ranking System"](https://dl.acm.org/doi/10.1145/3298689.3346997)을 기반으로 합니다.

## 모델 아키텍처


이 시스템은 다음과 같은 핵심 컴포넌트로 구성됩니다:

1. **Input Features**: 질의어, 후보 비디오, 사용자, 컨텍스트 특성 등 다양한 입력 특성
2. **공유 은닉 계층 (Shared Bottom Layer)**: 입력 특성과 임베딩을 처리하고 추상화하는 초기 계층
3. **Mixture-of-Experts (MoE)**: 여러 전문가 네트워크의 출력을 조합하는 층
   - **공유 전문가 (Shared Experts)**: 모든 태스크가 공유하는 전문가
   - **태스크별 전문가 (Task-specific Experts)**: 각 태스크에 특화된 전문가
4. **Gating Networks**: 각 태스크별로 전문가의 출력에 가중치를 부여하는 게이팅 네트워크
5. **Side Tower (Wide 부분)**: 선택 편향을 보정하기 위한 얕은 네트워크
6. **Multitask Objectives**: 두 가지 범주의 사용자 행동을 예측
   - **참여도 (Engagement)**: 클릭, 시청 시간 등과 같은 행동
   - **만족도 (Satisfaction)**: 좋아요, 공유, 댓글 등과 같은 행동
7. **Weighted Combination**: 여러 태스크의 예측값을 가중치 조합하여 최종 랭킹 점수 생성

### Wide & Deep 아키텍처

이 구현은 Google의 Wide & Deep 모델 아키텍처의 개념을 통합했습니다:
- **Wide 부분 (Side Tower)**: 암기(memorization)를 담당하는 얕은 네트워크로 선택 편향을 모델링
- **Deep 부분 (MMoE)**: 일반화(generalization)를 담당하는 깊은 신경망

## 주요 특징

- **멀티태스크 학습**: 한 번의 학습으로 참여도와 만족도를 동시에 예측
- **Mixture-of-Experts**: 다양한 전문가 네트워크의 결합으로 표현력 향상
- **선택 편향 보정**: Side-tower를 통한 선택 편향 보정
- **공유 은닉 계층**: 학습 효율성을 높이기 위한 초기 특성 추출
- **가중치 조합**: 태스크별 예측을 적절히 조합하여 최종 랭킹 생성

## 설치 요구사항

- Python 3.6+
- PyTorch 1.7+
- NumPy
- pandas
- scikit-learn
- tqdm

## 설치 방법

```bash
# 저장소 클론
git clone https://github.com/your-username/YouTube-Ranking-System.git
cd YouTube-Ranking-System

# 필요한 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

### 모델 학습

```bash
python train.py --model_name youtube_ranking --batch_size 256 --epochs 50 --use_wide_and_deep
```

### 모델 평가

```bash
python train.py --model_name youtube_ranking --batch_size 256 --epochs 0 --use_wide_and_deep
```

### 주요 매개변수

- `--model_name`: 사용할 모델 (`youtube_ranking` 또는 `mmoe`)
- `--num_samples`: 합성 데이터셋의 샘플 수
- `--task_num`: 태스크 수 (기본값: 2, 참여도와 만족도)
- `--shared_bottom_dims`: 공유 은닉 계층의 차원
- `--shared_expert_num`: 공유 전문가의 수
- `--task_expert_num`: 태스크별 전문가의 수
- `--embed_dim`: 임베딩 차원
- `--dropout`: 드롭아웃 비율
- `--use_wide_and_deep`: Wide & Deep 아키텍처 사용 여부
- `--learning_rate`: 학습률
- `--batch_size`: 배치 크기
- `--epochs`: 학습 에포크 수

## 디렉토리 구조

```
YouTube-Ranking-System/
├── models/               # 모델 정의
│   ├── layers.py         # 기본 레이어 구현
│   ├── mmoe.py           # MMoE 모델 구현
│   └── youtube_ranking.py # YouTube 랭킹 모델 구현
├── datasets/             # 데이터셋 클래스
│   └── synthetic.py      # 합성 데이터셋 구현
├── utils/                # 유틸리티 함수
│   └── common.py         # 공통 유틸리티 함수
├── checkpoints/          # 모델 체크포인트 저장 디렉토리
├── logs/                 # 학습 로그 저장 디렉토리
├── train.py              # 학습 및 평가 스크립트
└── README.md             # 이 문서
```

## 논문에서 제안된 아키텍처의 구현

이 코드는 논문에서 제안한 다음 핵심 구성 요소를 구현합니다:

1. **공유 은닉 계층**: 입력 처리 후 MMoE 레이어 전에 특성 추출 (`SharedBottomLayer` 클래스)
2. **Multi-gate Mixture-of-Experts (MMoE)**: 공유 및 태스크별 전문가 네트워크와 게이팅 메커니즘
3. **얕은 타워(Shallow Tower)**: 선택 편향을 모델링하기 위한 별도의 네트워크 (`SideTower` 클래스)
4. **Wide & Deep 아키텍처**: 얕은 타워(Wide)와 MMoE(Deep)의 조합
5. **가중치 랭킹 점수**: 여러 태스크의 예측값에 대한 가중치 조합

## 참고 문헌

- Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar, Maheswaran Sathiamoorthy, Xinyang Yi, and Ed Chi. 2019. **"Recommending What Video to Watch Next: A Multitask Ranking System."** In Proceedings of the 13th ACM Conference on Recommender Systems (RecSys '19).
- Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah. 2016. **"Wide & Deep Learning for Recommender Systems."** In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems (DLRS '16).
- Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H. Chi. 2018. **"Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts."** In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '18).

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 
=======
