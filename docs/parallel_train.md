# Massively Parallel PPO for SUMO

## 1. 학습 환경 설정

### 도로 복제와 CPU 워커

[도로 설정](../env/road_config.py)으로 [원본 SUMO 도로](../env/road_builder.py)를 만든다. [병렬 환경](../env/parallel_sumo.py)은 서로 연결되지 않은 도로 복제본에 ego를 한 대씩 배치한다. 한 SUMO 프로세스가 여러 도로를 처리하지만, 서로 다른 복제본의 차량은 만나거나 서로를 관측하지 않는다.

| 옵션 | 의미 | 기본값 |
|---|---|---:|
| `--num-envs` | 모든 워커를 합친 도로 복제본 수 | 4,096 |
| `--num-workers` | 독립적으로 실행하는 SUMO 프로세스 수 | 4 |

기본값에서는 워커 4개가 각각 1,024개 도로를 맡는다. 전체 환경 수는 워커 수로 나누어떨어져야 한다. 하나의 정책으로 모든 ego의 행동을 계산한 뒤, 각 워커가 SUMO를 한 스텝 진행한다. 에피소드가 끝난 차량만 재투입한다. 시뮬레이션과 환경 계산은 CPU에서 실행하고, 기본 장치 설정에서는 CUDA를 사용할 수 있으면 정책 추론과 PPO 학습을 GPU에서 실행한다.

### 학습 중 새 에피소드 시작 시 환경 설정

학습 중 각 도로에서 새 에피소드를 시작할 때, 주행 경로·시작 차선·신호 시작 시점을 각각 뽑는다. 이 세 항목과 배경 차량 투입은 하나의 실험 seed에서 나눈 별도의 난수를 사용한다.

- **주행 경로:** Ring 미경유 경로 18개에서 85%, Ring 경유 5개에서 15%확률로 그룹을 고르고, 그룹 안에서 균등하게 경로하나를 뽑는다.
- **시작 차선:** `--randomize-departures True`이면 출발 도로의 차선 중 하나를 균등하게 뽑는다. `--randomize-departures False`(기본값)이면 도로 설정의 시작 차선(기본 중앙 차선)으로 고정한다. 경로와 신호의 난수는 그대로 유지된다.
- **신호 시작 시점:** 125초 주기 안에서 시작 시점을 0.5초 단위로 균등하게 뽑는다. 같은 도로의 연동 신호에는 같은 시점을 적용한다. 신호 순서와 각 신호의 지속 시간은 그대로다.
- **교통량 커리큘럼 (game-inspired traffic curriculum):** `--phase 2`에서만 사용한다. 모든 환경은 0대/시간에서 시작하고, 에피소드가 끝난 환경만 다음 교통량을 바꾼다. 완주하면 +50대/시간, 실패하면 −150대/시간으로 조절하며 범위는 0~1,500대/시간이다.

ego 차량의 자동 순간이동은 꺼져있다. (`timeToTeleport="-1"`) 계속 정차해서 에피소드 시간 제한에 도달하면 시간 초과로 기록된다.

## 2. PPO 학습

### Phase 별 학습

```bash
python train_parallel.py --phase 1 --num-envs 256 --num-workers 4 \
  --steps-per-env 64 --iterations 500 --learning-epochs 10 --minibatch-size 512 \
  --randomize-departures False --eval-interval 0
python train_parallel.py --phase 2
python train_parallel.py --phase 2 --init-model results/parallel_1_YYYYMMDD_HHMMSS_ffffff/model.pt
```

- **Phase 1:** ego 차량 단독으로 배경차량 없이 신호 작동 상황에서 학습
- **Phase 2:** 환경마다 직전 episode의 완주 성공 여부로 조절되는 교통량 설정을 통해 최종 학습 *(아직 미구현)*

| 옵션 | Phase 1 권장값 | Phase 1 예비 |
|---|---:|---:|
| `--num-envs` | 256개 | 512개 |
| `--num-workers` | 4개 | 4개 |
| `--steps-per-env` | 64스텝 | 64스텝 |
| `--iterations` | 500회 | 500회 |
| `--learning-epochs` | 10회 | 10회 |
| `--minibatch-size` | 512스텝 | 1,024스텝 |
| `--randomize-departures` | `False` | `False` |
| `--eval-interval` | 0 (학습 중 정기 평가 끔) | 0 (학습 중 정기 평가 끔) |

워커당 64개 환경을 실행하며, iteration당 16,384스텝을 수집한다. PPO epoch당 미니배치는 32개이고, 총 학습량은 8,192,000스텝이다.


## 3. 모델 평가


결과는 `results/<학습 결과 폴더>/analysis/learning_curve.png`와 `route_evaluation.csv`에 저장된다. 

### 빠른 평가

고유 경로 23개를 신호 시작 시점 50개에서 각각 평가한다 (총 1,150회).

```bash
conda activate sumo
python scripts/evaluate_quick.py results/<학습 결과 폴더>
```

`./results/<학습 결과 폴더>/analysis/quick/learning_curve.png`를 보시면 한 눈에 평가 결과를 확인할 수 있다. 표시되는 정보는 다음과 같다.
- 학습 곡선과 최종 완주율
- 경로별 완주·실패 횟수와 신호 시작 시점별 결과
  - 각 경로는 50개의 서로 다른 신호등 주기에서 시작하는 경로
- 적신호 정차 시 전방 여유 거리의 경로별·전체 평균, 중앙값, Q1, Q3 및 표본 수


### 정밀 평가

*아직 미구현 - 랜덤 차선 출발 등 ...*


## 4. TODO
- `--phase 1`에서 정지선 멀리서 정차하는 문제
  - `--phase 2`에서 배경차량이 있는 것을 고려해서 정지선 멀리서 정차 + 앞차량과 너무 떨어져 정차 문제로 생각해주세요.
- `--phase 1`에서 `--randomize-departures True`를 활성화해도 잘되도록
- `--phase 1` 에피소드 시작 시 환경 설정 개선
- `--phase 2` 구현
- 정밀평가 구현
