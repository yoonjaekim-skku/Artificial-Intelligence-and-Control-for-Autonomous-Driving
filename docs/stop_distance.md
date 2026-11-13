# 빨간불 정지 거리 개선 (정지선 0~3m에 서기)

제혁님 `parallel_train` 최신 코드(경로 중복 수정 `ebafef3` 포함) 위에서 작업했다. 신호 O, 배경 차량 X.

## 결과

410만 step(250 iterations), `scripts/evaluate_quick.py` (23개 경로 × 신호 시점 50개 = 1,150회).

| | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| 완주 | 100% | 100% | 100% |
| 신호 위반 / 잘못된 차선 / 시간 초과 / 충돌 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| 빨간불 정지 횟수 | 720 | 720 | 720 |
| 정지 거리 중앙값 | 2.73m | 2.47m | 2.45m |
| 정지 거리 Q1 ~ Q3 | 2.45 ~ 2.92m | 2.45 ~ 2.89m | 2.19 ~ 2.69m |
| 정지 거리 최소 ~ 최대 | 2.26 ~ 6.73m | 2.38 ~ 3.65m | 1.54 ~ 3.22m |
| 3m 이내 | 82.9% | 89.6% | 99.9% |
| 5m 이내 | 94.6% | 100% | 100% |

참고로 같은 조건에서 기본 보상(`SIGNAL_STOP_REWARDS`)은 정지 거리 중앙값 약 29m였다 (경로 수정 이전 코드, seed 1개).

남은 흠: seed 0에서 A3→A2 경로만 6.7m에 선다. 경로로 이어지지 않는 차선에 들어간 경우로 보인다.

## 바꾼 것

새 항목은 전부 **기본값 꺼짐**이다. 켜지 않으면 기존 학습 결과는 그대로다.

| 항목 | 내용 | 파일 |
|---|---|---|
| `reward.red_park_penalty_coef` (새 보상) | 빨간불에 **멈춰 있는 동안만**, 목표 정차 위치까지 남은 거리에 비례해 감점. 달리는 중에는 감점이 없어서, 크게 줘도 빨간불 구간 접근 자체가 손해가 되지 않는다. 앞차가 있으면 앞차 기준(`_red_stop_surplus`) | `env/sumo_env.py` |
| `obs.signal_fine_distance_scale` (새 관측) | 정지선 거리를 이 거리(m) 안쪽만 0~1로 확대한 값을 신호마다 1차원 추가. 기존 거리 관측은 100m 기준이라 1m가 0.01 차이라서, 정지선 바로 앞 미세 조정이 어려웠다 | `env/sumo_env.py` |
| `--set GROUP.KEY=VALUE`, `--name` | 파일 수정 없이 reward / obs / hparams 값 변경. 바꾼 값은 `parallel_config.json`과 설정 사본에 남아 `test.py --saved-config`에도 적용된다 | `train_parallel.py` |
| 이전 작업 merge | 교차로 내부 차선 관측 수정, `off_route_lane_penalty`(김윤재), `obs.lookahead_edges`, `late_brake_penalty_coef`(배승준) | `env/sumo_env.py` |

## 실행 명령

```bash
python train_parallel.py --phase 1 --num-envs 256 --num-workers 4 --steps-per-env 64 \
  --iterations 250 --learning-epochs 10 --minibatch-size 512 \
  --randomize-departures False --eval-interval 0 \
  --set reward.red_park_penalty_coef=0.1 --set reward.red_park_scale=10 \
  --set obs.signal_fine_distance_scale=10 \
  --set reward.red_stop_margin=0.5 \
  --set reward.red_safe_zone_min_gap=0.5 --set reward.red_safe_zone_max_gap=3.0 \
  --set reward.red_stopped_brake_min_room=1.0 \
  --set reward.red_stop_penalty_coef=0.1 --set reward.red_stop_distance_scale=50 \
  --set reward.late_brake_penalty_coef=0.1 \
  --set reward.off_route_lane_penalty=0.05 --set obs.lookahead_edges=3 \
  --set hparams.lane_keep_bias=0.0 \
  --set hparams.seed=0

python scripts/evaluate_quick.py results/<결과 폴더>
```

- 첫 세 줄의 `--set`이 정지 거리 핵심이다: 멈춰 있을 때 감점, 정밀 거리 관측, 목표 정차 위치 0.5m.
- 다음 두 줄은 제혁님 가점 구간(`red_safe_zone`)과 정차 후 브레이크 감점을 새 목표(0.5~3m)에 맞춘 것이다.
- 나머지는 접근 중 거리 감점, 늦은 브레이크 감점, 차선 선택 수정이다.

## 실험 과정 요약

모두 같은 환경, 410만 step. 위 두 단계는 경로 수정 이전 코드 결과다.

| 단계 | 3m 이내 | 정지 거리 중앙값 | 비고 |
|---|---:|---:|---|
| 기본 보상 (가점 구조) | 거의 0% | 29m | 빨간불이 보이자마자(약 97m) 감속해 멀리 섬 |
| + 접근 중 거리 감점 | 1% | 6~7m | 다가가지만 5~7m에서 멈춤 |
| + 목표 1.5m, 멈춰 있을 때 감점 | 1~40% | 3~7m | seed에 따라 갈림 |
| + 정밀 거리 관측 | 66% | 2.7~2.9m | seed 2개 모두 개선 |
| + 목표 0.5m (최종, 경로 수정 후) | 83~99.9% | 2.45~2.73m | seed 3개 모두 완주 100%, 위반 0 |

- 목표 숫자만 옮기는 것은 효과가 없었다. 정지선 근처에서는 더 다가가서 얻는 이득(약 0.5점)보다 선을 넘었을 때 잃는 것(위반 감점 + 초록불 통과 보너스)이 훨씬 커서다.
- 정책은 목표보다 1.5~2m 정도 여유를 두고 선다. 그래서 목표를 0.5m로 두면 실제로는 2~3m에 모인다.
