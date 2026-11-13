"""Train one PPO policy with disconnected roads across libsumo workers."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
from multiprocessing import get_context
from pathlib import Path
import shutil

from algorithms.parallel_ppo import ParallelPPO
from env.parallel_sumo import ParallelSumo, ShardedParallelSumo
from env.parallel_traffic import TrafficCurriculum
from env import road_config
from train import (
    ACTION, BASE, EGO, HPARAMS, LOG_INTERVAL,
    OBSERVATION, RESULTS_DIR, REWARD, ROAD, SIMULATION, TRAFFIC, road_builder, snapshot_configs,
)
from utils.evaluator import evaluate_parallel_policy
from utils.logger import RunLogger


SIGNAL_STOP_REWARDS = dict(
    wrong_lane_penalty=5.0,
    green_pass_bonus=20.0,
    green_stop_penalty=0.02,       # 스텝당 감점
    green_stop_grace_seconds=1.0,
    green_stop_speed=0.1,          # m/s 이하를 정지로 간주
    green_stop_clear_gap=10.0,     # 앞차가 없거나 범퍼 간격이 이 이상이면 진행 가능
    red_stop_penalty_coef=0.0,      # 멀리 있다고 감점하지 않음
    red_stop_clearance_reward=0.0, # 정지 중에도 들어오던 거리 보상 비활성화
    red_stop_progress_per_meter=0.0,  # 적신호 전진량 보상 비활성화
    red_speed_tracking_cap=0.20,      # 0.02, 0.05 실험을 거쳐 57번 설정 채택.
    red_speed_tracking_linear_underspeed=True,  # 목표보다 느릴 때도 출발부터 보상
    red_speed_tracking_target_decel=1.5,  # 목표 속도 곡선의 감속도(m/s²)
    red_speed_tracking_safe_decel=2.5,    # 정지 여유 판정의 감속도(m/s²)
    red_stopped_brake_coef=0.01,    # 적신호 정차 중 음수 가속 명령의 제곱 감점
    red_stopped_brake_speed=0.1,    # 행동 전 속도(m/s) 기준
    red_stopped_brake_min_room=2.0, # 목표 정차 위치까지 남아 있어야 할 추가 여유(m)
    red_safe_zone_first_reward=5.0,   # 적신호 안전 구역 최초 정차 보상
    red_safe_zone_hold_reward=0.005,  # 이후 안전 구역 유지 스텝당 보상
    red_safe_zone_hold_cap=0.2,       # 신호별 유지 보상 누적 상한
    red_safe_zone_min_gap=3.0,
    red_safe_zone_max_gap=5.0,
    red_safe_zone_max_speed=0.1,
    red_safe_zone_dwell_seconds=1.0,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1,
                        help="1: 신호등, 배경 차량 없음; 2: 신호등과 배경 교통량 커리큘럼")
    parser.add_argument("--init-model", type=Path,
                        help="Phase 2 초기 가중치로 사용할 기존 model.pt")
    parser.add_argument("--num-envs", type=int, default=4096, help="전체 환경 수")
    parser.add_argument("--num-workers", type=int, default=4, help="학습용 SUMO 프로세스 수")
    parser.add_argument("--randomize-departures", choices=("True", "False"), default="False",
                        help="ego 출발 차선 랜덤화 (기본 False: 도로 설정의 차선으로 고정)")
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=333)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--learning-epochs", type=int, default=5,
                        help="한 rollout을 PPO로 반복 학습하는 횟수")
    parser.add_argument("--eval-interval", type=int, default=0,
                        help="정기 평가 간격 (iteration 수, 기본값 0: 비활성화)")
    parser.add_argument("--results-dir", type=Path, default=Path(BASE) / RESULTS_DIR)
    parser.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE",
                        help="파일 수정 없이 설정 덮어쓰기 (여러 번 사용 가능). GROUP은 reward / obs / hparams, "
                             "VALUE는 JSON. 예: --set reward.off_route_lane_penalty=0.05 --set hparams.seed=1")
    parser.add_argument("--name", help="결과 폴더 이름 뒤에 붙일 실험 이름")
    args = parser.parse_args()
    overrides = {}
    for item in args.set:
        target, _, value = item.partition("=")
        group, _, key = target.partition(".")
        if group not in ("reward", "obs", "hparams") or not key or not value:
            parser.error(f"--set 형식 오류: {item} (예: reward.off_route_lane_penalty=0.05)")
        try:
            overrides.setdefault(group, {})[key] = json.loads(value)
        except json.JSONDecodeError:
            parser.error(f"--set 값은 JSON이어야 합니다: {item}")
    HPARAMS.update(overrides.get("hparams", {}))
    randomize_departures = args.randomize_departures == "True"
    if args.init_model is not None:
        args.init_model = args.init_model.resolve()
        if args.phase != 2 or not args.init_model.is_file():
            parser.error("--init-model에는 Phase 2에서 사용할 기존 model.pt를 지정해야 합니다.")
    rollout_steps = args.num_envs * args.steps_per_env
    if min(args.num_envs, args.num_workers, args.steps_per_env, args.iterations,
           args.minibatch_size, args.learning_epochs) < 1:
        parser.error("환경·워커 수, rollout 길이, iteration·epoch 수, 미니배치는 양수여야 합니다.")
    if args.num_envs % args.num_workers:
        parser.error("전체 환경 수는 워커 수의 배수여야 합니다.")
    envs_per_worker = args.num_envs // args.num_workers
    if rollout_steps < 2 or rollout_steps % args.minibatch_size:
        parser.error("rollout은 2 step 이상이며 미니배치 크기의 배수여야 합니다.")
    if args.eval_interval < 0:
        parser.error("--eval-interval은 0 이상이어야 합니다.")
    minibatches_per_epoch = rollout_steps // args.minibatch_size

    road_config.configure_parallel_training(args.phase)
    curriculum = TrafficCurriculum() if args.phase == 2 else None
    # Reuse the existing signal reward and 49-dimensional policy in both phases.
    reward_overrides = {**SIGNAL_STOP_REWARDS, **overrides.get("reward", {})}
    reward = {**REWARD, **reward_overrides}
    observation = {**OBSERVATION, "destination_enabled": False, "signal_count": 1,
                   "signal_enabled": True, **overrides.get("obs", {})}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"parallel_{args.phase}_{timestamp}" + (f"_{args.name}" if args.name else "")
    run_dir = args.results_dir.resolve() / run_name
    run_dir.mkdir(parents=True)
    snapshot_configs(run_dir)
    for path in (run_dir / "mdp_config.py", run_dir / "source/env/mdp_config.py"):
        with path.open("a", encoding="utf-8") as file:
            file.write(f"\nOBSERVATION = {observation!r}\n")
    for path in (run_dir / "road_config.py", run_dir / "source/env/road_config.py"):
        with path.open("a", encoding="utf-8") as file:
            file.write(f"\nconfigure_parallel_training({args.phase})\n")
    for path in (run_dir / "train.py", run_dir / "source/train.py"):
        with path.open("a", encoding="utf-8") as file:
            file.write(f"\nREWARD.update({reward_overrides!r})\n")
            if overrides.get("hparams"):
                file.write(f"HPARAMS.update({overrides['hparams']!r})\n")
    for directory in (run_dir, run_dir / "source"):
        shutil.copy2(__file__, directory)
    total_steps = rollout_steps * args.iterations
    with (run_dir / "parallel_config.json").open("w", encoding="utf-8") as file:
        json.dump({
            "phase": args.phase,
            "num_envs": args.num_envs,
            "num_workers": args.num_workers,
            "envs_per_worker": envs_per_worker,
            "steps_per_env": args.steps_per_env,
            "rollout_steps": rollout_steps,
            "iterations": args.iterations,
            "total_timesteps": total_steps,
            "minibatch_size": args.minibatch_size,
            "minibatches_per_epoch": minibatches_per_epoch,
            "epochs": args.learning_epochs,
            "seed": HPARAMS["seed"],
            "eval_interval": args.eval_interval,
            "eval_episodes": envs_per_worker if args.eval_interval else 0,
            "eval_seed": HPARAMS["seed"] + 1 if args.eval_interval else None,
            "randomize_signals": True,
            "randomize_departures": randomize_departures,
            "independent_episode_rng": True,
            "traffic_curriculum": asdict(curriculum) if curriculum else None,
            "periodic_eval_traffic_vph": 800 if curriculum and args.eval_interval else 0,
            "ring_route_probability": road_config.NOHYEONG["ego_routes"]["ring_prob"],
            "initialization": "random" if args.init_model is None else str(args.init_model),
            "overrides": overrides,
        }, file, indent=2)

    config = road_builder.build(ROAD, EGO, TRAFFIC, str(run_dir / "sumo"))
    env_kwargs = dict(cfg_path=config, road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                      mdp_obs=observation, action=ACTION, reward=reward, traffic=TRAFFIC)
    hparams = dict(HPARAMS, n_steps=rollout_steps, minibatch_size=args.minibatch_size,
                   n_epochs=args.learning_epochs)
    print(f"결과 폴더: {run_dir}", flush=True)
    template = ParallelSumo(envs_per_worker, env_kwargs, run_dir / "parallel", hparams["seed"],
                            randomize_signals=True, randomize_departures=randomize_departures,
                            independent_episode_rng=True,
                            traffic_curriculum=curriculum)
    env = template if args.num_workers == 1 else ShardedParallelSumo(template, args.num_workers)
    agent = ParallelPPO(obs_dim=env.prototype.obs_dim, act_dim=env.prototype.act_dim, **hparams)
    if args.init_model is not None:
        agent.load(str(args.init_model))
        shutil.copy2(args.init_model, run_dir / "initial_model.pt")
    print(f"Phase {args.phase} | {args.num_workers} workers × {envs_per_worker} envs | "
          f"rollout: {args.num_envs} × {args.steps_per_env} = {rollout_steps} step, "
          f"{args.iterations} iterations, {total_steps} total steps, "
          f"{args.learning_epochs} epochs × {minibatches_per_epoch} minibatches, "
          f"device={agent.device}", flush=True)
    eval_env = None
    if args.eval_interval:
        # Share the generated network files. The worker owns its own simulator.
        eval_env = deepcopy(template)
        eval_env.seed += 1
        if curriculum is not None:
            eval_env.fixed_traffic_vph = 800
    logger = RunLogger(run_dir)
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
            def evaluate():
                policy = deepcopy(agent.policy).cpu()
                return executor.submit(evaluate_parallel_policy, policy, eval_env, str(agent.device)).result()

            agent.learn_parallel(env, total_steps, logger=logger, log_interval=LOG_INTERVAL,
                                 evaluate=evaluate, eval_interval=args.eval_interval,
                                 eval_episodes=envs_per_worker)
    except KeyboardInterrupt:
        print("학습을 중단하고 현재 모델을 저장합니다.")
    finally:
        try:
            env.close()
        finally:
            logger.close()
    agent.save(run_dir / "model.pt")
    print(f"저장 완료: {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
