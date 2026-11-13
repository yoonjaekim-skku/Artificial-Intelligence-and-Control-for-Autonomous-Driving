"""Evaluate one saved Phase 1 or Phase 2 PPO run and generate a learning-curve PNG."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.ppo import PPO
from env import road_builder
from env.ego_routes import read_ego_route_groups
from env.parallel_sumo import ParallelSumo
from env.parallel_traffic import TrafficCurriculum
from test import load_saved_configs
from utils.evaluator import evaluate_parallel_routes


SEEDS = tuple(range(5))
OUTCOMES = {
    "arrival": ("Arrival", "OK", "#78c679"),
    "wrong_lane": ("Wrong lane", "LANE", "#fdae6b"),
    "red_light": ("Red light", "RED", "#ef3b2c"),
    "timeout": ("Timeout", "TIME", "#9e9ac8"),
    "collision": ("Collision", "CRASH", "#de6d69"),
    "unknown": ("Unknown", "?", "#cccccc"),
}


def _read_log(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _curve(axis, rows, key, title, window=1, style="-"):
    if rows and key in rows[0]:
        steps = np.array([int(row["steps"]) for row in rows]) / 1000
        values = np.array([float(row[key]) for row in rows])
        if window > 1 and len(values) >= window:
            values = np.convolve(values, np.ones(window) / window, mode="valid")
            steps = steps[window - 1:]
        elif window > 1:
            title = title.split(" (", 1)[0]
        if len(values) == 1:
            style = "."
        axis.plot(steps, values, style)
    else:
        axis.text(.5, .5, "No logged data", ha="center", va="center", transform=axis.transAxes)
    axis.set_title(title)
    axis.set_xlabel("Training steps (thousands)")
    axis.grid(alpha=.25)


def _summarize(rows):
    return {
        "episodes": len(rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        **{key: float(np.mean([row[key] for row in rows])) for key in (
            "success_rate", "collision_rate", "wrong_lane_rate", "timeout_rate",
            "red_light_violation_rate", "ep_return", "lane_changes", "mean_speed",
            "ep_length", "green_passes",
        )},
    }


def _plot_training_panels(fig, grid, run):
    training = _read_log(run / "training_log.csv")
    evaluation = _read_log(run / "eval_log.csv")
    panels = (
        (training, "ep_rew_mean", "Training return (10-iteration mean)", 10, "-"),
        (evaluation, "success_rate", "Periodic success (5-evaluation mean)", 5, "-"),
        (training, "value_loss", "Value loss (10-iteration mean)", 10, "-"),
        (evaluation, "lane_changes", "Periodic lane-change commands / episode", 1, ".-"),
        (training, "approx_kl", "Approximate KL", 1, "-"),
        (training, "p_lane_change", "Training lane-change probability", 1, "-"),
    )
    for index, panel in enumerate(panels):
        axis = fig.add_subplot(grid[index // 2, index % 2])
        _curve(axis, *panel)
        if panel[1] == "success_rate":
            axis.set_ylim(-.05, 1.05)


def _plot(run, routes, rows, summary, out):
    fig = plt.figure(figsize=(13, 16), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=(1, 1, 1, 1.9))
    _plot_training_panels(fig, grid, run)

    left = grid[3, 0].subgridspec(2, 1, height_ratios=(3, 1))
    counts = fig.add_subplot(left[0])
    metrics = summary["groups"]["all"]
    values = [metrics["statuses"].get(status, 0) for status in OUTCOMES]
    bars = counts.barh([label for label, _, _ in OUTCOMES.values()], values,
                       color=[color for _, _, color in OUTCOMES.values()])
    counts.bar_label(bars, padding=4)
    counts.invert_yaxis()
    counts.set_xlim(0, len(rows) * 1.12)
    counts.xaxis.set_major_locator(MaxNLocator(integer=True))
    counts.set_title(f"Final evaluation outcomes ({len(rows)} episodes)")
    counts.set_xlabel("Episodes")
    counts.grid(axis="x", alpha=.25)

    details = fig.add_subplot(left[1])
    details.axis("off")
    details.text(.03, .95, "\n".join((
        f"Arrival rate: {metrics['success_rate']:.1%}",
        f"Mean episode return: {metrics['ep_return']:.2f}",
        f"Mean accepted lane changes: {metrics['lane_changes']:.2f} / episode",
        f"Mean speed: {metrics['mean_speed']:.2f} m/s",
        f"Mean episode length: {metrics['ep_length']:.1f} steps",
    )), transform=details.transAxes, va="top", linespacing=1.45)

    results = {(row["route"], row["seed"]): row for row in rows}
    outcomes = {key: row["status"] for key, row in results.items()}
    codes = {status: index for index, status in enumerate(OUTCOMES)}
    matrix = [[codes[outcomes[route, seed]] for seed in SEEDS] for route in routes]
    route_grid = fig.add_subplot(grid[3, 1])
    route_grid.imshow(matrix, cmap=ListedColormap([item[2] for item in OUTCOMES.values()]),
                      vmin=-.5, vmax=len(codes)-.5, aspect="auto", interpolation="nearest")
    route_grid.set_xticks(range(len(SEEDS)), [f"Seed {seed}" for seed in SEEDS])
    labels = [route.removeprefix("ego_d_").replace("ego_r_RING_", "Ring ") for route in routes]
    route_grid.set_yticks(range(len(routes)), labels, fontsize=8)
    route_grid.axhline(21.5, color="white", linewidth=2)
    for y, route in enumerate(routes):
        for x, seed in enumerate(SEEDS):
            row = results[route, seed]
            distance = row["last_stop_clearance_m"]
            clearance = f"{distance:.1f}m" if distance is not None else "—"
            label = f"{OUTCOMES[row['status']][1]} ({clearance})"
            route_grid.text(x, y, label, ha="center", va="center", fontsize=7)
    route_grid.set_title("Final evaluation by route and seed")
    route_grid.set_xlabel("Last stop clearance, metres (speed ≤ 0.5 m/s)\n"
                          "Nearest stop line or queued vehicle; — = no measured stop")

    signals = "All green" if summary["all_green"] else "Randomized signal starts"
    traffic = (f"Central demand: {summary['traffic_vph']} veh/h"
               if summary["background"] else "No background traffic")
    fig.suptitle(f"Phase {summary['phase']} PPO | {run.name}\n"
                 f"{signals} · {traffic} · Deterministic final evaluation")
    fig.savefig(out / "learning_curve.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="학습 결과 폴더 또는 해당 model.pt 경로")
    parser.add_argument("--traffic-vph", type=int, help="Phase 2의 고정 중앙 교통량 (기본 800)")
    args = parser.parse_args()
    run = args.run.resolve()
    if run.is_file() and run.name == "model.pt":
        run = run.parent
    model = run / "model.pt"
    if not model.is_file():
        parser.error(f"저장된 모델을 찾을 수 없습니다. {model}")

    road, mdp, training = load_saved_configs(str(model))
    all_green = road["NOHYEONG"]["all_signals_green"]
    config_path = run / "parallel_config.json"
    saved = json.loads(config_path.read_text()) if config_path.is_file() else {}
    settings = saved.get("traffic_curriculum")
    curriculum = TrafficCurriculum(**settings) if settings else None
    if road["TRAFFIC"]["background_enabled"] and curriculum is None:
        parser.error("배경 차량 평가에는 병렬 학습에서 저장한 교통량 설정이 필요합니다.")
    vph = args.traffic_vph if args.traffic_vph is not None else (800 if curriculum else 0)
    if not 0 <= vph <= (curriculum.max_vph if curriculum else 0):
        parser.error("교통량은 저장된 환경이 지원하는 범위 안이어야 합니다.")

    rows = []
    with TemporaryDirectory(prefix="sumo-evaluation-") as directory:
        root = Path(directory)
        config = road_builder.build(road["ROAD"], road["EGO"], road["TRAFFIC"], str(root / "sumo"))
        non_ring_routes, ring_routes = read_ego_route_groups(root / "sumo/highway.ego.rou.xml")
        if (len(non_ring_routes), len(ring_routes)) != (18, 5):
            parser.error("평가에는 Ring 미경유 18개와 Ring 경유 5개의 고유 경로가 필요합니다.")
        routes = non_ring_routes + ring_routes
        kwargs = dict(cfg_path=config, road=road["ROAD"], ego=road["EGO"], mdp_sim=mdp["SIMULATION"],
                      mdp_obs=mdp["OBSERVATION"], action=mdp["ACTION"], reward=training["REWARD"],
                      traffic=road["TRAFFIC"])
        env = ParallelSumo(len(routes), kwargs, root / "parallel", seed=SEEDS[0],
                           randomize_signals=not all_green,
                           randomize_departures=saved.get("randomize_departures", False),
                           independent_episode_rng=saved.get("independent_episode_rng", False),
                           traffic_curriculum=curriculum,
                           fixed_traffic_vph=vph if curriculum else None)
        agent = PPO(obs_dim=env.prototype.obs_dim, act_dim=env.prototype.act_dim, **training["HPARAMS"])
        agent.policy.load_state_dict(torch.load(model, map_location="cpu", weights_only=True))
        for seed in SEEDS:
            env.seed = seed
            results = evaluate_parallel_routes(agent.policy, env, agent.device, routes)
            rows.extend({"route": route, "seed": seed, **result}
                        for route, result in zip(routes, results, strict=True))
            arrivals = sum(result["status"] == "arrival" for result in results)
            print(f"Seed {seed} | 완주 {arrivals}/{len(routes)}", flush=True)

    rows.sort(key=lambda row: (row["route"], row["seed"]))
    out = run / "analysis"
    if args.traffic_vph is not None:
        out /= f"traffic_{vph}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "route_evaluation.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "observation_dim": env.prototype.obs_dim,
        "phase": saved.get("phase", 1 if all_green else 2),
        "deterministic": True,
        "all_green": all_green,
        "randomize_signals": not all_green,
        "background": vph > 0,
        "traffic_vph": vph,
        "seeds": SEEDS,
        "stop_clearance": {
            "sample": "last stopped state in each episode",
            "speed_threshold_m_s": 0.5,
            "reference": "ego front bumper to nearest stop line or stopped leader rear bumper",
            "missing": "no stop, or no reference ahead at the last stop",
        },
        "groups": {
            "all": _summarize(rows),
            "non_ring": _summarize([row for row in rows if row["route"] in non_ring_routes]),
            "ring": _summarize([row for row in rows if row["route"] in ring_routes]),
        },
    }
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _plot(run, routes, rows, summary, out)
    arrivals = summary["groups"]["all"]["statuses"].get("arrival", 0)
    print(f"최종 완주 {arrivals}/{len(rows)} | {out / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
