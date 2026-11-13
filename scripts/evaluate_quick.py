"""Evaluate 23 unique routes at 50 fixed signal offsets across independent SUMO workers."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.evaluate import OUTCOMES, _plot_training_panels, _summarize, plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from algorithms.ppo import PPO
from env import road_builder
from env.ego_routes import read_ego_route_groups
from env.parallel_sumo import ParallelSumo
from env.parallel_traffic import TrafficCurriculum
from test import load_saved_configs
from utils.evaluator import evaluate_parallel_routes


# 1.0, 3.5, ..., 123.5 s: samples green, yellow and all-red for every approach.
TICKS = tuple(range(2, 250, 5))
STEP = 0.5


def _evaluate_worker(model, config, work, saved, vph, cases):
    import torch

    torch.set_num_threads(1)
    road, mdp, train = load_saved_configs(str(model))
    settings = saved.get("traffic_curriculum")
    curriculum = TrafficCurriculum(**settings) if settings else None
    kwargs = dict(cfg_path=config, road=road["ROAD"], ego=road["EGO"],
                  mdp_sim=mdp["SIMULATION"], mdp_obs=mdp["OBSERVATION"],
                  action=mdp["ACTION"], reward=train["REWARD"], traffic=road["TRAFFIC"])
    env = ParallelSumo(len(cases), kwargs, work, seed=0, randomize_signals=True,
                       randomize_departures=saved.get("randomize_departures", False),
                       independent_episode_rng=saved.get("independent_episode_rng", False),
                       traffic_curriculum=curriculum,
                       fixed_traffic_vph=vph if curriculum else None)
    if env.prototype.step_length != STEP or env.signal_end_ticks[-1] != 250:
        raise ValueError("Quick evaluation requires a 125 s signal cycle and 0.5 s simulation steps")
    agent = PPO(obs_dim=env.prototype.obs_dim, act_dim=env.prototype.act_dim,
                **(train["HPARAMS"] | {"device": "cpu"}))
    agent.policy.load_state_dict(torch.load(model, map_location="cpu", weights_only=True))
    rows = evaluate_parallel_routes(
        agent.policy, env, agent.device, [route for _, route, _ in cases],
        signal_ticks=[tick for _, _, tick in cases],
        replica_seeds=[np.random.SeedSequence([0, route_index, tick])
                       for route_index, _, tick in cases])
    return [{"route": route, **row} for (_, route, _), row in zip(cases, rows, strict=True)]


def _red_wait_stats(rows):
    distances = [row["red_wait_clearance_m"] for row in rows
                 if row["red_wait_clearance_m"] is not None]
    if not distances:
        return dict(n=0, mean_m=None, median_m=None, q1_m=None, q3_m=None)
    q1, median, q3 = np.percentile(distances, [25, 50, 75]).tolist()
    return dict(n=len(distances), mean_m=float(np.mean(distances)), median_m=median,
                q1_m=q1, q3_m=q3)


def _plot_clearances(axis, routes, summary):
    axis.set_title("Red-light waiting clearance (m)", pad=28)
    axis.set_xlim(-.5, 4.5)
    axis.set_xticks([])
    axis.tick_params(axis="y", left=False, labelleft=False)
    columns = ("Mean", "Median", "Q1", "Q3", "n")
    for i, label in enumerate(columns):
        axis.text(i, 1.01, label, transform=axis.get_xaxis_transform(),
                  ha="center", va="bottom", weight="bold", fontsize=8)
    for y, route in enumerate(routes):
        stats = summary["by_route"][route]["red_wait_clearance"]
        if y % 2 == 0:
            axis.axhspan(y - .5, y + .5, color="#f2f4f7", zorder=0)
        for i, key in enumerate(("mean_m", "median_m", "q1_m", "q3_m", "n")):
            value = stats[key]
            label = str(value) if key == "n" else ("—" if value is None else f"{value:.1f}")
            axis.text(i, y, label, ha="center", va="center", fontsize=8)
    stats = summary["all"]["red_wait_clearance"]
    values = [stats[k] for k in ("mean_m", "median_m", "q1_m", "q3_m")]
    totals = "  |  ".join(f"{label} {value:.1f}" for label, value in zip(columns, values)
                        if value is not None)
    axis.set_xlabel(f"All routes (n={stats['n']})\n{totals or 'No red-light stops'}",
                    fontsize=9, labelpad=8)


def _plot(run, out, routes, rows, summary):
    fig = plt.figure(figsize=(18, 16), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=(1, 1, 1, 3.3))
    _plot_training_panels(fig, grid, run)
    final = grid[3, :].subgridspec(1, 3, width_ratios=(1.0, 1.75, 1.05))
    rates = fig.add_subplot(final[0, 0])
    offsets = fig.add_subplot(final[0, 1], sharey=rates)
    clearance = fig.add_subplot(final[0, 2], sharey=rates)
    _plot_clearances(clearance, routes, summary)
    labels = [route.removeprefix("ego_d_").replace("ego_r_RING_", "Ring ") for route in routes]
    counts = {route: Counter() for route in routes}
    for row in rows:
        counts[row["route"]][row["status"]] += 1
    y = np.arange(len(routes))
    left = np.zeros(len(routes))
    for key, (label, _, color) in OUTCOMES.items():
        values = np.array([counts[route][key] for route in routes])
        rates.barh(y, values, left=left, height=.72, color=color)
        left += values
    for i, route in enumerate(routes):
        rates.text(len(TICKS) + 1, i, f"{counts[route]['arrival']}/{len(TICKS)}",
                   ha="left", va="center", fontsize=8)
    rates.set(xlim=(0, len(TICKS) + 10), title="Final outcomes by route",
              xlabel="Episodes (one per signal offset)")
    rates.set_xticks(range(0, len(TICKS) + 1, 10))
    rates.set_yticks(y, labels, fontsize=8)
    rates.grid(axis="x", alpha=.2)
    rates.set_axisbelow(True)

    codes = {key: i for i, key in enumerate(OUTCOMES)}
    index = {(row["route"], row["signal_offset_s"]): codes[row["status"]] for row in rows}
    matrix = [[index[route, tick * STEP] for tick in TICKS] for route in routes]
    colors = [item[2] for item in OUTCOMES.values()]
    spacing = (TICKS[1] - TICKS[0]) * STEP
    offsets.imshow(matrix, cmap=ListedColormap(colors),
                   norm=BoundaryNorm(np.arange(len(colors) + 1) - .5, len(colors)),
                   interpolation="nearest", aspect="auto",
                   extent=(TICKS[0] * STEP - spacing / 2, TICKS[-1] * STEP + spacing / 2,
                           len(routes) - .5, -.5))
    offsets.set(xlim=(0, 125), ylim=(len(routes) - .5, -.5),
                title="Final outcome by route and signal start",
                xlabel="Signal offset at episode start (s)")
    offsets.set_xticks(np.arange(0, 126, 12.5))
    offsets.tick_params(axis="y", left=False, labelleft=False)
    for boundary in np.arange(.5, len(routes) - .5, 1):
        offsets.axhline(boundary, color="white", linewidth=.35, alpha=.7)
    for axis in (rates, offsets, clearance):
        axis.axhline(21.5, color="#374151", linewidth=.8)
    handles = [Patch(color=color, label=label) for label, _, color in OUTCOMES.values()]
    offsets.legend(handles=handles, ncol=3, fontsize=8, loc="upper center",
                   bbox_to_anchor=(.5, -.10), frameon=False)
    total = summary["all"]["episodes"]
    arrived = summary["all"]["statuses"].get("arrival", 0)
    lane_mode = "sampled start lane" if summary["randomize_departures"] else "fixed start lane"
    fig.suptitle(f"{run.name} | Quick evaluation: {arrived}/{total} arrivals ({arrived / total:.1%})\n"
                 f"Deterministic policy · {lane_mode} · Central demand {summary['traffic_vph']} veh/h")
    fig.savefig(out / "learning_curve.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="학습 결과 폴더 또는 model.pt")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--traffic-vph", type=int, help="배경 교통이 있는 모델의 고정 중앙 투입량")
    args = parser.parse_args()
    run = args.run.resolve()
    if run.is_file() and run.name == "model.pt":
        run = run.parent
    model = run / "model.pt"
    if not model.is_file():
        parser.error(f"모델을 찾을 수 없습니다: {model}")
    if not 1 <= args.num_workers <= len(TICKS):
        parser.error(f"--num-workers는 1~{len(TICKS)} 사이여야 합니다.")
    road, _, _ = load_saved_configs(str(model))
    if road["NOHYEONG"]["all_signals_green"]:
        parser.error("빠른 신호 평가에는 신호가 작동하는 모델 설정이 필요합니다.")
    config_file = run / "parallel_config.json"
    saved = json.loads(config_file.read_text()) if config_file.is_file() else {}
    settings = saved.get("traffic_curriculum")
    if road["TRAFFIC"].get("background_enabled", False) and not settings:
        parser.error("배경 교통에는 저장된 커리큘럼 설정이 필요합니다.")
    maximum = settings["max_vph"] if settings else 0
    vph = min(800, maximum) if args.traffic_vph is None else args.traffic_vph
    if not 0 <= vph <= maximum:
        parser.error(f"--traffic-vph는 0~{maximum} 사이여야 합니다.")

    start = perf_counter()
    with TemporaryDirectory(prefix="sumo-quick-eval-") as directory:
        root = Path(directory)
        config = road_builder.build(road["ROAD"], road["EGO"], road["TRAFFIC"], str(root / "base"))
        non_ring_routes, ring_routes = read_ego_route_groups(root / "base/highway.ego.rou.xml")
        if (len(non_ring_routes), len(ring_routes)) != (18, 5):
            parser.error("평가에는 Ring 미경유 18개와 Ring 경유 5개의 고유 경로가 필요합니다.")
        routes = non_ring_routes + ring_routes
        chunks = np.array_split(TICKS, args.num_workers)
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=get_context("spawn")) as pool:
            jobs = [pool.submit(_evaluate_worker, model, config, root / f"worker_{i}", saved, vph,
                                [(j, route, int(tick)) for j, route in enumerate(routes) for tick in chunk])
                    for i, chunk in enumerate(chunks)]
            rows = [row for job in jobs for row in job.result()]
    rows.sort(key=lambda row: (row["route"], row["signal_offset_s"]))
    out = run / "analysis" / ("quick" if args.traffic_vph is None else f"quick_traffic_{vph}")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "route_evaluation.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def summarize(group):
        return {**_summarize(group), "red_wait_clearance": _red_wait_stats(group)}

    summary = {
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "phase": saved.get("phase"),
        "num_workers": args.num_workers,
        "traffic_vph": vph,
        "deterministic": True,
        "randomize_departures": saved.get("randomize_departures", False),
        "signal_offsets_s": [tick * STEP for tick in TICKS],
        "red_wait_definition": "Last measurable stop per episode with speed <= 0.5 m/s "
                               "and nearest observed signal red; nearest stop line or stopped leader.",
        "all": summarize(rows),
        "by_route": {route: summarize([row for row in rows if row["route"] == route])
                     for route in routes},
    }
    _plot(run, out, routes, rows, summary)
    summary["elapsed_seconds"] = round(perf_counter() - start, 2)
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    arrived = summary["all"]["statuses"].get("arrival", 0)
    print(f"완주 {arrived}/{len(rows)} | {summary['elapsed_seconds']}초 | {out / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
