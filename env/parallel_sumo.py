"""Isolated roads stepped together by one libsumo simulation."""

from copy import copy, deepcopy
from functools import partial
from multiprocessing import get_context
from pathlib import Path
import signal
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from env.ego_routes import group_unique_ego_routes
from env.sumo_env import SumoHighwayEnv
from env.parallel_traffic import TrafficSchedule, parse_traffic_flows

import libsumo


def _replica_id(index, name):
    prefix = f"r{index}_"
    # Internal edges must keep their leading colon.
    return f":{prefix}{name[1:]}" if name.startswith(":") else prefix + name


def _signal_schedule(net, step_length, randomize_signals):
    programs = net.findall("tlLogic")
    if not randomize_signals:
        if any(set(phase.get("state")) - {"G", "g"} for phase in net.iter("phase")):
            raise ValueError("All-green replicas require permanently green traffic lights")
        return (), ()

    schedule = None
    for program in programs:
        phases = program.findall("phase")
        timings = [(phase.get("name"), float(phase.get("duration"))) for phase in phases]
        if (program.get("type") != "static" or float(program.get("offset", "0")) != 0
                or any("next" in phase.attrib for phase in phases)
                or (schedule is not None and timings != schedule)):
            raise ValueError("Signal randomization requires synchronized static signal programs")
        schedule = timings
    if not schedule or not any("r" in phase.get("state").lower() for phase in net.iter("phase")):
        raise ValueError("Signal randomization requires a cycling traffic-light program")
    ticks = [duration / step_length for _, duration in schedule]
    if any(tick <= 0 or not tick.is_integer() for tick in ticks):
        raise ValueError("Signal durations must be positive multiples of the SUMO step length")
    return tuple(program.get("id") for program in programs), tuple(np.cumsum(ticks, dtype=int))


def _write_network(net, destination, count):
    groups = ("edge", "tlLogic", "junction", "connection")
    if {node.tag for node in net} - {*groups, "location"}:
        raise ValueError("Unsupported elements in replicated network")
    location = net.find("location")
    xmin, ymin, xmax, ymax = map(float, location.get("convBoundary").split(","))
    stride_x, stride_y = xmax - xmin + 100, ymax - ymin + 100
    width = min(count, 32)
    bounds = f"{xmin},{ymin},{xmax + (width - 1) * stride_x},{ymax + ((count - 1) // width) * stride_y}"
    location.set("convBoundary", bounds)
    location.set("origBoundary", bounds)
    header = ET.tostring(ET.Element("net", net.attrib), encoding="utf-8", short_empty_elements=False)
    with destination.open("wb") as file:
        file.write(header.removesuffix(b"</net>") + b"\n")
        file.write(ET.tostring(location, encoding="utf-8"))
        for tag in groups:
            templates = net.findall(tag)
            for index in range(count):
                dx, dy = index % width * stride_x, index // width * stride_y
                for template in templates:
                    element = deepcopy(template)
                    for node in element.iter():
                        for attr, value in node.attrib.items():
                            if attr in {"id", "from", "to", "tl"}:
                                node.set(attr, _replica_id(index, value))
                            elif attr in {"incLanes", "intLanes", "via"}:
                                node.set(attr, " ".join(_replica_id(index, name) for name in value.split()))
                            elif attr in {"x", "y"}:
                                node.set(attr, f"{float(value) + (dx if attr == 'x' else dy):.2f}")
                            elif attr == "shape":
                                points = []
                                for point in value.split():
                                    x, y, *z = point.split(",")
                                    points.append(",".join([f"{float(x) + dx:.2f}", f"{float(y) + dy:.2f}", *z]))
                                node.set(attr, " ".join(points))
                    file.write(ET.tostring(element, encoding="utf-8"))
        file.write(b"</net>\n")


def _build_replicas(config, directory, count, step_length, randomize_signals,
                    traffic_curriculum, reference_vph):
    config = Path(config)
    inputs = ET.parse(config).getroot().find("input")
    net = directory / "parallel.net.xml"
    routes = directory / "parallel.rou.xml"
    types, original_routes, flows = [], {}, []
    for filename in inputs.find("route-files").get("value").split(","):
        for node in ET.parse(config.parent / filename).getroot():
            if node.tag in {"vType", "vTypeDistribution"}:
                types.append(node)
            elif node.tag == "route":
                original_routes[node.get("id")] = node.get("edges").split()
            elif node.tag == "flow" and traffic_curriculum is not None:
                flows.append(node)
            else:
                raise ValueError(f"Replicas require a road without background demand, got {node.tag}")

    directory.mkdir(parents=True, exist_ok=True)
    net_path = config.parent / inputs.find("net-file").get("value")
    source = ET.parse(net_path).getroot()
    ego_routes = {name: edges[0] for name, edges in original_routes.items()
                  if name == "r0" or name.startswith(("ego_d_", "ego_r_"))}
    traffic_flows = (parse_traffic_flows(flows, original_routes, net_path, reference_vph, traffic_curriculum)
                     if traffic_curriculum is not None else ())
    signal_ids, signal_end_ticks = _signal_schedule(source, step_length, randomize_signals)
    _write_network(source, net, count)
    route_xml = ET.Element("routes")
    route_xml.extend(types)
    for index in range(count):
        for name, edges in original_routes.items():
            ET.SubElement(route_xml, "route", id=_replica_id(index, name),
                          edges=" ".join(_replica_id(index, edge) for edge in edges))
    ET.ElementTree(route_xml).write(routes, encoding="utf-8")

    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value=net.name)
    ET.SubElement(inputs, "route-files", value=routes.name)
    result = directory / "parallel.sumocfg"
    ET.ElementTree(root).write(result, encoding="utf-8")
    return str(result), ego_routes, signal_ids, signal_end_ticks, traffic_flows, group_unique_ego_routes(original_routes)


class _ReplicaEgo(SumoHighwayEnv):
    _api = libsumo

    def _ego_is_present(self):
        return self.present

    def _destination_edge(self):
        return super()._destination_edge().split("_", 1)[1]

    def _neighbor(self, offset, leader):
        if self.background is not None and self.background.vph:
            return super()._neighbor(offset, leader)
        return None

    def _scan_lane_neighbors(self, direction):
        if self.background is not None and self.background.vph:
            return super()._scan_lane_neighbors(direction)
        return None, None


class ParallelSumo:
    def __init__(self, num_envs, env_kwargs, directory, seed, *, randomize_signals=False,
                 randomize_departures=False, independent_episode_rng=False,
                 traffic_curriculum=None, fixed_traffic_vph=None):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.seed = seed
        self.randomize_departures = randomize_departures
        self.independent_episode_rng = independent_episode_rng or randomize_departures
        self.traffic_curriculum = traffic_curriculum
        self.fixed_traffic_vph = fixed_traffic_vph
        if fixed_traffic_vph is not None and (
                traffic_curriculum is None or not 0 <= fixed_traffic_vph <= traffic_curriculum.max_vph):
            raise ValueError("Fixed traffic requires a traffic profile and a value within its limits")
        self.prototype = _ReplicaEgo(**env_kwargs)
        self.prototype.background = None
        if (self.prototype.road_type != "nohyeong" or self.prototype.warmup_seconds
                or self.prototype.prefill_enabled or self.prototype.gui):
            raise ValueError("ParallelSumo requires a headless intersection without global warm-up")
        if randomize_signals and not self.prototype.signal_enabled:
            raise ValueError("Randomized signals require signal observations")
        (self.config, self.route_entries, self.signal_ids, self.signal_end_ticks,
         self.traffic_flows, self.route_groups) = (
            _build_replicas(env_kwargs["cfg_path"], directory, num_envs, self.prototype.step_length,
                            randomize_signals, traffic_curriculum,
                            self.prototype.traffic.get("vehs_per_hour", 0)))
        self.routes = tuple(sorted(self.route_entries))
        self.departures = {}
        self.egos = []
        self.started = False
        self._live_background = set()

    def _start(self, config):
        libsumo.start(["sumo", "-c", config, "--step-length", str(self.prototype.step_length),
                       "--seed", str(self.seed), "--collision.action", "remove",
                       "--no-warnings", "true", "--no-step-log", "true"])
        self.started = True

    def _find_departures(self):
        if self.independent_episode_rng:
            self._start(self.config)
            try:
                for route, edge in self.route_entries.items():
                    replica_edge = _replica_id(0, edge)
                    lanes = []
                    for index in range(libsumo.edge.getLaneNumber(replica_edge)):
                        lane = f"{edge}_{index}"
                        allowed = libsumo.lane.getAllowed(_replica_id(0, lane))
                        disallowed = libsumo.lane.getDisallowed(_replica_id(0, lane))
                        if (not allowed or "passenger" in allowed) and "passenger" not in disallowed:
                            lanes.append((lane, 0.0, index))
                    if not lanes:
                        raise ValueError(f"No passenger departure lanes on {edge}")
                    if not self.randomize_departures:
                        configured = self.prototype._depart_lane_for(replica_edge)
                        if not configured.isdigit():
                            raise ValueError("Fixed departure requires a numeric or center lane")
                        fixed_lane = int(configured)
                        lanes = [lane for lane in lanes if lane[2] == fixed_lane]
                        if not lanes:
                            raise ValueError(f"Configured departure lane is unavailable on {edge}")
                    self.departures[route] = tuple(lanes)
            finally:
                self.close()
            return
        # Query the exact lane SUMO picks for every route under the baseline's
        # departLane setting. moveTo can then insert there without a global tick.
        ego = self.prototype
        self._start(ego.cfg_path)
        try:
            for route in self.routes:
                ego._spawn_ego(route)
                libsumo.simulationStep()
                lane = libsumo.vehicle.getLaneID(ego.ego_id)
                if not lane:
                    raise RuntimeError(f"SUMO did not insert the ego for route {route}")
                position = libsumo.vehicle.getLanePosition(ego.ego_id)
                self.departures[route] = ((lane, position, libsumo.vehicle.getLaneIndex(ego.ego_id)),)
                libsumo.vehicle.remove(ego.ego_id)
        finally:
            self.close()

    def _respawn(self, ego, route=None, traffic_vph=None, signal_tick=None):
        if ego.present:
            libsumo.vehicle.remove(ego.ego_id)
        if route is None:
            route = ego._sample_ego_route(self.route_groups) if ego.ego_route_random else "r0"
        ego.signal_offset_s = None
        if self.signal_ids:
            offset = ego.signal_random.integers(self.signal_end_ticks[-1]) if signal_tick is None else signal_tick
            ego.signal_offset_s = float(offset * ego.step_length)
            phase = int(np.searchsorted(self.signal_end_ticks, offset, side="right"))
            remaining = float((self.signal_end_ticks[phase] - offset) * ego.step_length)
            for signal in self.signal_ids:
                tls_id = _replica_id(ego.replica_index, signal)
                libsumo.trafficlight.setPhase(tls_id, phase)
                libsumo.trafficlight.setPhaseDuration(tls_id, remaining)
        candidates = self.departures[route]
        lane, position, lane_index = candidates[ego.lane_random.integers(len(candidates))]
        ego.start_lane = lane_index
        if ego.background is not None:
            vph = ego.background.vph if traffic_vph is None else traffic_vph
            ego.background.reset(libsumo, libsumo.simulation.getTime(), vph,
                                 ego.traffic_random, self._live_background)
        ego.episode += 1
        ego.ego_id = _replica_id(ego.replica_index, f"ego_{ego.episode}")
        ego._spawn_ego(_replica_id(ego.replica_index, route), str(lane_index))
        libsumo.vehicle.moveTo(ego.ego_id, _replica_id(ego.replica_index, lane), position)
        ego.present = True
        return ego._get_obs()

    def reset(self, routes=None, *, replica_seeds=None, signal_ticks=None):
        if routes is not None and (len(routes) != self.num_envs or set(routes) - set(self.routes)):
            raise ValueError("Expected one valid route per replica")
        if signal_ticks is not None and (
                not self.signal_ids or len(signal_ticks) != self.num_envs
                or any(not isinstance(tick, (int, np.integer)) or
                       not 0 <= tick < self.signal_end_ticks[-1] for tick in signal_ticks)):
            raise ValueError("Expected one valid signal tick per replica")
        self.close()
        if not self.departures:
            self._find_departures()
        self._start(self.config)
        self._live_background.clear()
        self.egos = []
        observations = []
        if replica_seeds is None:
            replica_seeds = np.random.SeedSequence(self.seed).spawn(self.num_envs)
        for index, seed in enumerate(replica_seeds):
            ego = copy(self.prototype)
            ego.replica_index = index
            ego.episode = 0
            ego.present = False
            # Recreating the parent keeps repeated resets with the same seed reproducible.
            seed = np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key,
                                          pool_size=seed.pool_size)
            if self.independent_episode_rng:
                route_seed, lane_seed, signal_seed, traffic_seed = seed.spawn(4)
                ego.np_random = np.random.default_rng(route_seed)
                ego.signal_random = np.random.default_rng(signal_seed)
            else:
                ego.np_random = ego.signal_random = np.random.default_rng(seed)
                lane_seed, traffic_seed = seed.spawn(2)
            ego.lane_random = np.random.default_rng(lane_seed)
            ego.traffic_random = np.random.default_rng(traffic_seed)
            ego.prefix = partial(_replica_id, index)
            ego.background = TrafficSchedule(self.traffic_flows) if self.traffic_flows else None
            ego._rewarded_green_signals = set()
            ego._signal_stop_offsets = {
                (_replica_id(index, tid), link): offset
                for (tid, link), offset in self.prototype._signal_stop_offsets.items()
            }
            ego._signal_stop_lines = {
                (_replica_id(index, tid), link): dict(line, lane_id=_replica_id(index, line["lane_id"]))
                for (tid, link), line in self.prototype._signal_stop_lines.items()
            }
            ego.no_lane_change_edges = {_replica_id(index, edge) for edge in ego.no_lane_change_edges}
            self.egos.append(ego)
            route = routes[index] if routes is not None else None
            vph = self.fixed_traffic_vph
            if vph is None and self.traffic_curriculum is not None:
                vph = self.traffic_curriculum.initial_vph
            tick = signal_ticks[index] if signal_ticks is not None else None
            observations.append(self._respawn(ego, route, traffic_vph=vph, signal_tick=tick))
        return np.stack(observations)

    def _advance(self, egos, actions):
        states = [ego._prepare_step(action) for ego, action in zip(egos, actions, strict=True)]
        if self.traffic_flows:
            now = libsumo.simulation.getTime()
            for ego in egos:
                ego.background.spawn_due(libsumo, now, ego.prefix, ego.traffic_random)
        libsumo.simulationStep()
        active = set(libsumo.vehicle.getIDList())
        if self.traffic_flows:
            self._live_background = active | set(libsumo.simulation.getPendingVehicles())
        collided = set(libsumo.simulation.getCollidingVehiclesIDList())
        arrived = set(libsumo.simulation.getArrivedIDList())
        for ego, state in zip(egos, states, strict=True):
            ego.present = ego.ego_id in active
            yield ego._finish_step(
                state, collided=ego.ego_id in collided, arrived=ego.ego_id in arrived)

    def step(self, actions):
        transitions = []
        for ego, (obs, reward, terminated, truncated, info) in zip(
                self.egos, self._advance(self.egos, actions), strict=True):
            timeout_obs = obs if truncated and not terminated else None
            if terminated or truncated:
                vph = self.fixed_traffic_vph
                if vph is None and ego.background is not None:
                    success = info["arrived"] and not (info["red_light_violation"] or info["collided"]
                                                       or info["wrong_lane"])
                    vph = self.traffic_curriculum.after_episode(ego.background.vph, success)
                obs = self._respawn(ego, traffic_vph=vph)
            transitions.append((obs, reward, terminated, truncated, timeout_obs))
        return transitions

    def step_evaluation(self, indices, actions):
        """Advance only active evaluation egos without starting another episode."""
        egos = [self.egos[index] for index in indices]
        for ego, transition in zip(egos, self._advance(egos, actions), strict=True):
            transition[4].update(start_lane=ego.start_lane, signal_offset_s=ego.signal_offset_s,
                                 traffic_vph=ego.background.vph if ego.background is not None else 0)
            clearance = ego.get_stop_clearance()
            if clearance is not None:
                transition[4].update(clearance)
            if transition[2] or transition[3]:
                if ego.present:
                    libsumo.vehicle.remove(ego.ego_id)
                    ego.present = False
                if ego.background is not None:
                    ego.background.reset(libsumo, libsumo.simulation.getTime(), 0,
                                         ego.traffic_random, self._live_background)
            yield transition

    def traffic_levels(self):
        return [ego.background.vph for ego in self.egos] if self.traffic_flows else []

    def close(self):
        if self.started:
            libsumo.close()
            self.started = False


def _run_sumo_worker(conn, env, replica_seeds):
    # The main process handles Ctrl+C and saves the current model.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        while True:
            command, data = conn.recv()
            if command == "reset":
                conn.send((True, env.reset(data, replica_seeds=replica_seeds)))
            elif command == "step":
                rows = env.step(data)
                # One observation array instead of thousands of pickled arrays.
                conn.send((True, (np.stack([row[0] for row in rows]),
                                  [row[1:] for row in rows])))
            elif command == "close":
                break
            elif command == "traffic_levels":
                conn.send((True, env.traffic_levels()))
            else:
                raise ValueError(f"Unknown SUMO worker command: {command}")
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        conn.send((False, traceback.format_exc()))
    finally:
        conn.close()
        env.close()


class ShardedParallelSumo:
    """Collect synchronous rollouts from one independent SUMO per worker."""

    def __init__(self, template, num_workers):
        self.template = template
        self.prototype = template.prototype
        self.num_envs = template.num_envs * num_workers
        self.num_workers = num_workers
        self._workers = []

    def _start(self):
        context = get_context("spawn")
        seeds = np.random.SeedSequence(self.template.seed).spawn(self.num_envs)
        size = self.template.num_envs
        for i in range(self.num_workers):
            conn, child = context.Pipe()
            process = context.Process(target=_run_sumo_worker,
                                      args=(child, self.template, seeds[i * size:(i + 1) * size]))
            try:
                process.start()
            except BaseException:
                conn.close()
                raise
            finally:
                child.close()
            self._workers.append((process, conn))

    def _exchange(self, command, payloads):
        for (_, conn), data in zip(self._workers, payloads, strict=True):
            conn.send((command, data))
        for i, (_, conn) in enumerate(self._workers):
            try:
                ok, result = conn.recv()
            except (EOFError, ConnectionError) as exc:
                raise RuntimeError(f"SUMO worker {i} disconnected") from exc
            if not ok:
                raise RuntimeError(f"SUMO worker {i} failed:\n{result}")
            yield result

    def reset(self, routes=None):
        if routes is not None and len(routes) != self.num_envs:
            raise ValueError("Expected one route per environment")
        if not self._workers:
            self._start()
        size = self.template.num_envs
        batches = (None if routes is None else routes[i:i + size]
                   for i in range(0, self.num_envs, size))
        return np.concatenate(list(self._exchange("reset", batches)))

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError("Expected one action per environment")
        size = self.template.num_envs
        batches = (actions[i:i + size] for i in range(0, self.num_envs, size))
        return [(obs, *rest) for observations, rows in self._exchange("step", batches)
                for obs, rest in zip(observations, rows, strict=True)]

    def traffic_levels(self):
        if self.template.traffic_curriculum is None:
            return []
        return [value for shard in self._exchange("traffic_levels", [None] * self.num_workers)
                for value in shard]

    def close(self):
        for _, conn in self._workers:
            try:
                conn.send(("close", None))
            except (EOFError, ConnectionError):
                pass
            finally:
                conn.close()
        for process, _ in self._workers:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()
        self._workers.clear()
