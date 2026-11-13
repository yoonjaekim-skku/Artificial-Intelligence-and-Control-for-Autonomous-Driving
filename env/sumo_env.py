"""SUMO 고속도로 RL 환경 — Gymnasium 인터페이스 + TraCI.

┌─────────────────────────────────────────────────────────────┐
│  강화학습 루프에서 이 클래스의 역할                              │
│                                                             │
│    obs, _ = env.reset()          # 에피소드 시작, 초기 관측     │
│    obs, r, term, trunc, _ = env.step(action)  # 1스텝 진행    │
│                                                             │
│  내부적으로는 TraCI(Traffic Control Interface)라는 소켓 API로  │
│  SUMO 시뮬레이터를 원격 조종한다:                               │
│    traci.simulationStep()  → 시뮬레이션을 1스텝 진행            │
│    traci.vehicle.setSpeed() → 차량 속도 명령                   │
│    traci.vehicle.getSpeed() 등 → 상태 조회                     │
└─────────────────────────────────────────────────────────────┘

관측/행동/보상의 구체적인 정의는 configs/mdp_config.py 에 있고,
이 파일은 그 설정을 받아서 실행만 한다. (MDP를 바꿀 때 이 파일 수정 불필요)
"""
from math import sqrt
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.ego_routes import group_unique_ego_routes

# SUMO 설치 폴더의 tools/ 안에 traci, sumolib 파이썬 모듈이 들어있다.
# pip로도 설치했지만, 버전 불일치를 피하려면 SUMO_HOME 쪽을 우선 경로에 추가.
#
# SUMO_HOME이 설정 안 돼 있으면 OS별 기본 설치 경로를 순서대로 탐색해서
# 자동으로 잡아준다 (환경변수 설정을 깜빡해도 동작하도록).
if "SUMO_HOME" not in os.environ:
    _candidates = [
        # macOS (Homebrew) — Apple Silicon / Intel
        "/opt/homebrew/opt/sumo/share/sumo",
        "/usr/local/opt/sumo/share/sumo",
        # Linux (apt)
        "/usr/share/sumo",
        # Windows (기본 인스톨러 경로)
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
    ]
    for _p in _candidates:
        if os.path.isdir(_p):
            os.environ["SUMO_HOME"] = _p
            print(f"[sumo_env] SUMO_HOME 자동 감지: {_p}")
            break
    else:
        raise RuntimeError(
            "SUMO_HOME 환경변수가 없고 기본 경로에서도 SUMO를 찾지 못했습니다.\n"
            "  Windows: 인스톨러 재설치(SUMO_HOME 체크) 또는 환경변수 수동 설정\n"
            "  macOS  : ./setup_mac.sh 실행 또는 README_MAC.md 참고\n"
            "  Linux  : sudo apt install sumo sumo-tools 후 "
            "export SUMO_HOME=/usr/share/sumo")

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci
from sumolib import checkBinary


class SumoHighwayEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    @property
    def _api(self):
        return traci

    def _ego_is_present(self):
        return self.ego_id in self._api.vehicle.getIDList()

    def __init__(self, cfg_path, road, ego, mdp_sim, mdp_obs, action, reward,
                 traffic=None, gui=False, gui_autostart=True):
        """
        Args:
            cfg_path: road_builder.build()가 만든 .sumocfg 경로
            road:     configs.road_config.ROAD
            ego:      configs.road_config.EGO (가속/감속 한계가 여기서 옴)
            mdp_sim:  configs.mdp_config.SIMULATION
            mdp_obs:  configs.mdp_config.OBSERVATION
            action:   configs.mdp_config.ACTION (행동 차원 수)
            reward:   configs.mdp_config.REWARD
            traffic:  configs.road_config.TRAFFIC (프리필/컨트롤러 정보).
                None이면 프리필 없이 flow만으로 동작 (하위호환)
            gui:      True면 sumo-gui(화면 있는 버전)로 실행
            gui_autostart: (gui=True일 때만 의미)
                True  → 창이 뜨면 시뮬레이션이 자동으로 시작
                False → 사용자가 ▶(플레이) 버튼을 누를 때까지 대기.
                        sumo-gui는 --start 옵션이 없으면 TraCI의
                        simulationStep() 호출도 플레이 전까지 블록되므로,
                        에피소드 전체가 "▶를 눌러야 시작"되는 효과가 난다.
        """
        super().__init__()
        self.cfg_path = cfg_path
        self.gui = gui
        self.gui_autostart = gui_autostart

        # ----- 도로/차량 설정에서 필요한 값들 -----
        self.vmax = float(road["speed_limit"])   # 속도 정규화 기준
        self.num_lanes = int(road["num_lanes"])
        # 시작 차선: "center"면 중앙 차선(차선수//2)으로 자동 결정.
        #   3차선 → 1번(가운데), 5차선 → 2번. 차선 수를 바꿔도 항상 가운데라
        #   좌/우 양쪽으로 변경 여지가 대칭으로 생긴다 (탐험에 유리).
        _dl = ego["depart_lane"]
        if str(_dl).lower() == "center":
            self.ego_depart_lane = str(int(road["num_lanes"]) // 2)
        else:
            self.ego_depart_lane = str(_dl)
        # ---- 프리필(도로 전체 사전 배치) 설정 ----
        self.traffic = traffic or {}
        pf = (self.traffic.get("prefill") or {})
        self.prefill_enabled = bool(pf.get("enabled", False))
        self.prefill_start = float(pf.get("start_offset", 80.0))
        self.prefill_end_margin = float(pf.get("end_margin", 50.0))
        self.prefill_jitter = float(pf.get("spacing_jitter", 0.3))
        # ---- 차선변경 안전 게이트 설정 (mdp_config.ACTION["lane_change_safety"]) ----
        lcs = (action.get("lane_change_safety") or {})
        self.lc_safety_enabled = bool(lcs.get("enabled", True))
        self.lc_front_min = float(lcs.get("min_front_gap", 5.0))   # m
        self.lc_rear_min = float(lcs.get("min_rear_gap", 5.0))     # m
        self.lc_front_tau = float(lcs.get("front_tau", 0.3))       # s
        self.lc_rear_tau = float(lcs.get("rear_tau", 0.5))         # s
        self.lc_spawn_protect = float(lcs.get("spawn_protect", 60.0))  # m
        self.lc_horizon = float(lcs.get("approach_horizon", 1.0))       # s
        self.road_length = float(road["length"])
        # ---- 노형오거리(교차로) 모드 설정 ----
        #   직선 도로("straight")는 기존 동작 그대로. "nohyeong" 이면:
        #     · prefill(도로에 배경차를 미리 줄 세워 두는 기능, 직선 도로 전용) 대신
        #       warmup_seconds 동안 배경차만 먼저 돌려 교차로를 채움
        #     · ego 경로를 에피소드마다 무작위로 뽑음 (원형도로는 ring_prob 확률로만)
        self.road_type = str(road.get("type", "straight"))
        self.ego_route_random, self.ego_ring_prob, self.warmup_seconds = False, 0.0, 0.0
        self.no_lane_change_edges = set()
        if self.road_type == "nohyeong":
            from env.road_config import NOHYEONG
            er = NOHYEONG.get("ego_routes", {})
            self.ego_route_random = bool(er.get("random", False))
            self.ego_ring_prob = float(er.get("ring_prob", 0.0))
            self.warmup_seconds = float(NOHYEONG.get("warmup_seconds", 90))
            self.no_lane_change_edges = set(NOHYEONG.get("no_lane_change_edges", []))
        self.ego_route_id = "r0"     # 이번 에피소드에 ego 가 타는 경로 (reset 에서 정해짐)
        self._ego_depart_lane_cfg = str(ego["depart_lane"])
        self.traffic_max_speed = float(self.traffic.get("max_speed", self.vmax))
        # 차선당 배치 간격: 지정값이 없으면 교통류 이론대로 자동 계산
        #   차선당 유량 q_lane = (vehs/h) / 3600 / 차선수  [대/초]
        #   간격 = 속도 / q_lane   (그 속도로 흐를 때 차 사이 평균 거리)
        if pf.get("spacing"):
            self.prefill_spacing = float(pf["spacing"])
        else:
            vph = float(self.traffic.get("vehs_per_hour", 1800))
            q_lane = max(vph / 3600.0 / max(self.num_lanes, 1), 1e-6)
            self.prefill_spacing = self.traffic_max_speed / q_lane
        # 컨트롤러 타입 샘플링 준비 (rou.xml의 car_<이름> vType과 대응)
        ctrls = self.traffic.get("controllers") or {}
        if ctrls:
            names = list(ctrls.keys())
            probs = np.array([float(ctrls[n].get("probability", 1.0)) for n in names])
            self._prefill_types = [f"car_{n}" for n in names]
            self._prefill_probs = probs / probs.sum()
        else:
            self._prefill_types, self._prefill_probs = ["car"], np.array([1.0])
        self.ego_depart_speed = str(ego["depart_speed"])
        # 행동 a ∈ [-1,1] 을 실제 가속도로 변환할 때 쓰는 스케일:
        #   a = +1 → +max_accel (풀 액셀),  a = -1 → -max_decel (풀 브레이크)
        self.max_accel = float(ego["accel"])   # m/s^2
        self.max_decel = float(ego["decel"])   # m/s^2 (양수로 저장, 부호는 계산 시)

        # ----- MDP 설정 -----
        self.step_length = float(mdp_sim["step_length"])
        self.max_steps = int(mdp_sim["max_steps"])
        self.warmup_steps = int(mdp_sim["warmup_steps"])
        self.visibility = float(mdp_obs["visibility"])   # 관측 거리 W
        # 밀도 정규화용 "차량 1대가 점유하는 길이" (차체 + 최소 간격)
        self.density_veh_len = float(mdp_obs.get("density_veh_length", 7.5))
        self.connectivity_half_width = int(mdp_obs.get("connectivity_half_width", 2))
        if self.connectivity_half_width < 0:
            raise ValueError("connectivity_half_width must be nonnegative")
        self.connectivity_offsets = tuple(range(self.connectivity_half_width,
                                                 -self.connectivity_half_width - 1, -1))
        # 누락된 키는 False: 기존 저장 모델의 관측 레이아웃을 유지한다.
        self.lookahead_connectivity_enabled = bool(mdp_obs.get("lookahead_connectivity_enabled", False))
        # 추가 연결성이 내다보는 route edge 수. 기본 2는 기존 모델과 같은 관측.
        # 노형오거리 진입로는 [긴 도로 → 짧은 구간 2개 → 교차로]라서 2로는 긴 도로에서
        # 회전 차선을 알 수 없다 (예: A1 출발 5차선 중 A4행은 맨 왼쪽만 가능).
        self.lookahead_edges = int(mdp_obs.get("lookahead_edges", 2))
        # 연결 차선 ID는 정적 정보다. 신호색/진입 가능 시점은 캐시하지 않는다.
        self._lane_successors_cache = {}
        self.destination_enabled = bool(mdp_obs.get("destination_enabled", False))
        self.destination_arms = tuple(mdp_obs.get("destination_arms", ("A1", "A2", "A3", "A4", "A5")))
        if self.destination_enabled and (not self.destination_arms
                                        or len(set(self.destination_arms)) != len(self.destination_arms)):
            raise ValueError("destination_arms must contain unique destination arms")
        self.signal_enabled = bool(mdp_obs.get("signal_enabled", False))
        self.signal_visibility = float(mdp_obs.get("signal_visibility", 100.0))
        self.signal_count = int(mdp_obs.get("signal_count", 1))
        if self.signal_count < 1:
            raise ValueError("signal_count must be at least 1")
        if self.signal_visibility <= 0:
            raise ValueError("signal_visibility must be positive")
        self._signal_stop_offsets = {}
        self._signal_stop_lines = {}
        if self.signal_enabled:
            config = ET.parse(cfg_path).getroot()
            net_file = config.find("input/net-file").get("value")
            net = ET.parse(os.path.join(os.path.dirname(cfg_path), net_file)).getroot()
            lane_offsets = {}
            for edge in net.findall("edge"):
                offset = edge.find("stopOffset")
                default = float(offset.get("value")) if offset is not None else 0.0
                for lane in edge.findall("lane"):
                    offset = lane.find("stopOffset")
                    lane_offsets[lane.get("id")] = (float(offset.get("value"))
                                                     if offset is not None else default)
            for connection in net.findall("connection"):
                if connection.get("tl") is not None:
                    key = (connection.get("tl"), int(connection.get("linkIndex")))
                    lane = connection.get("from") + "_" + connection.get("fromLane")
                    self._signal_stop_offsets[key] = lane_offsets.get(lane, 0.0)
                    lane_node = net.find(f"edge/lane[@id='{lane}']")
                    self._signal_stop_lines[key] = {
                        "lane_id": lane,
                        "lane_position": float(lane_node.get("length")) - lane_offsets.get(lane, 0.0),
                    }
        self.act_dim = int(action["dim"])
        self.accel_index = int(action.get("accel_index", 0))
        self.lane_change_index = int(action.get("lane_change_index", 1))
        self.lane_change_threshold = float(action.get("lane_change_threshold", 0.5))
        self.lane_change_duration = float(action.get("lane_change_duration", 1.0))
        self.rw = reward

        # GUI는 ego를 따라가며 가까이 보이도록 기본값을 둔다.
        self.gui_zoom = float(mdp_sim.get("gui_zoom", 1200.0))
        self.gui_delay = float(mdp_sim.get("gui_delay", 100.0))
        self.gui_track_ego = bool(mdp_sim.get("gui_track_ego", True))

        self.ego_id = "ego"        # SUMO 안에서 ego 차량을 식별하는 ID
        self.step_count = 0
        self._started = False      # SUMO 프로세스가 떠 있는지 여부
        self._rewarded_green_signals = set()
        self._green_stop_since = None
        self._red_safe_zone_since = None
        self._red_safe_zone_hold_paid = {}

        # Gymnasium 행동 공간:
        #   action[0] = 종방향 가감속 raw ∈ [-1,1]
        #   action[1] = 차선변경 raw ∈ [-1,1], 환경 내부에서 {-1,0,+1}로 양자화
        # PPO의 Gaussian 정책과 호환하기 위해 Box를 유지한다. 실제 적용된
        # 차선변경 명령은 info["lane_change"]에 정확한 -1/0/+1로 반환한다.
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.act_dim,), dtype=np.float32)
        # 차량/밀도는 ±2차선, 연결성 폭과 목적지 입력은 개별 설정한다.
        # 새 키가 없는 저장 설정은 기존 31차원 + 신호별 5차원을 유지한다.
        self.obs_offsets = (+2, +1, 0, -1, -2)
        n_lanes_view = len(self.obs_offsets)
        self.obs_dim = 1 + n_lanes_view * 4 + len(self.connectivity_offsets) + n_lanes_view
        # 정지선 근처 정밀 거리(선택): 신호당 1차원 추가. 기본 0 = 기존 레이아웃 유지.
        self.signal_fine_scale = float(mdp_obs.get("signal_fine_distance_scale", 0.0))
        self._signal_obs_size = 5 + (1 if self.signal_fine_scale > 0 else 0)
        self.obs_dim += self._signal_obs_size * self.signal_count if self.signal_enabled else 0
        self.obs_dim += len(self.connectivity_offsets) if self.lookahead_connectivity_enabled else 0
        self.obs_dim += len(self.destination_arms) if self.destination_enabled else 0
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,),
                                            dtype=np.float32)

    # ==================================================================
    # SUMO 프로세스 시작
    # ==================================================================
    def _start_sumo(self):
        # gui=True면 화면이 있는 sumo-gui, 아니면 화면 없는 sumo (학습용, 훨씬 빠름)
        binary = checkBinary("sumo-gui" if self.gui else "sumo")
        cmd = [
            binary, "-c", self.cfg_path,
            "--step-length", str(self.step_length),
            # 충돌한 차량을 시뮬레이션에서 제거. (기본값 teleport는 순간이동시켜버려서
            # RL 학습에는 부적합. remove로 해야 "충돌 → 에피소드 종료"가 깔끔함)
            "--collision.action", "remove",
            "--no-warnings", "true",   # 콘솔 경고 끄기
            "--no-step-log", "true",   # 스텝 로그 끄기
            # 매 에피소드 다른 시드 → 배경 교통 패턴이 달라져서 과적합 방지
            "--seed", str(np.random.randint(0, 100000)),
        ]
        if self.gui:
            cmd += ["--quit-on-end"]      # 에피소드 종료 시 창 정리
            cmd += ["--delay", str(self.gui_delay)]
            if self.gui_autostart:
                cmd += ["--start"]        # 자동 재생
            # --start가 없으면: 창이 뜬 뒤 ▶(플레이)를 눌러야 진행된다.
            #   매 에피소드가 reset()에서 SUMO를 새로 띄우므로,
            #   에피소드마다 ▶를 눌러 시작하게 된다.
        self._api.start(cmd)   # SUMO를 자식 프로세스로 띄우고 소켓 연결
        self._started = True

    # ==================================================================
    # 에피소드 초기화
    # ==================================================================
    def _prefill_traffic(self):
        """도로 전 구간에 배경차를 일정 간격(±jitter)으로 미리 배치한다.

        flow는 도로 입구에서만 차를 만들기 때문에, 프리필이 없으면
        에피소드 초반의 긴 도로가 텅 비어 "빈 도로 주행"만 잔뜩 학습된다.
        프리필은 첫 스텝부터 ego가 실제 교통류 한가운데서 시작하게 만든다.

        - 위치: start_offset ~ (길이 - end_margin), 차선별 독립 배치
        - 간격: prefill_spacing × (1 ± jitter) 무작위
        - 타입: controllers 확률 비율로 샘플링 (rou.xml의 vType과 동일)
        - 속도: departSpeed="max" → 각 개체가 "자기 허용 최고속도"로 시작.
          컨트롤러별 max_speed/speedDev 차등이 프리필에도 그대로 반영되고,
          개체별 무작위 speedFactor 때문에 숫자 속도를 주면 SUMO가
          "허용속도 초과"로 거부할 수 있는 문제도 함께 피한다.
        배치된 차들은 SUMO의 각자 컨트롤러(IDM 등)가 정상적으로 운전한다.
        """
        rng = np.random.default_rng()
        count = 0
        # 차선별 간격 배율 (없으면 전 차선 동일). 인덱스 0 = 맨 오른쪽.
        factors = (self.traffic.get("prefill", {})
                   .get("lane_spacing_factor")
                   or [1.0] * self.num_lanes)
        for lane in range(self.num_lanes):
            f = float(factors[lane]) if lane < len(factors) else 1.0
            lane_spacing = self.prefill_spacing * f
            pos = self.prefill_start + rng.uniform(0, lane_spacing)
            while pos < self.road_length - self.prefill_end_margin:
                vid = f"pre_{lane}_{count}"
                vtype = str(rng.choice(self._prefill_types, p=self._prefill_probs))
                try:
                    self._api.vehicle.add(
                        vid, "r0", typeID=vtype,
                        departLane=str(lane),
                        departPos=f"{pos:.1f}",
                        departSpeed="max")
                    count += 1
                except self._api.TraCIException:
                    pass  # 겹침 등으로 실패한 자리는 건너뜀
                step = lane_spacing * rng.uniform(
                    1.0 - self.prefill_jitter, 1.0 + self.prefill_jitter)
                pos += max(step, 10.0)
        return count

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rewarded_green_signals.clear()

        # SUMO 상태를 완전히 초기화하는 가장 확실한 방법은 껐다 켜는 것.
        if self._started:
            self._api.close()
            self._started = False
        self._start_sumo()

        if self.road_type == "nohyeong":
            # 교차로: 배경차만 warmup_seconds 동안 먼저 돌려서 다섯 방향을 자연스럽게 채운다.
            #   (prefill 은 "한 줄짜리 도로에 차를 줄 세우는" 직선 도로 전용 기능이라 쓰지 않는다)
            for _ in range(int(self.warmup_seconds / self.step_length)):
                self._api.simulationStep()
        else:
            # 도로 전 구간에 배경차를 미리 배치 (첫 스텝부터 실제 교통류 상태)
            if self.prefill_enabled:
                self._prefill_traffic()
            # 배치된 차들이 자리를 잡도록 몇 스텝만 정착시킨다.
            for _ in range(self.warmup_steps):
                self._api.simulationStep()

        self._spawn_ego(self._choose_ego_route())
        self._api.simulationStep()  # ego가 실제로 도로에 나타나도록 1스텝 진행

        # ── ego 삽입 보증 ──
        # 도로 입구는 flow가 차를 스폰하는 자리라, 시작 차선 입구가 잠시
        # 막혀 있으면 SUMO가 삽입을 "지연"시킨다. 그 경우 ego 없이
        # 에피소드가 진행되어 즉시 종료되는 버그가 되므로, 삽입될 때까지
        # (최대 30스텝) 기다린다. 지연되어도 요청한 차선은 유지된다.
        _wait = 0
        while not self._ego_is_present() and _wait < 30:
            self._api.simulationStep()
            _wait += 1
        if not self._ego_is_present():
            raise RuntimeError(
                "ego 삽입 실패: 도로 입구가 30스텝 이상 막혀 있습니다. "
                "TRAFFIC['vehs_per_hour']를 낮추거나 depart_speed를 확인하세요.")

        # GUI 기본 시점: ego를 화면 중앙에서 추적하고 충분히 가까이 확대한다.
        # 긴 직선도로 전체를 한 화면에 맞추는 기본 뷰는 차량이 점/삼각형처럼
        # 보이기 쉬우므로 학습/평가 관찰용으로는 이 설정이 더 직관적이다.
        if self.gui and self._ego_is_present():
            try:
                view_ids = self._api.gui.getIDList()
                if view_ids:
                    view_id = view_ids[0]
                    # tracking을 먼저 켜고 zoom을 마지막에 적용해야 SUMO가
                    # 차량 추적을 시작하면서 viewport를 다시 덮어쓰지 않는다.
                    if self.gui_track_ego:
                        self._api.gui.trackVehicle(view_id, self.ego_id)
                    self._api.gui.setZoom(view_id, self.gui_zoom)
            except self._api.TraCIException:
                pass

        return self._get_obs(), {"route": self.ego_route_id}

    def _spawn_ego(self, route_id, depart_lane=None):
        # ego 경로 고르기 + 시작 차선 정하기
        self.ego_route_id = route_id
        self.step_count = 0
        self._rewarded_green_signals.clear()
        self._green_stop_since = None
        self._red_safe_zone_since = None
        self._red_safe_zone_hold_paid = {}
        if depart_lane is None:
            start_edge = self._api.route.getEdges(route_id)[0]
            depart_lane = self._depart_lane_for(start_edge)
        self.ego_depart_lane = depart_lane

        # ego 차량 투입 — 반드시 "도로 시작점(0m)"에서 출발
        self._api.vehicle.add(self.ego_id, self.ego_route_id, typeID="ego_type",
                              departLane=self.ego_depart_lane,
                              departPos="0", departSpeed=self.ego_depart_speed)

        # ★ 중요: speedMode=0 → SUMO의 내장 안전장치(자동 안전제동 등)를 전부 끔.
        #   이걸 꺼야 RL이 속도를 온전히 제어하고, 잘못 배우면 실제로 충돌한다.
        #   (켜두면 SUMO가 알아서 브레이크를 밟아줘서 충돌이 거의 안 나고,
        #    그러면 "충돌 회피"라는 학습 신호 자체가 사라진다)
        self._api.vehicle.setSpeedMode(self.ego_id, 0)
        # SUMO의 자율 차선변경은 끄고, RL이 내리는 TraCI 차선변경 명령만 사용한다.
        self._api.vehicle.setLaneChangeMode(self.ego_id, 0)

    def _choose_ego_route(self) -> str:
        """고유 경로를 실제 순환도로 통과 여부에 따라 뽑는다."""
        if not self.ego_route_random:
            return "r0"
        routes = {rid: self._api.route.getEdges(rid) for rid in self._api.route.getIDList()
                  if rid.startswith(("ego_d_", "ego_r_"))}
        return self._sample_ego_route(group_unique_ego_routes(routes))

    def _sample_ego_route(self, route_groups):
        direct, ring = route_groups
        rng = self.np_random
        if ring and (not direct or rng.random() < self.ego_ring_prob):
            return str(rng.choice(ring))
        if direct:
            return str(rng.choice(direct))
        return "r0"

    def _depart_lane_for(self, edge: str) -> str:
        """EGO["depart_lane"] 설정을 실제 출발 도로의 차선 수에 맞춰 바꾼다.
        (도로마다 차선 수가 달라서, 4차선 기준 번호가 2차선 도로에선 없는 차선이 될 수 있음)"""
        n = self._api.edge.getLaneNumber(edge)
        dl = self._ego_depart_lane_cfg.lower()
        if dl == "center":
            return str(n // 2)
        if dl.isdigit():
            return str(min(int(dl), n - 1))
        return self._ego_depart_lane_cfg

    # ==================================================================
    # 관측 만들기 (레이아웃은 env/mdp_config.py 주석 참고)
    #
    # 핵심 설계: 특정 도로에 대한 하드코딩 없이, 매 스텝 SUMO 네트워크를
    # 실시간 조회해서 "지금 내 주변"을 계산한다. 그래서 도로 구조가
    # 어떻게 바뀌어도(차선 수, 합류, 차선 감소 등) 이 코드는 그대로 동작한다.
    #
    # 사용하는 TraCI API:
    #   getLeader / getFollower : 같은 차선의 앞/뒤 차량 (edge 경계도 넘어 탐색)
    #   getNeighbors(mode)      : 옆 차선의 앞/뒤 차량.
    #       mode 비트: bit0 = 0:왼쪽 / 1:오른쪽,  bit1 = 0:후행 / 1:선행
    #   lane.getLinks           : 차선이 다음 edge로 이어지는지 (연결성 판단)
    # ==================================================================

    # (차선 오프셋, 선행차 getNeighbors mode, 후행차 mode)
    #   오프셋 +1 = 왼쪽 차선 (SUMO는 차선 index가 왼쪽으로 갈수록 커짐)
    _NEIGHBOR_MODES = {+1: (0b10, 0b00),   # 왼쪽: 선행=2, 후행=0
                       -1: (0b11, 0b01)}   # 오른쪽: 선행=3, 후행=1

    def _neighbor(self, offset: int, leader: bool):
        """오프셋 차선(-1=오른쪽, 0=현재, +1=왼쪽)의 가장 가까운
        선행(leader=True)/후행 차량을 (거리 m, 그 차량 속도)로 반환.
        visibility 밖이거나 없으면 None."""
        if offset == 0:
            if leader:
                res = self._api.vehicle.getLeader(self.ego_id, self.visibility)
                if res is None:
                    return None
                vid, gap = res
            else:
                vid, gap = self._api.vehicle.getFollower(self.ego_id, self.visibility)
                if vid == "":
                    return None
        elif abs(offset) >= 2:
            # 두 칸 이상 옆 차선: SUMO getNeighbors는 "바로 옆"만 지원하므로
            # 차선 직접 스캔(_scan_lane_neighbors — 같은 edge의 차량 목록을
            # 훑어 가장 가까운 앞/뒤를 찾음)으로 감지한다.
            front, rear = self._scan_lane_neighbors(offset)
            res = front if leader else rear
            if res is None or res[0] < 0 or res[0] > self.visibility:
                return None
            return float(res[0]), float(res[1])
        else:
            mode_leader, mode_follower = self._NEIGHBOR_MODES[offset]
            pairs = self._api.vehicle.getNeighbors(
                self.ego_id, mode_leader if leader else mode_follower)
            if not pairs:
                return None
            vid, gap = min(pairs, key=lambda p: p[1])   # 가장 가까운 차량

        if not vid or gap < 0 or gap > self.visibility:
            return None
        return float(gap), self._api.vehicle.getSpeed(vid)

    def _lane_follows_route(self, lane_id) -> bool:
        """이 차선이 ego 경로의 '다음 도로'로 이어지면 True."""
        route = self._api.vehicle.getRoute(self.ego_id)
        idx = self._api.vehicle.getRouteIndex(self.ego_id)
        # 교차로 내부에서는 route index가 직전 일반 edge를 가리킨다.
        # 기준 차선이 그다음 edge(_reference_lane)라면 한 칸 앞에서 본다.
        if 0 <= idx and idx + 1 < len(route) and route[idx + 1] == self._api.lane.getEdgeID(lane_id):
            idx += 1
        if idx < 0 or idx + 1 >= len(route):
            return True                       # 마지막 도로: 어느 차선이든 도착
        next_edge = route[idx + 1]
        for link in self._api.lane.getLinks(lane_id):
            if self._api.lane.getEdgeID(link[0]) == next_edge:
                return True
        return False

    def _stuck_in_wrong_lane(self) -> bool:
        """경로로 못 가는 차선의 끝(5m 이내)에 멈춰 섰으면 True."""
        try:
            if self._api.vehicle.getRoadID(self.ego_id).startswith(":"):
                return False                  # 교차로 내부 통과 중
            lane_id = self._api.vehicle.getLaneID(self.ego_id)
            remaining = (self._api.lane.getLength(lane_id)
                         - self._api.vehicle.getLanePosition(self.ego_id))
            return (remaining < 5.0
                    and self._api.vehicle.getSpeed(self.ego_id) < 0.5
                    and not self._lane_follows_route(lane_id))
        except self._api.TraCIException:
            return False

    def _reference_lane(self):
        """차선 구조 관측의 기준 (edge, 차선 index, 차선 위치)를 반환한다.

        일반 도로에서는 현재 차선 그대로다. 교차로 내부 차선(':'로 시작)에서는
        그 차선이 이어지는 다음 일반 차선을 기준으로 삼고, 위치는 그 차선 시작점까지
        남은 거리만큼 음수로 둔다. 그래서 내부 차선을 지나는 동안에도 차선 연결성·밀도
        값이 앞뒤 도로와 끊김 없이 이어진다. (예전에는 내부 차선에서 모든 차선을
        '있음·이어짐'으로 고정해, 정책이 이 특이한 입력을 정지 위치 신호로 썼다.)
        연결을 따라갈 수 없으면 None.
        """
        lane = self._api.vehicle.getLaneID(self.ego_id)
        position = self._api.vehicle.getLanePosition(self.ego_id)
        if not lane.startswith(":"):
            return (self._api.lane.getEdgeID(lane),
                    self._api.vehicle.getLaneIndex(self.ego_id), position)
        ahead = self._api.lane.getLength(lane) - position
        for _ in range(8):                    # 내부 차선이 여러 개로 나뉜 경우까지 따라간다
            links = self._api.lane.getLinks(lane)
            if not links:
                return None
            lane = links[0][0]                # 내부 차선의 진출 연결은 하나뿐이다
            if not lane.startswith(":"):
                edge = self._api.lane.getEdgeID(lane)
                return edge, int(lane[len(edge) + 1:]), -ahead
            ahead += self._api.lane.getLength(lane)
        return None

    def _lane_connectivity(self, offset: int) -> float:
        """오프셋 차선의 '끊김 정도'를 반환.

        -1 : 그 차선이 존재하지 않음
         1 : visibility 이내에서 계속 이어짐
        0~1: (차선이 끝나는 지점까지의 거리 / visibility)
             → 값이 작을수록 곧 차선이 사라진다는 경고

        일반화 원리: 차선의 남은 길이와, 차선 끝에서 다음 edge로의
        연결(link) 존재 여부를 SUMO 네트워크에서 직접 조회한다.
        차선 감소/합류 지점은 "링크 없는 차선의 끝"으로 자연히 감지된다.
        """
        try:
            reference = self._reference_lane()
            if reference is None:             # 연결을 못 찾은 교차로 내부: 중립값
                return 1.0
            edge, ego_index, ego_pos = reference
            lane_index = ego_index + offset
            if lane_index < 0 or lane_index >= self._api.edge.getLaneNumber(edge):
                return -1.0                   # 그런 차선이 없음
            lane_id = f"{edge}_{lane_index}"
            remaining = self._api.lane.getLength(lane_id) - ego_pos
            if remaining > self.visibility:
                return 1.0                    # 시야 내에선 계속 이어짐
            if self._lane_follows_route(lane_id):
                return 1.0                    # edge가 끝나도 내 경로의 다음 도로로 연결됨
            return float(np.clip(remaining / self.visibility, 0.0, 1.0))
        except self._api.TraCIException:
            return 1.0                        # 조회 실패 시 중립값

    def _route_successor_lanes(self, lane_id, target_edge):
        """내부 차선을 건너 target_edge에 도달하는 연결 차선들을 반환한다.

        외부 차선의 getLinks는 보통 다음 일반 차선을 바로 반환하지만,
        내부 junction이 여러 edge로 나뉜 경우도 실제 링크를 따라간다.
        목표와 다른 일반 도로로 나가는 연결은 탐색하지 않는다.
        """
        pending, visited, result = [lane_id], set(), set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if current not in self._lane_successors_cache:
                self._lane_successors_cache[current] = tuple(
                    (link[0], self._api.lane.getEdgeID(link[0]))
                    for link in self._api.lane.getLinks(current))
            for next_lane, edge in self._lane_successors_cache[current]:
                if edge == target_edge:
                    result.add(next_lane)
                elif edge.startswith(":"):
                    pending.append(next_lane)
        return result

    def _lookahead_connectivity_observation(self):
        """각 후보 차선에서 다음 두 route edge까지 무변경 연결: -1/0/1.

        거리/신호 상태와 독립적인 차선 선택 안내다. 0이어도 중간 도로에서
        차선을 바꿔 진행할 수 있으므로 잘못된 차선 종료 판정에는 사용하지 않는다.
        route의 끝이 가까우면 남아 있는 edge만 확인하고 마지막 edge에서는 1.
        junction 내부에서는 내부 차선이 이어지는 다음 일반 차선(_reference_lane)을
        기준으로, 그다음 두 route edge까지의 연결을 조회한다.
        """
        reference = self._reference_lane()
        if reference is None:                 # 연결을 못 찾은 교차로 내부: 중립값
            return [1.0] * len(self.connectivity_offsets)
        edge, current_index, _ = reference
        lane_count = self._api.edge.getLaneNumber(edge)
        route = self._api.vehicle.getRoute(self.ego_id)
        route_index = self._api.vehicle.getRouteIndex(self.ego_id)
        # 교차로 내부라면 기준 차선은 다음 route edge에 있다 (_reference_lane).
        if route_index + 1 < len(route) and route[route_index + 1] == edge:
            route_index += 1
        next_edges = route[route_index + 1:route_index + 1 + self.lookahead_edges]
        values = []
        for offset in self.connectivity_offsets:
            index = current_index + offset
            if not 0 <= index < lane_count:
                values.append(-1.0)
                continue
            reachable = {f"{edge}_{index}"}
            for next_edge in next_edges:
                reachable = {target for lane in reachable
                             for target in self._route_successor_lanes(lane, next_edge)}
                if not reachable:
                    break
            values.append(float(bool(reachable)))
        return values

    def _lane_density(self, offset: int) -> float:
        """오프셋 차선의 '전방 차량 밀도'를 0~1로 반환.

        밀도 = (시야 W 이내 전방 차량 수) / (그 구간의 최대 수용 대수)
               최대 수용 대수 = 사용가능거리 / density_veh_length

        인코딩 관례 (막힘 정도로 읽히도록):
            1.0 : 차선이 없거나 사실상 진입 불가  ← "꽉 찬 것"처럼 취급해서
                  에이전트가 자연스럽게 회피하게 만든다
            0.0 : 텅 빈 차선
        일반화: 차선이 시야보다 짧으면 link를 따라 다음 차선까지 이어서 센다.
        곧 끊기는 차선(link 없음)은 남은 거리만큼만 세므로 분모가 줄어
        같은 대수여도 밀도가 높게 나온다 → "끊기는 차선에 차가 몰려있음" 신호.
        """
        try:
            reference = self._reference_lane()
            if reference is None:             # 연결을 못 찾은 교차로 내부: 중립값
                return 0.0
            edge, ego_index, ego_pos = reference
            lane_index = ego_index + offset
            if lane_index < 0 or lane_index >= self._api.edge.getLaneNumber(edge):
                return 1.0                    # 차선 없음 = 꽉 참 취급
            lane_id = f"{edge}_{lane_index}"
            remaining = self._api.lane.getLength(lane_id) - ego_pos
            links = self._api.lane.getLinks(lane_id)

            # 이 차선에서 앞으로 실제로 쓸 수 있는 거리
            usable = self.visibility if links else min(self.visibility, remaining)
            if usable <= 0:
                return 1.0

            count = 0
            # (a) 현재 차선 구간의 전방 차량
            for vid in self._api.lane.getLastStepVehicleIDs(lane_id):
                if vid == self.ego_id:
                    continue
                d = self._api.vehicle.getLanePosition(vid) - ego_pos
                if 0.0 < d <= min(self.visibility, remaining):
                    count += 1
            # (b) 시야가 edge 끝을 넘고 다음 차선으로 이어지면 이어서 센다
            if links and remaining < self.visibility:
                next_lane_id = links[0][0]    # 첫 번째 연결 차선
                for vid in self._api.lane.getLastStepVehicleIDs(next_lane_id):
                    if self._api.vehicle.getLanePosition(vid) <= \
                            self.visibility - remaining:
                        count += 1

            max_count = usable / self.density_veh_len
            return float(np.clip(count / max_count, 0.0, 1.0))
        except self._api.TraCIException:
            return 0.0

    def get_detected_vehicles(self):
        """현재 관측 시야에 들어온(=감지된) 배경차들의 좌표 목록을 반환.

        데이터셋 수집용 부가 정보 — 관측 벡터에는 들어가지 않는다.
        "감지"의 정의는 관측과 동일: 관측 대상 차선(obs_offsets 범위)에서
        ego 기준 종방향 ±visibility(W) 이내에 있는 차량.

        Returns: list[dict]  각 원소 =
            {"id": 차량 id, "x": 절대 x, "y": 절대 y,
             "rel_pos": ego 기준 종방향 상대거리(m, 앞 +), "speed": m/s,
             "lane_offset": ego 차선 기준 오프셋(+왼쪽)}
        ego가 없으면 빈 리스트.
        """
        out = []
        if not self._ego_is_present():
            return out
        try:
            edge = self._api.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return out
            ego_lane = self._api.vehicle.getLaneIndex(self.ego_id)
            ego_pos = self._api.vehicle.getLanePosition(self.ego_id)
            n_lanes = self._api.edge.getLaneNumber(edge)
            for offset in self.obs_offsets:
                li = ego_lane + offset
                if li < 0 or li >= n_lanes:
                    continue
                for vid in self._api.lane.getLastStepVehicleIDs(f"{edge}_{li}"):
                    if vid == self.ego_id:
                        continue
                    rel = self._api.vehicle.getLanePosition(vid) - ego_pos
                    if abs(rel) > self.visibility:
                        continue
                    x, y = self._api.vehicle.getPosition(vid)
                    out.append({"id": vid, "x": float(x), "y": float(y),
                                "rel_pos": float(rel),
                                "speed": float(self._api.vehicle.getSpeed(vid)),
                                "lane_offset": int(offset)})
        except self._api.TraCIException:
            pass
        return out

    def _upcoming_signals(self):
        """경로의 신호별 실제 제어 정지선을 조회한다. 비신호 구간의 표시는 사용하지 않는다."""
        if not self.signal_enabled or not self._ego_is_present():
            return []
        signals = []
        for tid, index, distance, state in self._api.vehicle.getNextTLS(self.ego_id):
            key = (tid, index)
            # 제어 연결에 대응하는 정지선만 선택. 차선 끝의 일반 정지선/장식은 제외.
            if key not in self._signal_stop_lines:
                continue
            distance -= self._signal_stop_offsets[key]
            if distance >= 0:
                signals.append((tid, index, distance, state))
        return sorted(signals, key=lambda signal: signal[2])

    def _next_signal(self):
        signals = self._upcoming_signals()
        return signals[0] if signals else None

    def get_stop_clearance(self):
        """Front-bumper clearance at a stop, measured along the road."""
        if not self._ego_is_present() or self._api.vehicle.getSpeed(self.ego_id) > 0.5:
            return None
        signals = self._upcoming_signals()
        candidates = [(signals[0][2], "stop_line")] if signals else []
        leader = self._api.vehicle.getLeader(self.ego_id, 1e6)
        if leader and leader[0] and self._api.vehicle.getSpeed(leader[0]) <= 0.5:
            # SUMO's leader gap excludes the ego's minGap.
            distance = leader[1] + self._api.vehicle.getMinGap(self.ego_id)
            candidates.append((distance, "queued_vehicle"))
        distance, reference = min(candidates) if candidates else (None, None)
        return {"stop_clearance_m": distance, "stop_reference": reference}

    def _observed_signals(self, signals):
        return [signal for signal in signals[:self.signal_count]
                if signal[2] <= self.signal_visibility]

    def _nearest_red_signal(self, signals):
        return next((signal for signal in self._observed_signals(signals)
                     if signal[3].lower() == "r"), None)

    def _signal_observation(self, signal):
        # 정밀 거리: 거리/visibility(100m)로는 1m가 0.01이라 정지선 앞 미세 조정이 어렵다.
        # signal_fine_distance_scale(m) 이내를 0~1로 다시 펼친 값을 덧붙인다.
        fine = self.signal_fine_scale > 0
        if signal is None or signal[2] > self.signal_visibility:
            return [1.0, 0.0, 0.0, 0.0, 1.0] + ([1.0] if fine else [])
        color = signal[3].lower()
        index = {"g": 1, "y": 2, "r": 3}.get(color, 0)
        one_hot = [float(i == index) for i in range(4)]
        obs = one_hot + [float(np.clip(signal[2] / self.signal_visibility, 0, 1))]
        if fine:
            obs.append(float(np.clip(signal[2] / self.signal_fine_scale, 0, 1)))
        return obs

    def _red_stop_penalty(self, signal):
        """관측된 적신호까지의 이동 여유에 비례한 감점. 속도 조건은 두지 않는다."""
        coef = float(self.rw.get("red_stop_penalty_coef", 0.0))
        if (coef <= 0 or signal is None or signal[3].lower() != "r"
                or signal[2] > self.signal_visibility):
            return 0.0
        stop_room = max(signal[2] - float(self.rw.get("red_stop_margin", 3.0)), 0.0)
        leader = self._api.vehicle.getLeader(self.ego_id, self.signal_visibility)
        front_room = float("inf")
        if leader is not None and leader[0]:
            front_room = max(leader[1] - float(self.rw.get("red_stop_safe_gap", 3.0)), 0.0)
        scale = float(self.rw.get("red_stop_distance_scale", self.signal_visibility))
        if scale <= 0:
            raise ValueError("red_stop_distance_scale must be positive")
        return coef * float(np.clip(min(front_room, stop_room) / scale, 0.0, 1.0))

    def _red_park_penalty(self, signal, speed):
        """적신호에서 '멈춰 있는 동안' 목표 정차 위치까지 남은 거리에 비례한 감점. 0이면 비활성.

        _red_stop_penalty 는 달리는 중에도 붙어서 크게 키우면 접근 자체가 손해가 된다
        (coef 0.2 에서 학습 붕괴). 이 항은 느릴 때(red_park_max_speed 이하)만 붙으므로
        접근은 공짜이고, 목표보다 멀리 '주차'한 상태만 비용이 된다. 목표 위치는 다른 항과
        같은 _red_stop_surplus (정지선 - red_stop_margin, 또는 앞차 - red_stop_safe_gap).
        """
        coef = float(self.rw.get("red_park_penalty_coef", 0.0))
        if (coef <= 0 or signal is None or signal[3].lower() != "r"
                or signal[2] > self.signal_visibility
                or speed > float(self.rw.get("red_park_max_speed", 2.0))):
            return 0.0
        surplus = self._red_stop_surplus(signal)
        if surplus <= 0:
            return 0.0
        return coef * min(surplus / float(self.rw.get("red_park_scale", 10.0)), 1.0)

    def _red_stop_clearance_reward(self, signal, speed):
        """Reward a safe, small gap to the next red stop line or a nearer leader."""
        cap = float(self.rw.get("red_stop_clearance_reward", 0.0))
        if (cap <= 0 or signal is None or signal[3].lower() != "r"
                or signal[2] > self.signal_visibility
                or speed > float(self.rw.get("red_stop_clearance_max_speed", 2.0))):
            return 0.0

        surplus = self._red_stop_surplus(signal)
        if surplus < 0:
            return 0.0
        scale = float(self.rw.get("red_stop_clearance_reward_scale", 5.0))
        return cap / (1.0 + surplus / scale)

    def _red_stop_surplus(self, signal):
        surplus = signal[2] - float(self.rw.get("red_stop_margin", 3.0))
        gap = self._leader_gap_before_signal(signal)
        if gap is not None:
            surplus = min(surplus, gap - float(self.rw.get("red_stop_safe_gap", 3.0)))
        return surplus

    def _leader_gap_before_signal(self, signal):
        leader = self._api.vehicle.getLeader(self.ego_id, signal[2])
        if leader and leader[0]:
            gap = leader[1] + self._api.vehicle.getMinGap(self.ego_id)
            if gap <= signal[2]:
                return gap
        return None

    def _red_stop_progress_reward(self, signal_before, signal_after, speed, traveled):
        """Reward ego's forward movement towards a red signal it can still stop for."""
        per_meter = float(self.rw.get("red_stop_progress_per_meter", 0.0))
        if (per_meter <= 0 or traveled <= 0 or signal_before is None or signal_after is None
                or signal_before[:2] != signal_after[:2]
                or signal_before[3].lower() != "r" or signal_after[3].lower() != "r"
                or max(signal_before[2], signal_after[2]) > self.signal_visibility):
            return 0.0
        # Use the current target for the stopping check, and the ego odometer
        # for progress so a leader moving alone does not generate reward.
        decel = float(self.rw.get("red_stop_progress_decel", 2.0))
        if self._red_stop_surplus(signal_after) < speed ** 2 / (2.0 * decel):
            return 0.0
        return min(per_meter * traveled, float(self.rw.get("red_stop_progress_cap", 0.02)))

    def _red_speed_tracking_reward(self, signal, speed):
        """Reward matching a braking speed profile towards the current red stop target."""
        cap = float(self.rw.get("red_speed_tracking_cap", 0.0))
        if (cap <= 0 or signal is None or signal[3].lower() != "r"
                or signal[2] > self.signal_visibility):
            return 0.0

        surplus = self._red_stop_surplus(signal)
        if surplus < 0:
            return 0.0
        safe_decel = float(self.rw.get("red_speed_tracking_safe_decel", 2.5))
        if speed ** 2 > 2.0 * safe_decel * surplus:
            return 0.0
        target_decel = float(self.rw.get("red_speed_tracking_target_decel", 1.5))
        # Limit speed to the remaining metres per second near the target.
        target_speed = min(self.vmax, sqrt(2.0 * target_decel * surplus), surplus)
        tolerance = max(0.5, 0.5 * target_speed)
        if speed < target_speed and self.rw.get("red_speed_tracking_linear_underspeed", False):
            # Credit slow starts, while keeping the half-metre-per-second
            # tolerance near a full stop to avoid a discontinuity.
            tolerance = max(0.5, target_speed)
        score = max(0.0, 1.0 - abs(speed - target_speed) / tolerance)
        # Give a small reward for holding the target stop, below the moving cap.
        weight = 0.25 + 0.75 * min(target_speed / 2.0, 1.0)
        return cap * weight * score

    def _red_stopped_brake_penalty(self, signal_before, signal_after, speed_before, accel):
        """Discourage braking after stopping, when there is room to approach the red."""
        coef = float(self.rw.get("red_stopped_brake_coef", 0.0))
        if (coef <= 0 or accel >= 0 or speed_before is None
                or speed_before > float(self.rw.get("red_stopped_brake_speed", 0.1))
                or signal_before is None or signal_after is None
                or signal_before[:2] != signal_after[:2]
                or signal_before[3].lower() != "r" or signal_after[3].lower() != "r"
                or max(signal_before[2], signal_after[2]) > self.signal_visibility):
            return 0.0
        if self._red_stop_surplus(signal_after) < float(self.rw.get("red_stopped_brake_min_room", 2.0)):
            return 0.0
        return coef * accel ** 2

    def _red_safe_zone_rewards(self, signal, speed):
        """Pay once for establishing a safe stop, then a capped holding reward."""
        first_reward = float(self.rw.get("red_safe_zone_first_reward", 0.0))
        if first_reward <= 0:
            return 0.0, 0.0
        if (signal is None or signal[3].lower() != "r" or signal[2] > self.signal_visibility
                or speed > float(self.rw.get("red_safe_zone_max_speed", 0.1))):
            self._red_safe_zone_since = None
            return 0.0, 0.0

        gap = self._leader_gap_before_signal(signal)
        clearance = signal[2] if gap is None else gap
        if not (float(self.rw.get("red_safe_zone_min_gap", 3.0)) <= clearance
                <= float(self.rw.get("red_safe_zone_max_gap", 5.0))):
            self._red_safe_zone_since = None
            return 0.0, 0.0

        # Share the payout across links of one signal to prevent repeated
        # first rewards after switching lanes or waiting another signal cycle.
        signal_id = signal[0]
        paid = self._red_safe_zone_hold_paid
        if signal_id in paid:
            cap = float(self.rw.get("red_safe_zone_hold_cap", 0.2))
            hold = min(float(self.rw.get("red_safe_zone_hold_reward", 0.005)),
                       max(0.0, cap - paid[signal_id]))
            paid[signal_id] += hold
            return 0.0, hold

        link = signal[:2]
        if self._red_safe_zone_since is None or self._red_safe_zone_since[0] != link:
            self._red_safe_zone_since = (link, self.step_count)
        elapsed = (self.step_count - self._red_safe_zone_since[1]) * self.step_length
        if elapsed < float(self.rw.get("red_safe_zone_dwell_seconds", 1.0)):
            return 0.0, 0.0
        paid[signal_id] = 0.0
        return first_reward, 0.0

    def _green_stop_penalty(self, signals, speed):
        """Penalty for staying stopped at an observed green with room to move."""
        coef = float(self.rw.get("green_stop_penalty", 0.0))
        if coef <= 0:
            return 0.0
        observed = self._observed_signals(signals)
        if (not observed or observed[0][3].lower() != "g"
                or any(signal[3].lower() in ("r", "y") for signal in observed)
                or speed > float(self.rw.get("green_stop_speed", 0.1))):
            self._green_stop_since = None
            return 0.0
        clear_gap = float(self.rw.get("green_stop_clear_gap", 10.0))
        leader = self._api.vehicle.getLeader(self.ego_id, clear_gap)
        if (leader and leader[0]
                and leader[1] + self._api.vehicle.getMinGap(self.ego_id) < clear_gap):
            self._green_stop_since = None
            return 0.0

        signal = observed[0][:2]
        if self._green_stop_since is None or self._green_stop_since[0] != signal:
            self._green_stop_since = (signal, self.step_count)
        elapsed = (self.step_count - self._green_stop_since[1]) * self.step_length
        return coef if elapsed >= float(self.rw.get("green_stop_grace_seconds", 1.0)) else 0.0

    def _late_brake_penalty(self, signal, speed):
        """적신호 앞에서 정지선까지 서는 데 필요한 감속이 기준을 넘으면 감점. 0이면 비활성.

        정지 위치 감점만 있으면 '최대한 늦게, 가깝게' 서는 쪽으로 학습되어
        정지선을 1m 안팎으로 넘는 위반이 생긴다 (A2 진입로). 미리 부드럽게 서게 한다.
        """
        coef = float(self.rw.get("late_brake_penalty_coef", 0.0))
        if coef <= 0 or signal is None or signal[3].lower() != "r":
            return 0.0
        limit = float(self.rw.get("late_brake_decel", 2.5))     # m/s², 편안한 감속 기준
        required = speed * speed / (2.0 * max(signal[2], 0.1))  # v² / 2d
        return coef * float(np.clip((required - limit) / limit, 0.0, 1.0))

    def _destination_observation(self):
        """경로의 마지막 edge에서 목적지를 읽어 설정 순서의 one-hot을 반환한다."""
        final_edge = self._destination_edge()
        arm, separator, suffix = final_edge.partition("_out")
        if not separator or (suffix and not suffix.isdigit()) or arm not in self.destination_arms:
            raise ValueError(f"Unknown destination edge for destination observation: {final_edge}")
        return [float(arm == destination) for destination in self.destination_arms]

    def _destination_edge(self):
        return self._api.vehicle.getRoute(self.ego_id)[-1]

    def _get_obs(self):
        # ego가 이미 사라졌다면(충돌/도착) 0벡터 반환 — 종료 직전 마지막 관측용
        if not self._ego_is_present():
            return np.zeros(self.obs_dim, dtype=np.float32)

        v = self._api.vehicle.getSpeed(self.ego_id)
        obs = [v / self.vmax]                      # [0] 내 절대속도

        # ----- 5개 차선(좌2→좌1→현재→우1→우2)의 선행/후행 상대거리·상대속도 -----
        for offset in self.obs_offsets:
            lead = self._neighbor(offset, leader=True)
            if lead is None:
                obs += [1.0, 1.0]                  # 없음: 멀고 + 위협 없음
            else:
                gap, lv = lead
                obs += [gap / self.visibility,
                        float(np.clip((lv - v) / self.vmax, -1.0, 1.0))]

            foll = self._neighbor(offset, leader=False)
            if foll is None:
                obs += [-1.0, -1.0]                # 없음: 멀고 + 위협 없음
            else:
                gap, fv = foll
                obs += [-gap / self.visibility,    # 후행은 음수 부호
                        float(np.clip((fv - v) / self.vmax, -1.0, 1.0))]

        # ----- 경로 기준 차선 연결성: 실험 8은 좌4→현재→우4 -----
        for offset in self.connectivity_offsets:
            obs.append(self._lane_connectivity(offset))

        # ----- 5개 차선의 전방 차량 밀도 (같은 순서) -----
        for offset in self.obs_offsets:
            obs.append(self._lane_density(offset))

        if self.signal_enabled:
            signals = self._upcoming_signals()
            for i in range(self.signal_count):
                obs.extend(self._signal_observation(signals[i] if i < len(signals) else None))
        if self.lookahead_connectivity_enabled:
            obs.extend(self._lookahead_connectivity_observation())
        if self.destination_enabled:
            obs.extend(self._destination_observation())
        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    # ==================================================================
    # 1 스텝 진행: 행동 적용 → 시뮬레이션 1스텝 → 보상/종료 판정
    # ==================================================================
    def _quantize_lane_change(self, raw: float) -> int:
        """연속 policy 출력을 실제 차선 명령 {-1,0,+1}로 변환한다.

        +1 = 왼쪽, 0 = 유지, -1 = 오른쪽. SUMO 차선 index가 왼쪽으로
        갈수록 증가하는 규약과 동일해서 해석이 직관적이다.
        """
        raw = float(np.clip(raw, -1.0, 1.0))
        if raw >= self.lane_change_threshold:
            return 1
        if raw <= -self.lane_change_threshold:
            return -1
        return 0

    def _scan_lane_neighbors(self, direction: int):
        """목표 차선을 직접 스캔해 (가장 가까운 앞차 gap·속도, 뒷차 gap·속도) 반환.

        getNeighbors의 백업. getNeighbors는 SUMO 차선변경모델의 이웃 캐시를
        읽는데, "방금 스폰된 차량"은 다음 스텝까지 이 캐시에 없어서
        도로 입구 근처에서 사각지대가 생긴다. 차선의 차량 목록을 직접
        훑으면 이미 삽입된 차량은 전부 잡힌다.
        """
        try:
            edge = self._api.vehicle.getRoadID(self.ego_id)
            li = self._api.vehicle.getLaneIndex(self.ego_id) + direction
            # 존재하지 않는 차선이면 조회하지 않는다.
            # (없는 lane id를 traci.lane.*로 조회하면 SUMO가 매번
            #  에러 로그를 찍으므로, 범위 검사를 먼저 해서 조용히 건너뛴다)
            if edge.startswith(":") or li < 0 \
                    or li >= self._api.edge.getLaneNumber(edge):
                return None, None
            lane_id = f"{edge}_{li}"
            my_pos = self._api.vehicle.getLanePosition(self.ego_id)
            front, rear = None, None
            for vid in self._api.lane.getLastStepVehicleIDs(lane_id):
                d = self._api.vehicle.getLanePosition(vid) - my_pos
                if d >= 0:
                    g = d - self._api.vehicle.getLength(vid)
                    if front is None or g < front[0]:
                        front = (g, self._api.vehicle.getSpeed(vid))
                else:
                    g = -d - self._api.vehicle.getLength(self.ego_id)
                    if rear is None or g < rear[0]:
                        rear = (g, self._api.vehicle.getSpeed(vid))
            return front, rear
        except self._api.TraCIException:
            return None, None

    def _lane_change_is_safe(self, direction: int) -> bool:
        """목표 차선으로 지금 이동해도 측면충돌이 나지 않는지 검사한다.

        3중 안전 규칙:
          (0) 입구 보호구간: ego가 출발 도로 시작 spawn_protect(m) 이내면 변경 금지.
              flow가 입구(0m)에서 차를 새로 만드는데, 생성 "직전" 차량은
              어떤 API로도 미리 볼 수 없어 원천적으로 사각지대이기 때문.
          (1) 앞: gap > min_front_gap + front_tau × 내속도
              + 접근 여유: gap > (내속도 − 앞차속도)⁺ × 1s   (내가 더 빠를 때)
          (2) 뒤: gap > min_rear_gap + rear_tau × 뒤차속도
              + 접근 여유: gap > (뒤차속도 − 내속도)⁺ × 1s   (뒤가 더 빠를 때)
        이웃은 getNeighbors 결과와 차선 직접 스캔 결과 중 "더 가까운 쪽"을
        쓴다 (스폰 직후 차량 사각지대 보완).
        """
        my_pos = self._api.vehicle.getLanePosition(self.ego_id)
        # 입구 보호구간은 ego 의 "첫 출발 도로"에서만 적용한다.
        #   (교차로 도로는 갈림점마다 구간이 잘게 나뉘는데, 그 구간들 시작마다 금지하면
        #    교차로 직전에서 차선을 아예 못 바꾼다. 새 차가 생겨나는 곳은 출발 도로 입구뿐.)
        if (my_pos < self.lc_spawn_protect
                and self._api.vehicle.getRouteIndex(self.ego_id) == 0):
            return False

        v = self._api.vehicle.getSpeed(self.ego_id)
        scan_front, scan_rear = self._scan_lane_neighbors(direction)

        def closer(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return a if a[0] <= b[0] else b

        lead = closer(self._neighbor(direction, leader=True), scan_front)
        if lead is not None:
            gap, lv = lead
            if gap <= self.lc_front_min + self.lc_front_tau * v:
                return False
            if gap <= max(v - lv, 0.0) * self.lc_horizon:  # 시간여유 내 접촉 위험
                return False

        foll = closer(self._neighbor(direction, leader=False), scan_rear)
        if foll is not None:
            gap, fv = foll
            if gap <= self.lc_rear_min + self.lc_rear_tau * fv:
                return False
            if gap <= max(fv - v, 0.0) * self.lc_horizon:  # 시간여유 내 접촉 위험
                return False
        return True

    def _apply_lane_change(self, lane_change: int) -> bool:
        """가능하고 "안전한" 경우에만 차선변경을 적용하고 성공 여부를 반환.

        안전 게이트(_lane_change_is_safe)를 통과하지 못한 명령은 실행하지
        않고 False를 반환한다 → step()에서 "불가능한 명령"과 동일하게
        invalid_action_penalty가 부과된다. 즉 에이전트 입장에서
        "위험한 변경 = 실행도 안 되고 감점"이라, 측면충돌로 즉사하며
        배우는 대신 안전한 타이밍을 고르는 쪽으로 학습이 유도된다.
        """
        if not self._ego_is_present():
            return False
        try:
            edge = self._api.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return False
            if lane_change != 0 and edge in self.no_lane_change_edges:
                return False
            current = self._api.vehicle.getLaneIndex(self.ego_id)
            n_lanes = self._api.edge.getLaneNumber(edge)
            target = current + lane_change
            if target < 0 or target >= n_lanes:
                return False
            # ★ 측면충돌 방지: 실제 이동 명령(0이 아닌 변경)에만 안전 검사
            if (lane_change != 0 and self.lc_safety_enabled
                    and not self._lane_change_is_safe(lane_change)):
                return False
            self._api.vehicle.changeLane(
                self.ego_id, target, self.lane_change_duration)
            return True
        except self._api.TraCIException:
            return False

    def get_privileged_state(self) -> dict:
        """POMDP 데이터셋용 simulator-side state.

        agent가 실제로 받는 입력은 _get_obs()의 부분관측이고, 이 메서드는
        분석/지도신호용 privileged state만 수집한다. 정책 입력으로 쓰지 않는다.
        """
        if not self._ego_is_present():
            return {"ego_alive": False}
        try:
            x, y = self._api.vehicle.getPosition(self.ego_id)
            vehicles = []
            for vid in self._api.vehicle.getIDList():
                try:
                    vx, vy = self._api.vehicle.getPosition(vid)
                    vehicles.append({
                        "id": vid,
                        "x": float(vx),
                        "y": float(vy),
                        "speed": float(self._api.vehicle.getSpeed(vid)),
                        "lane": int(self._api.vehicle.getLaneIndex(vid)),
                        "lane_id": self._api.vehicle.getLaneID(vid),
                        "road_id": self._api.vehicle.getRoadID(vid),
                        "lane_pos": float(self._api.vehicle.getLanePosition(vid)),
                    })
                except self._api.TraCIException:
                    continue
            return {
                "ego_alive": True,
                "ego_x": float(x),
                "ego_y": float(y),
                "ego_speed": float(self._api.vehicle.getSpeed(self.ego_id)),
                "ego_lane": int(self._api.vehicle.getLaneIndex(self.ego_id)),
                "ego_lane_id": self._api.vehicle.getLaneID(self.ego_id),
                "ego_road_id": self._api.vehicle.getRoadID(self.ego_id),
                "ego_lane_pos": float(self._api.vehicle.getLanePosition(self.ego_id)),
                "vehicles": vehicles,
            }
        except self._api.TraCIException:
            return {"ego_alive": False}

    def step(self, action):
        state = self._prepare_step(action)
        self._api.simulationStep()
        return self._finish_step(
            state,
            collided=self.ego_id in self._api.simulation.getCollidingVehiclesIDList(),
            arrived=self.ego_id in self._api.simulation.getArrivedIDList(),
        )

    def _prepare_step(self, action):
        # action = [가감속 raw, 차선변경 raw]
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size < self.act_dim:
            raise ValueError(
                f"action은 {self.act_dim}차원이어야 합니다. 받은 shape={np.asarray(action).shape}")

        accel_raw = float(np.clip(action_arr[self.accel_index], -1.0, 1.0))
        lane_raw = float(np.clip(action_arr[self.lane_change_index], -1.0, 1.0))
        lane_change = self._quantize_lane_change(lane_raw)
        lane_change_applied = False
        lane_before = None
        speed_before = None

        # ----- (1) 종방향 + 횡방향 행동 적용 -----
        if self._ego_is_present():
            speed_before = self._api.vehicle.getSpeed(self.ego_id)
            lane_before = int(self._api.vehicle.getLaneIndex(self.ego_id))

            accel_cmd = (accel_raw * self.max_accel
                         if accel_raw >= 0.0 else accel_raw * self.max_decel)
            dv = accel_cmd * self.step_length
            self._api.vehicle.setSpeed(
                self.ego_id, float(np.clip(speed_before + dv, 0.0, self.vmax)))

            lane_change_applied = self._apply_lane_change(lane_change)

        # 신호 상태는 행동이 적용되는 스텝 시작 시점을 기준으로 판정한다.
        signals_before = self._upcoming_signals()
        route_connected_before = (bool(signals_before)
                                  and self._lane_follows_route(self._api.vehicle.getLaneID(self.ego_id)))
        distance_before = (self._api.vehicle.getDistance(self.ego_id)
                           if signals_before else None)
        return (accel_raw, lane_raw, lane_change, lane_change_applied, lane_before, speed_before,
                signals_before, route_connected_before, distance_before)

    def _finish_step(self, state, *, collided, arrived):
        (accel_raw, lane_raw, lane_change, lane_change_applied, lane_before, speed_before,
         signals_before, route_connected_before, distance_before) = state
        self.step_count += 1
        alive = self._ego_is_present()
        lane_after = int(self._api.vehicle.getLaneIndex(self.ego_id)) if alive else None

        red_light_violation = False
        green_passed = set()
        traveled = 0.0
        if signals_before and alive:
            traveled = self._api.vehicle.getDistance(self.ego_id) - distance_before
            for signal in signals_before:
                crossed = traveled > signal[2] + 1e-6
                red_light_violation |= crossed and signal[3].lower() == "r"
                if (crossed and signal[3].lower() == "g"
                        and route_connected_before
                        and signal[0] not in self._rewarded_green_signals
                        and self._lane_follows_route(self._api.vehicle.getLaneID(self.ego_id))):
                    green_passed.add(signal[0])
        signals_after = self._upcoming_signals()
        signal_after = signals_after[0] if signals_after else None
        red_signal = self._nearest_red_signal(signals_after)
        red_visible = red_signal is not None
        obs = self._get_obs()
        terminated, truncated = False, False
        ego_speed, ego_gap = None, None
        red_stop_penalty = 0.0
        red_stop_clearance_reward = 0.0
        red_stop_progress_reward = 0.0
        red_speed_tracking_reward = 0.0
        red_stopped_brake_penalty = 0.0
        red_safe_zone_first_reward = red_safe_zone_hold_reward = 0.0
        green_stop_penalty = 0.0
        green_pass_reward = 0.0

        # ----- (4) 기존 보상 유지 -----
        if red_light_violation:
            reward = -float(self.rw.get("red_light_penalty", 10.0))
            terminated = True
        elif collided:
            reward = -float(self.rw["collision_penalty"])
            terminated = True
        elif arrived:
            reward = float(self.rw["arrival_bonus"])
            terminated = True
        elif not alive:
            # 충돌도 도착도 아닌데 ego 가 사라짐.
            #   교차로(nohyeong)에서는 "경로로 이어지지 않는 차선으로 교차로에 들어가서
            #   SUMO 가 ego 를 지운 경우"라서, 잘못된 차선 감점을 준다.
            reward = (-float(self.rw.get("wrong_lane_penalty", 2.0))
                      if self.road_type == "nohyeong" else 0.0)
            terminated = True
        else:
            ego_speed = self._api.vehicle.getSpeed(self.ego_id)
            same_lane_leader = self._neighbor(0, leader=True)
            if same_lane_leader is not None:
                ego_gap = same_lane_leader[0]
            reward = float(self.rw["speed_weight"]) * ego_speed / self.vmax
            red_stop_penalty = self._red_stop_penalty(red_signal)
            reward -= red_stop_penalty
            red_stop_clearance_reward = self._red_stop_clearance_reward(signal_after, ego_speed)
            reward += red_stop_clearance_reward
            red_stop_progress_reward = self._red_stop_progress_reward(
                signals_before[0] if signals_before else None, signal_after, ego_speed, traveled)
            reward += red_stop_progress_reward
            red_speed_tracking_reward = self._red_speed_tracking_reward(signal_after, ego_speed)
            reward += red_speed_tracking_reward
            red_stopped_brake_penalty = self._red_stopped_brake_penalty(
                signals_before[0] if signals_before else None, signal_after, speed_before, accel_raw)
            reward -= red_stopped_brake_penalty
            reward -= self._red_park_penalty(signal_after, ego_speed)
            red_safe_zone_first_reward, red_safe_zone_hold_reward = self._red_safe_zone_rewards(
                signal_after, ego_speed)
            reward += red_safe_zone_first_reward + red_safe_zone_hold_reward
            green_stop_penalty = self._green_stop_penalty(signals_after, ego_speed)
            reward -= green_stop_penalty
            reward -= self._late_brake_penalty(red_signal, ego_speed)
            exempt_close = (red_visible and ego_speed <=
                            float(self.rw.get("signal_wait_speed", 0.5)))
            if (not exempt_close and ego_gap is not None and
                    ego_gap < float(self.rw["close_gap_threshold"])):
                reward -= float(self.rw["close_gap_penalty"])

            # ---- 막힘 페널티: 느린 앞차 뒤에 갇혀 있는 동안 매 스텝 감점 ----
            # "차선을 유지한 채 막혀 있기"와 "변경으로 탈출하기" 사이에
            # 즉각적인 보상 차등을 만들어 추월 학습 신호를 조밀하게 한다.
            # (자세한 설계 이유는 env/mdp_config.py의 REWARD 주석 참고)
            bp = float(self.rw.get("blocked_penalty", 0.0))
            if (not red_visible and bp > 0.0 and ego_gap is not None
                    and ego_gap < float(self.rw.get("blocked_gap", 20.0))
                    and ego_speed < float(self.rw.get("blocked_speed_frac", 0.7))
                        * self.vmax):
                reward -= bp

            # ---- 경로 밖 차선 페널티: 차선을 '확실히' 바꾸도록 만드는 신호 ----
            # 경로로 이어지지 않는 차선에 있고 그 끝이 시야 안에 들어온 동안
            # (= 현재 차선 연결성 관측이 1 미만인 동안) 매 스텝 감점한다.
            # 이게 없으면 차선 끝 5m 밖에 서 있는 것은 감점이 전혀 없어서,
            # 정책이 "서서 우연한 차선변경을 기다리기"를 배운다 (학습 중에는
            # 표본 추출로 빠져나오지만 결정적 평가에서는 영원히 멈춰 있음).
            orp = float(self.rw.get("off_route_lane_penalty", 0.0))
            if orp > 0.0 and self._lane_connectivity(0) < 1.0:
                reward -= orp

            # ---- 차선변경 보상 항 ----
            # lane_change        : 이번 스텝에 policy가 낸 양자화 명령 (-1/0/+1)
            # lane_change_applied: 그 명령이 실제로 실행됐는지
            # 주의: _apply_lane_change는 유지 명령(0)에도 현재 차선으로
            # changeLane을 호출해 True를 반환한다(차선 고정 용도).
            # 따라서 "실제 변경"은 lane_change != 0 과 함께 판정해야 한다.
            if lane_change != 0 and lane_change_applied:
                # 실행된 변경마다 소액 감점 → 이득 없는 지그재그 억제.
                # 변경 덕에 속도 보상이 그 이상 늘면 순이득이므로,
                # "필요할 때만 바꾸는" 정책으로 수렴하게 만드는 항이다.
                reward -= float(self.rw.get("lane_change_penalty", 0.0))
            elif lane_change != 0:
                # 명령은 냈는데 실행 불가(차선 없음/교차로)였던 경우 감점
                # → 의미 없는 행동 출력을 줄이는 정칙화 (rl_action_penalty 대응)
                reward -= float(self.rw.get("invalid_action_penalty", 0.0))

        # ---- 잘못된 차선에 갇힘: 감점 후 Episode 종료 ----
        #   경로로 이어지지 않는 차선 끝에서 멈추면 더 갈 방법이 없으므로 끝낸다.
        wrong_lane = (not alive and not collided and not arrived
                      and self.road_type == "nohyeong")
        if alive and not terminated and self._stuck_in_wrong_lane():
            wrong_lane = True
            reward -= float(self.rw.get("wrong_lane_penalty", 2.0))
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        if green_passed and not (red_light_violation or collided or wrong_lane):
            green_pass_reward = len(green_passed) * float(self.rw.get("green_pass_bonus", 0.0))
            reward += green_pass_reward
            self._rewarded_green_signals.update(green_passed)

        info = {
            "route": self.ego_route_id,
            "collided": collided,
            "arrived": arrived,
            "wrong_lane": wrong_lane,
            "red_light_violation": red_light_violation,
            "green_pass_reward": green_pass_reward,
            "green_stop_penalty": green_stop_penalty,
            "green_pass_count": len(green_passed) if green_pass_reward > 0 else 0,
            "observed_signals": [
                {"tls_id": sig[0], "link_index": sig[1], "state": sig[3],
                 "distance": sig[2], **self._signal_stop_lines[(sig[0], sig[1])]}
                for sig in self._observed_signals(signals_after)],
            "red_signal_distance": red_signal[2] if red_signal is not None else None,
            "red_stop_penalty": red_stop_penalty,
            "red_stop_clearance_reward": red_stop_clearance_reward,
            "red_stop_progress_reward": red_stop_progress_reward,
            "red_speed_tracking_reward": red_speed_tracking_reward,
            "red_stopped_brake_penalty": red_stopped_brake_penalty,
            "red_safe_zone_first_reward": red_safe_zone_first_reward,
            "red_safe_zone_hold_reward": red_safe_zone_hold_reward,
            "signal_red_visible": red_visible,
            "signal_distance": signal_after[2] if signal_after is not None else None,
            "speed": ego_speed,
            "gap": ego_gap,
            "accel_raw": accel_raw,
            "lane_change_raw": lane_raw,
            "lane_change": lane_change,
            "lane_change_applied": lane_change_applied,
            "lane_before": lane_before,
            "lane_after": lane_after,
        }
        return obs, float(reward), terminated, truncated, info

    # ==================================================================
    def close(self):
        """환경 정리. 학습이 끝나면 반드시 호출해서 SUMO 프로세스를 종료할 것."""
        if self._started:
            self._api.close()
            self._started = False
