"""configs/road_config.py 를 읽어 SUMO 도로 파일들을 자동 생성하는 모듈.

SUMO는 도로를 여러 XML 파일로 정의한다:
    highway.nod.xml  : 노드(교차점/도로 끝점)의 좌표
    highway.edg.xml  : 엣지(도로 구간) — 어느 노드에서 어느 노드로, 차선 수, 제한속도
    highway.net.xml  : 위 두 파일을 netconvert 도구로 컴파일한 "완성된 도로망"
                       (SUMO가 실제로 읽는 파일. 직접 작성하기엔 복잡해서 도구로 생성)
    highway.rou.xml  : 차량 타입(vType)과 교통 수요(flow/route) 정의
    highway.sumocfg  : 위 파일들을 묶어주는 설정 파일 (시뮬레이션 실행의 진입점)

train.py / test.py 가 시작할 때마다 build()를 호출하므로,
road_config.py만 수정하면 도로가 자동으로 다시 만들어진다.
"""
import os
import subprocess

from sumolib import checkBinary  # SUMO 실행파일(sumo, netconvert 등) 경로를 찾아주는 유틸
from env.mdp_config import SIMULATION
try:
    from env.road_config import SCENERY
except ImportError:      # 구버전 설정 하위호환
    SCENERY = {"trees": False}
try:
    from env.road_config import NOHYEONG     # 노형오거리 형상 설정
except ImportError:
    NOHYEONG = None


def build(road: dict, ego: dict, traffic: dict, out_dir: str) -> str:
    """SUMO 파일들을 out_dir에 생성하고, sumocfg 파일 경로를 반환한다.

    Args:
        road:    configs.road_config.ROAD    (도로 형상)
        ego:     configs.road_config.EGO     (RL 차량 속성)
        traffic: configs.road_config.TRAFFIC (배경 교통류)
        out_dir: 생성 위치 (보통 프로젝트의 sumo/ 폴더)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── 도로 종류 분기 ──
    # ROAD["type"] == "nohyeong" 이면 노형오거리 생성 (파일 맨 아래 _build_nohyeong).
    # 그 외("straight" 또는 키 없음)는 기존 직선 도로 그대로 생성한다.
    if road.get("type", "straight") == "nohyeong":
        return _build_nohyeong(road, ego, traffic, out_dir)

    # ──────────────────────────────────────────────────────
    # 1) 노드 파일: 직선 도로의 양 끝점 2개
    #    n0 = 원점(출발), n1 = length만큼 떨어진 지점(도착)
    # ──────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "highway.nod.xml"), "w") as f:
        f.write(f"""<nodes>
    <node id="n0" x="0" y="0"/>
    <node id="n1" x="{road['length']}" y="0"/>
</nodes>
""")

    # ──────────────────────────────────────────────────────
    # 2) 엣지 파일: n0 → n1 을 잇는 도로 구간 하나
    #    numLanes = 차선 수, speed = 제한속도(m/s)
    # ──────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "highway.edg.xml"), "w") as f:
        f.write(f"""<edges>
    <edge id="e0" from="n0" to="n1" numLanes="{road['num_lanes']}" speed="{road['speed_limit']}"/>
</edges>
""")

    # ──────────────────────────────────────────────────────
    # 3) 라우트 파일: 차량 타입 2종 + 배경 교통 flow
    #
    #    <vType>  : 차량의 물리 속성 템플릿
    #       ego_type : RL이 조종할 차 (빨간색, color="1,0,0" = RGB)
    #    <vTypeDistribution> : 배경차 컨트롤러 믹스.
    #       TRAFFIC["controllers"]에 정의된 여러 차량추종모델(Krauss/IDM/
    #       EIDM/ACC 등)을 probability 비율로 섞는다. flow가 이 분포를
    #       type으로 쓰면, 투입되는 차량마다 SUMO가 확률적으로 타입을 뽑는다.
    #    <route>  : 지나갈 엣지들의 나열. 여기선 e0 하나뿐.
    #    <flow>   : 교통 수요 정의. period="X"는 "X초마다 정확히 1대씩"
    #               → 등간격(uniform) 생성. (vehsPerHour와 등가지만
    #               "비슷한 간격으로 새로 생성"을 명시적으로 보장)
    #               ego는 여기에 없고, 환경(sumo_env.py)이 reset마다
    #               traci.vehicle.add()로 직접 투입한다.
    # ──────────────────────────────────────────────────────
    vtypes_block, flow_type = _vtypes_block(traffic)

    with open(os.path.join(out_dir, "highway.rou.xml"), "w") as f:
        f.write(f"""<routes>
    <vType id="ego_type" accel="{ego['accel']}" decel="{ego['decel']}" maxSpeed="{ego['max_speed']}" color="1,0,0" vClass="passenger" guiShape="passenger/sedan"/>
{vtypes_block}
    <route id="r0" edges="e0"/>
    <flow id="traffic" type="{flow_type}" route="r0" begin="0" end="100000"
          period="{3600.0 / traffic['vehs_per_hour']:.2f}"
          departLane="{traffic['depart_lane']}" departSpeed="{traffic['depart_speed']}"/>
</routes>
""")

    # ──────────────────────────────────────────────────────
    # 4) GUI 기본 시각화 설정
    #    GUI 관련 값의 source of truth는 env/mdp_config.py의 SIMULATION 하나뿐이다.
    #    view_road.py / train.py / test.py / 데이터 수집 모두 build()를 거치므로
    #    gui_zoom/gui_delay를 바꾸면 생성되는 highway.gui.xml에도 즉시 반영된다.
    # ──────────────────────────────────────────────────────
    # GUI 시작 화면: "도로 시작 구간"이 차선까지 보이게 자동 확대해둔다.
    #   (12km 전체 맞춤으로 시작하면 도로가 가느다란 선으로만 보여 직관성이
    #    떨어지므로, ego가 등장할 시작부를 확대해 대기 화면으로 삼는다.
    #    ▶ 재생 후에는 env가 ego를 추적(trackVehicle)하며 시점을 가져간다.)
    # 좌표/배율은 도로 길이·차선 수에서 "계산"하므로 설정을 바꿔도 안 어긋난다:
    #   zoom(%) = 도로길이 / 보고싶은폭(gui_view_width) × 100
    #     (SUMO zoom은 "네트워크 전체 = 100%" 기준)
    #   y 중심 = -차선폭(3.2m) × 차선수 / 2   (SUMO는 차선을 y<0 쪽으로 그림)
    gui_delay = float(SIMULATION.get("gui_delay", 100.0))
    view_width = float(SIMULATION.get("gui_view_width", 500.0))   # 시작 화면 시야(m)
    zoom = float(road["length"]) / view_width * 100.0
    center_x = view_width * 0.4                                   # 시작점이 화면 왼쪽에 오게
    center_y = -3.2 * int(road["num_lanes"]) / 2.0
    gui_settings_path = os.path.join(out_dir, "highway.gui.xml")
    with open(gui_settings_path, "w") as f:
        f.write(f"""<viewsettings>
    <scheme name="real world"/>
    <viewport zoom="{zoom:.0f}" x="{center_x:.1f}" y="{center_y:.1f}" angle="0"/>
    <delay value="{gui_delay}"/>
</viewsettings>
""")

    # ──────────────────────────────────────────────────────
    # 4.7) 도로변 장식(나무) — 순수 시각 요소
    #
    # SUMO의 <poly>(다각형)로 "초록 수관 + 갈색 줄기"를 그려 도로
    # 양옆에 심는다. additional 파일로 로드되며 차량 물리와 완전히
    # 무관하다 (충돌/관측/보상 어디에도 안 잡힘).
    # 좌표계: 차선은 y ∈ [0, -3.2×차선수] 영역 → 왼쪽 숲은 y > 0,
    # 오른쪽 숲은 y < -3.2×차선수 - side_offset 에 배치.
    # ──────────────────────────────────────────────────────
    _write_scenery(out_dir, road)

    # ──────────────────────────────────────────────────────
    # 5) sumocfg: "이 net 파일과 이 route 파일로 시뮬레이션해라"
    # ──────────────────────────────────────────────────────
    cfg_path = os.path.join(out_dir, "highway.sumocfg")
    with open(cfg_path, "w") as f:
        f.write("""<configuration>
    <input>
        <net-file value="highway.net.xml"/>
        <route-files value="highway.rou.xml"/>
        <additional-files value="highway.scenery.xml"/>
    </input>
    <gui_only>
        <gui-settings-file value="highway.gui.xml"/>
    </gui_only>
</configuration>
""")

    # ──────────────────────────────────────────────────────
    # 6) netconvert 실행: nod + edg → net.xml 컴파일
    #    (SUMO 설치 시 함께 깔리는 커맨드라인 도구)
    # ──────────────────────────────────────────────────────
    netconvert = checkBinary("netconvert")
    subprocess.run(
        [netconvert,
         "--node-files", os.path.join(out_dir, "highway.nod.xml"),
         "--edge-files", os.path.join(out_dir, "highway.edg.xml"),
         "-o", os.path.join(out_dir, "highway.net.xml")],
        check=True,           # 실패하면 예외를 던져서 바로 알 수 있게
        capture_output=True,  # 출력은 조용히 삼킴
    )
    return cfg_path


def _vtypes_block(traffic: dict):
    """TRAFFIC["controllers"] → <vTypeDistribution> XML 문자열과 flow가 쓸 type 이름.

    직선 도로/회전교차로 공용. (배경차 종류·비율 정의는 도로 형상과 무관하므로 분리)
    """
    controllers = traffic.get("controllers")
    if controllers:
        # probability 합이 1이 아니어도 되도록 자동 정규화
        total_p = sum(float(c.get("probability", 1.0))
                      for c in controllers.values())
        vtype_lines = []
        for name, c in controllers.items():
            prob = float(c.get("probability", 1.0)) / total_p
            attrs = [
                f'id="car_{name}"',
                f'probability="{prob:.4f}"',
                f'carFollowModel="{c["carFollowModel"]}"',
                f'accel="{c.get("accel", 2.5)}"',
                f'decel="{c.get("decel", 4.5)}"',
                f'tau="{c.get("tau", 1.0)}"',
                f'minGap="{c.get("min_gap", 2.5)}"',
                # 컨트롤러별 희망속도 오버라이드 (없으면 공통 max_speed).
                # 컨트롤러마다 속도를 다르게 주면 차선 간 속도차가 생겨
                # "추월할 이유"가 자연스럽게 만들어진다.
                f'maxSpeed="{c.get("max_speed", traffic["max_speed"])}"',
                # speedDev: 같은 타입 안에서도 개체별 희망속도 편차 (0~)
                f'speedDev="{c.get("speed_dev", 0.1)}"',
                f'color="{c.get("color", "1,1,0")}"',
                'vClass="passenger"', 'guiShape="passenger/sedan"',
            ]
            # sigma는 Krauss 계열에만 의미가 있으므로 지정된 경우에만 넣는다
            if "sigma" in c:
                attrs.append(f'sigma="{c["sigma"]}"')
            # ── 배경차 차선 유지 (keep_lane, 기본 True) ──
            # 배경차의 자율 차선변경(전략/협조/속도이득/우측유지)을 전부 꺼서
            # 각자 자기 차선을 지키게 한다. 이유:
            #   배경차가 자유롭게 돌아다니면 (측정: 150스텝에 477회 변경)
            #   차선 간 틈이 수시로 닫혀서, ego가 틈을 노려 파고드는
            #   "칼치기 추월"을 학습할 안정적인 기회 자체가 사라진다.
            #   배경차가 차선을 유지하면 "느린 차 뒤 → 옆 틈으로 변경 →
            #   가속"이라는 추월 패턴이 재현 가능한 형태로 나타난다.
            # 특정 컨트롤러만 돌아다니게 하려면 그 항목에 "keep_lane": False.
            # 전체 기본값은 TRAFFIC["keep_lane"] (없으면 True). 교차로 도로(nohyeong)에서는
            # 배경차가 좌/우회전 차선으로 옮겨가야 하므로 False 로 둔다.
            if c.get("keep_lane", traffic.get("keep_lane", True)):
                attrs += ['lcStrategic="0"', 'lcCooperative="0"',
                          'lcSpeedGain="0"', 'lcKeepRight="0"']
            vtype_lines.append("        <vType " + " ".join(attrs) + "/>")
        vtypes_block = ('    <vTypeDistribution id="carDist">\n'
                        + "\n".join(vtype_lines)
                        + "\n    </vTypeDistribution>")
        flow_type = "carDist"
    else:
        # controllers 설정이 없으면 예전 방식(단일 Krauss)으로 동작 (하위호환)
        vtypes_block = (f'    <vType id="car" accel="2.5" decel="4.5" '
                        f'maxSpeed="{traffic["max_speed"]}" '
                        f'sigma="{traffic.get("sigma", 0.5)}" '
                        f'vClass="passenger" guiShape="passenger/sedan"/>')
        flow_type = "car"

    return vtypes_block, flow_type


def _blob(cx: float, cy: float, r: float, rng, n: int = 14,
          roughness: float = 0.35) -> str:
    """유기적인(울퉁불퉁한) 수관 실루엣 다각형을 만든다.

    실제 나무를 위에서 내려다보면 완전한 원이 아니라 잎 뭉치들이
    만드는 불규칙한 테두리다. 꼭짓점마다 반지름을 무작위로 흔들되,
    이웃 꼭짓점과 평균을 내서(smoothing) 뾰족하지 않고 부드럽게
    울퉁불퉁한 실루엣을 얻는다.
    """
    import math
    raw = [1.0 + rng.uniform(-roughness, roughness) for _ in range(n)]
    sm = [(raw[i - 1] + raw[i] + raw[(i + 1) % n]) / 3.0 for i in range(n)]
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * sm[i]
        pts.append(f"{cx + rr * math.cos(a):.2f},{cy + rr * math.sin(a):.2f}")
    return " ".join(pts)


def _write_scenery(out_dir: str, road: dict):
    """도로변 나무를 highway.scenery.xml (additional 파일)로 생성.

    항공뷰(top-down) 조경처럼 보이도록 나무 한 그루를 여러 겹으로 그린다:
      layer 1  그림자  : 수관을 남동쪽으로 살짝 밀어낸 반투명 검정 blob
                         (해가 북서쪽에 있는 항공사진 느낌)
      layer 2  수관 몸통: 중간톤 초록의 불규칙 실루엣
      layer 3  잎 뭉치  : 수관 안에 어두운/짙은 클럼프 1~2개
                         → 캐노피의 입체적인 질감
      layer 4  하이라이트: 북서쪽으로 치우친 밝은 초록 작은 blob
    배치도 실제 조경처럼: 단독목 / 2~3그루 군락 / 낮은 관목이 섞인다.
    SCENERY["trees"]가 False면 빈 additional 파일 생성.
    """
    import random
    path = os.path.join(out_dir, "highway.scenery.xml")
    cfg = SCENERY if isinstance(SCENERY, dict) else {}
    if not cfg.get("trees", False):
        with open(path, "w") as f:
            f.write("<additional/>\n")
        return

    rng = random.Random(int(cfg.get("seed", 7)))
    spacing = float(cfg.get("tree_spacing", 45.0))
    jitter = float(cfg.get("tree_jitter", 0.5))
    side = float(cfg.get("side_offset", 6.0))
    length = float(road["length"])
    road_bottom = -3.2 * int(road["num_lanes"])

    # (몸통, 어두운 클럼프, 밝은 하이라이트) 초록 팔레트 — 수종별 톤
    palettes = [
        ("46,110,44", "32,84,32", "96,158,84"),
        ("38,98,40",  "26,72,28", "84,146,74"),
        ("56,122,50", "40,92,38", "110,168,92"),
        ("30,88,36",  "20,64,26", "76,138,70"),
    ]
    bush_greens = ["70,124,58", "84,136,64", "62,112,52"]

    lines = ["<additional>"]
    tid = [0]

    def draw_tree(x, y, r):
        body, dark, light = rng.choice(palettes)
        i = tid[0]; tid[0] += 1
        # 그림자 (남동쪽 +x, -y 로 오프셋, 반투명)
        lines.append(f'    <poly id="sh_{i}" color="20,26,18,70" fill="1" '
                     f'layer="1" shape="{_blob(x + r*0.35, y - r*0.35, r*1.02, rng)}"/>')
        # 수관 몸통
        lines.append(f'    <poly id="tr_{i}" color="{body}" fill="1" '
                     f'layer="2" shape="{_blob(x, y, r, rng)}"/>')
        # 잎 뭉치 1~2개 (어두운 톤, 수관 내부 무작위 위치)
        for k in range(rng.randint(1, 2)):
            ox = rng.uniform(-0.35, 0.35) * r
            oy = rng.uniform(-0.35, 0.35) * r
            lines.append(f'    <poly id="cl_{i}_{k}" color="{dark}" fill="1" '
                         f'layer="3" shape="{_blob(x + ox, y + oy, r * rng.uniform(0.4, 0.6), rng, n=10)}"/>')
        # 하이라이트 (북서쪽 -x, +y — 그림자 반대 방향)
        lines.append(f'    <poly id="hl_{i}" color="{light}" fill="1" '
                     f'layer="4" shape="{_blob(x - r*0.28, y + r*0.28, r*0.34, rng, n=8, roughness=0.25)}"/>')

    def draw_bush(x, y):
        i = tid[0]; tid[0] += 1
        g = rng.choice(bush_greens)
        r = rng.uniform(1.0, 1.8)
        lines.append(f'    <poly id="bsh_{i}" color="20,26,18,55" fill="1" '
                     f'layer="1" shape="{_blob(x + 0.3, y - 0.3, r, rng, n=8)}"/>')
        lines.append(f'    <poly id="bs_{i}" color="{g}" fill="1" '
                     f'layer="2" shape="{_blob(x, y, r, rng, n=8)}"/>')

    for side_sign, base_y in [(+1, side), (-1, road_bottom - side)]:
        x = rng.uniform(0, spacing)
        while x < length:
            y = base_y + side_sign * rng.uniform(0, 4.0)
            roll = rng.random()
            if roll < 0.12:
                draw_bush(x, y)                        # 15%: 낮은 관목
            elif roll < 0.67:
                draw_tree(x, y, rng.uniform(2.6, 4.6))  # 55%: 단독목 (실제 수관 스케일)
            else:
                # 25%: 2~3그루 군락 (겹치듯 붙여 심어 숲 덩어리 느낌)
                n = rng.randint(2, 4)
                for _ in range(n):
                    gx = x + rng.uniform(-5.5, 5.5)
                    gy = y + side_sign * rng.uniform(-2.0, 4.5)
                    draw_tree(gx, gy, rng.uniform(2.2, 3.8))
                x += spacing * 0.4                     # 군락 뒤엔 간격 조금 더
            x += spacing * rng.uniform(1.0 - jitter, 1.0 + jitter)
    lines.append("</additional>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════
#  제주 노형오거리 (ROAD["type"] == "nohyeong")
#
#  용어는 road_config.py 의 "용어 정리" 와 같다.
#    도로 A1~A5 (코드 이름 arms) / 원형도로 (코드 이름 ring_segments) / 갈림점 / 노드·엣지
#
#  만들어지는 것 (NOHYEONG 설정 기준):
#    노드 (길이 만나는 점)
#      nC          : 교차로 가운데 (신호등). 다섯 도로가 모두 여기서 만난다.
#      nE_A1 …     : 각 도로의 바깥 끝. 차가 생겨나고 사라지는 곳.
#      nM_A1_0 …   : 각 도로 위의 갈림점 (원형도로가 갈라지거나 합류하는 곳). 교차로에 가까운 것부터 0, 1, 2 …
#    엣지 (두 노드 사이의 한 방향 도로 구간). 도로 하나가 갈림점마다 잘려서 여러 구간이 된다.
#      A1_in, A1_in1, A1_in2 …    : 교차로로 "들어오는" 방향. A1_in 이 교차로에 바로 닿는 구간, 숫자가 클수록 바깥.
#      A1_out, A1_out1, A1_out2 … : 교차로에서 "나가는" 방향. A1_out 이 교차로에서 바로 나오는 구간.
#      (갈림점이 없는 도로는 A1_in / A1_out 두 개뿐 — 교수님 코드와 같은 이름 규칙)
#      RING_A1_A3, OUTER_N …      : 원형도로 (road_config.py 의 id 그대로)
#    연결 정리 (highway.con.xml)
#      도로 끝·갈림점에서 SUMO 가 멋대로 만드는 U턴, 반대 방향으로 원형도로에 들어가는 연결 등을 지운다.
#    경로/교통
#      r0          : ego 기본 경로 (ego_route)
#      ego_d_*/ego_r_* : ego 후보 경로 (highway.ego.rou.xml). sumo_env.py 가 에피소드마다 하나를 뽑는다.
#      f_A1_A2 …   : 도로 A1 → 도로 A2 로 가는 배경 차량 (SUMO 가 가장 빠른 길로 보냄)
#      lf_* / lr_* : 원형도로를 일부러 지나가는 배경 차량
#    장식
#      녹지 + 횡단보도 줄무늬 그림 (highway.scenery.xml). 차 움직임과 무관.
# ══════════════════════════════════════════════════════════════
import math


def _polar(r: float, angle_deg: float):
    """극좌표 (반지름, 각도°) → 직교좌표 (x, y). 각도는 동쪽 0°, 반시계 +."""
    a = math.radians(angle_deg)
    return r * math.cos(a), r * math.sin(a)


def _fmt_shape(points) -> str:
    """[(x,y), ...] → SUMO shape 속성 문자열 'x1,y1 x2,y2 ...'."""
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)




def _build_nohyeong(road: dict, ego: dict, traffic: dict, out_dir: str) -> str:
    cfg = NOHYEONG
    if cfg is None:
        raise RuntimeError("ROAD['type']='nohyeong' 인데 road_config.py 에 NOHYEONG 설정이 없습니다.")

    arm_speed = float(road["speed_limit"])
    arms = list(cfg["arms"])
    by_id = {a["id"]: a for a in arms}
    segs = list(cfg.get("ring_segments", []))

    # ── 0) 도로마다 갈림점 위치(교차로 중심에서의 거리) 모으기 ──
    #   원형도로가 갈라지는 곳(from_r)과 합류하는 곳(to_r). 같은 거리는 하나의 갈림점으로 합친다.
    radii = {a["id"]: set() for a in arms}
    for sg in segs:
        radii[sg["from"]].add(float(sg["from_r"]))
        radii[sg["to"]].add(float(sg["to_r"]))
    for a in arms:
        if a.get("split_at"):
            radii[a["id"]].add(float(a["split_at"]))
    splits = {aid: sorted(rs) for aid, rs in radii.items()}       # 안쪽(작은 r) → 바깥쪽

    def node_at(aid, r):
        """도로 aid 위, 교차로 중심에서 r 떨어진 갈림점의 노드 이름."""
        return f"nM_{aid}_{splits[aid].index(float(r))}"

    # 도로 구간(엣지) 이름: k=0 이 교차로에 바로 붙는 구간
    def in_edge(aid, k):  return f"{aid}_in" if k == 0 else f"{aid}_in{k}"
    def out_edge(aid, k): return f"{aid}_out" if k == 0 else f"{aid}_out{k}"

    # ── 1) 노드 ──
    nodes = [f'    <node id="nC" x="0" y="0" type="{cfg.get("center_node_type", "traffic_light")}"/>']
    for a in arms:
        x, y = _polar(float(a["length"]), a["angle"])
        nodes.append(f'    <node id="nE_{a["id"]}" x="{x:.2f}" y="{y:.2f}"/>')
        for k, r in enumerate(splits[a["id"]]):
            x, y = _polar(r, a["angle"])
            nodes.append(f'    <node id="nM_{a["id"]}_{k}" x="{x:.2f}" y="{y:.2f}" type="priority"/>')
    with open(os.path.join(out_dir, "highway.nod.xml"), "w") as f:
        f.write("<nodes>\n" + "\n".join(nodes) + "\n</nodes>\n")

    # ── 2) 엣지 ──
    edges = []
    entry_edge, exit_edge = {}, {}     # 도로 id → 차가 처음 들어서는 구간 / 마지막으로 빠져나가는 구간
    chain_in, chain_out = {}, {}       # 도로 id → 들어오는 구간들(바깥→안 순서), 나가는 구간들(안→바깥 순서)
    for a in arms:
        aid, L, S = a["id"], int(a["lanes"]), arm_speed
        K = len(splits[aid])
        # 교차로에서부터 노드 나열: nC, 갈림점 0, 갈림점 1, …, 도로 끝
        pts = ["nC"] + [f"nM_{aid}_{k}" for k in range(K)] + [f"nE_{aid}"]
        ins, outs = [], []
        for k in range(K + 1):
            inner, outer = pts[k], pts[k + 1]
            # priority="3": 갈림점에서 큰 도로(3)가 원형도로(1)보다 우선 (원형도로 차가 양보)
            edges.append(f'    <edge id="{in_edge(aid, k)}" from="{outer}" to="{inner}" '
                         f'numLanes="{L}" speed="{S}" priority="3"/>')
            edges.append(f'    <edge id="{out_edge(aid, k)}" from="{inner}" to="{outer}" '
                         f'numLanes="{L}" speed="{S}" priority="3"/>')
            ins.append(in_edge(aid, k)); outs.append(out_edge(aid, k))
        chain_in[aid] = ins[::-1]          # 바깥 → 안 순서
        chain_out[aid] = outs              # 안 → 바깥 순서
        entry_edge[aid], exit_edge[aid] = ins[-1], outs[-1]
    # 원형도로
    Lr, Sr = int(cfg.get("ring_lanes", 1)), float(cfg.get("ring_speed", 8.33))
    seg_count = {}
    for sg in segs:
        via = list(sg.get("via", []))
        shp = f' shape="{_fmt_shape(via)}"' if via else ""
        key = (sg["from"], sg["to"]); seg_count[key] = seg_count.get(key, 0) + 1
        suffix = "" if seg_count[key] == 1 else f"_{seg_count[key]}"     # 같은 도로 쌍을 잇는 길이 두 개면 _2
        sg["_eid"] = sg.get("id") or f'RING_{sg["from"]}_{sg["to"]}{suffix}'
        edges.append(f'    <edge id="{sg["_eid"]}" from="{node_at(sg["from"], sg["from_r"])}" '
                     f'to="{node_at(sg["to"], sg["to_r"])}" numLanes="{int(sg.get("lanes", Lr))}" '
                     f'speed="{float(sg.get("speed", Sr))}" priority="1"{shp}/>')
    with open(os.path.join(out_dir, "highway.edg.xml"), "w") as f:
        f.write("<edges>\n" + "\n".join(edges) + "\n</edges>\n")

    # ── 3) 연결 파일: 엉뚱한 U턴 제거 ──
    #   netconvert(SUMO 도로 변환 도구)는 모든 노드에서 "맨 왼쪽 차선 U턴"을 자동으로 만든다.
    #   교차로 가운데(nC) U턴만 남기고, 도로 끝·갈림점의 U턴은 지운다. (uturn=False 면 nC 것도 지운다)
    cons = []
    for a in arms:
        aid, K = a["id"], len(splits[a["id"]])
        if not cfg.get("uturn", True):
            cons.append(f'    <delete from="{in_edge(aid, 0)}" to="{out_edge(aid, 0)}"/>')
        for k in range(K + 1):
            # 갈림점·도로 끝에서: 나가던 차가 되돌아오는 U턴 / 들어오던 차가 되돌아가는 U턴
            cons.append(f'    <delete from="{out_edge(aid, k)}" to="{in_edge(aid, k)}"/>')
            if k + 1 <= K:
                cons.append(f'    <delete from="{in_edge(aid, k + 1)}" to="{out_edge(aid, k + 1)}"/>')
    # 원형도로는 올바른 방향으로만 드나들게 한다:
    #   · 갈라지는 갈림점: 교차로로 "들어오는" 차만 원형도로로 빠질 수 있다.
    #                     교차로에서 "나가던" 차가 다시 꺾어 들어가는 연결은 지운다.
    #   · 합류하는 갈림점: 원형도로 차는 교차로에서 "나가는" 방향으로만 합류한다.
    #                     교차로 쪽으로 되돌아 들어가는 연결은 지운다.
    #   (갈림점이 도로 한가운데 선 위에 있어서 netconvert 가 양방향 모두 이어 버리기 때문)
    for sg in segs:
        k_from = splits[sg["from"]].index(float(sg["from_r"]))
        k_to = splits[sg["to"]].index(float(sg["to_r"]))
        cons.append(f'    <delete from="{out_edge(sg["from"], k_from)}" to="{sg["_eid"]}"/>')
        cons.append(f'    <delete from="{sg["_eid"]}" to="{in_edge(sg["to"], k_to)}"/>')
    with open(os.path.join(out_dir, "highway.con.xml"), "w") as f:
        f.write("<connections>\n" + "\n".join(cons) + "\n</connections>\n")

    # ── 4) 경로: ego 기본 경로 r0 + 배경 차량 ──
    vtypes_block, flow_type = _vtypes_block(traffic)
    er = cfg["ego_route"]
    r0_edges = chain_in[er["from"]] + chain_out[er["to"]]

    pairs = [(a, b) for a in arms for b in arms if cfg.get("uturn", True) or a["id"] != b["id"]]
    uf = float(cfg.get("uturn_flow_factor", 0.15))        # U턴 수요는 직진/회전보다 훨씬 적다
    def _pw(a, b):
        w = float(a.get("weight", 1)) * float(b.get("weight", 1))
        return w * uf if a["id"] == b["id"] else w
    total_w = sum(_pw(a, b) for a, b in pairs)
    flows = []
    for a, b in pairs:
        w = _pw(a, b) / total_w
        period = 3600.0 / (float(traffic["vehs_per_hour"]) * w)      # 이 경로에 몇 초마다 1대
        flows.append(
            f'    <flow id="f_{a["id"]}_{b["id"]}" type="{flow_type}" '
            f'from="{entry_edge[a["id"]]}" to="{exit_edge[b["id"]]}" '
            f'begin="0" end="100000" period="{period:.2f}" '
            f'departLane="{traffic["depart_lane"]}" departSpeed="{traffic["depart_speed"]}"/>')
    with open(os.path.join(out_dir, "highway.rou.xml"), "w") as f:
        f.write(f"""<routes>
    <vType id="ego_type" accel="{ego['accel']}" decel="{ego['decel']}" maxSpeed="{ego['max_speed']}" color="1,0,0" vClass="passenger" guiShape="passenger/sedan"/>
{vtypes_block}
    <route id="r0" edges="{' '.join(r0_edges)}"/>
{chr(10).join(flows + _loop_flows(cfg, segs, entry_edge, exit_edge, flow_type, traffic))}
</routes>
""")

    # ── 5) GUI: 교차로 중심부가 화면에 들어오게 ──
    gui_delay = float(SIMULATION.get("gui_delay", 100.0))
    extent = 2 * max(float(a["length"]) for a in arms)      # 네트워크 전체 폭 (m)
    view_width = float(SIMULATION.get("gui_view_width", 500.0))
    zoom = extent / min(view_width, extent) * 100.0
    with open(os.path.join(out_dir, "highway.gui.xml"), "w") as f:
        f.write(f"""<viewsettings>
    <scheme name="real world"/>
    <viewport zoom="{zoom:.0f}" x="0" y="0" angle="0"/>
    <delay value="{gui_delay}"/>
</viewsettings>
""")

    # ── 6) sumocfg + netconvert ──
    cfg_path = os.path.join(out_dir, "highway.sumocfg")
    with open(cfg_path, "w") as f:
        f.write("""<configuration>
    <input>
        <net-file value="highway.net.xml"/>
        <route-files value="highway.rou.xml,highway.ego.rou.xml"/>
        <additional-files value="highway.scenery.xml"/>
    </input>
    <gui_only>
        <gui-settings-file value="highway.gui.xml"/>
    </gui_only>
</configuration>
""")
    netconvert = checkBinary("netconvert")
    result = subprocess.run(
        [netconvert,
         "--node-files", os.path.join(out_dir, "highway.nod.xml"),
         "--edge-files", os.path.join(out_dir, "highway.edg.xml"),
         "--connection-files", os.path.join(out_dir, "highway.con.xml"),
         "-o", os.path.join(out_dir, "highway.net.xml"),
         "--offset.disable-normalization", "true",   # 좌표를 (0,0) 구석으로 옮기지 말 것
                                                     # (녹지 poly·GUI 중심이 우리 좌표계 기준이므로)
         "--junctions.join", "false",     # 가까운 노드끼리 멋대로 합치지 않게
         "--junctions.corner-detail", "5",  # 교차로 모서리를 둥글게 (보기 좋게)
         "--tls.default-type", "static"],   # 신호: 고정 주기 (netconvert 가 자동 생성)
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("netconvert 실패:\n" + result.stderr)

    # ── 7) ego 후보 경로 (sumo_env.py 가 에피소드마다 하나를 뽑는다) ──
    _write_ego_routes(out_dir, cfg, arms, segs, entry_edge, exit_edge)

    # ── 8) 장식: 녹지 + 횡단보도 줄무늬 ──
    #   횡단보도는 SUMO의 보행자 <crossing> 이 아니라 그림(poly)이다.
    #   (정식 crossing 은 인도 차선이 필요해서 차선 번호가 밀리고 sumo_env 의 차선 로직과 충돌한다.)
    _write_nohyeong_scenery(out_dir, cfg, arms, splits)
    return cfg_path


def _write_ego_routes(out_dir, cfg, arms, segs, entry_edge, exit_edge):
    """ego 가 탈 수 있는 후보 경로를 highway.ego.rou.xml 에 저장한다.

    경로 이름 규칙 (sumo_env.py 가 이 접두어로 구분한다):
      ego_d_<출발>_<도착>          : 교차로를 바로 지나는 경로 (최단 경로)
      ego_r_<원형도로>_<출발>_<도착> : 원형도로를 반드시 한 번 이상 거치는 경로
    완성된 도로망(net.xml)에서 sumolib 로 실제 최단 경로를 계산하므로,
    갈 수 없는 조합은 자동으로 빠진다.
    """
    import sumolib
    net = sumolib.net.readNet(os.path.join(out_dir, "highway.net.xml"))
    er = cfg.get("ego_routes", {})
    uturn = bool(er.get("uturn", True))

    def path(edge_ids):
        """edge_ids 를 순서대로 지나는 최단 경로의 엣지 목록 (불가능하면 None)."""
        full = []
        for a, b in zip(edge_ids[:-1], edge_ids[1:]):
            p, _ = net.getShortestPath(net.getEdge(a), net.getEdge(b), vClass="passenger")
            if p is None:
                return None
            ids = [e.getID() for e in p]
            full += ids if not full else ids[1:]
        return full

    routes = {}
    for a in arms:                                   # 1) 바로 가는 경로
        for b in arms:
            if a["id"] == b["id"] and not uturn:
                continue
            r = path([entry_edge[a["id"]], exit_edge[b["id"]]])
            if r:
                routes[f'ego_d_{a["id"]}_{b["id"]}'] = r
    # 2) 원형도로를 거치는 경로: 원형도로 하나(또는 loop_routes 묶음)를 지난 뒤 아무 도로로나 나간다
    ring_groups = [([sg["_eid"]], sg["from"], sg["_eid"]) for sg in segs]
    ids = {sg.get("id"): sg["_eid"] for sg in segs if sg.get("id")}
    for i, lr in enumerate(cfg.get("loop_routes", [])):
        ring_groups.append(([ids.get(v, v) for v in lr["via"]], lr["from"], f"loop{i}"))
    for via, start, tag in ring_groups:
        for b in arms:
            r = path([entry_edge[start]] + via + [exit_edge[b["id"]]])
            if r and len(set(r)) == len(r):          # 같은 길을 두 번 지나는 경로는 제외
                routes[f'ego_r_{tag}_{start}_{b["id"]}'] = r
    with open(os.path.join(out_dir, "highway.ego.rou.xml"), "w") as f:
        f.write("<routes>\n" + "\n".join(
            f'    <route id="{rid}" edges="{" ".join(r)}"/>' for rid, r in routes.items()) + "\n</routes>\n")
    return routes


def _loop_flows(cfg, segs, entry_edge, exit_edge, flow_type, traffic):
    """원형도로를 일부러 지나가는 배경 차량(flow = 일정 간격으로 계속 생겨나는 차량 흐름)을 만든다.

    일반 flow 는 출발·도착만 정하고 SUMO 가 최단 경로를 고르기 때문에, 돌아가는 원형도로는
    아무도 안 쓴다. 여기서는 via(반드시 지나갈 엣지)를 지정해서 원형도로에도 차가 다니게 한다.
      · ring_segments[*]["vph"] : 그 길 하나만 지나가는 차 (from 도로 → 그 길 → to 도로)
      · loop_routes[*]          : 여러 길을 이어서 지나가는 차
    """
    def flow(fid, fa, ta, via, vph):
        if not vph or float(vph) <= 0:
            return None
        return (f'    <flow id="{fid}" type="{flow_type}" from="{entry_edge[fa]}" to="{exit_edge[ta]}" '
                f'via="{" ".join(via)}" begin="0" end="100000" period="{3600.0 / float(vph):.2f}" '
                f'departLane="{traffic["depart_lane"]}" departSpeed="{traffic["depart_speed"]}"/>')
    out = []
    for sg in segs:
        out.append(flow(f'lf_{sg["_eid"]}', sg["from"], sg["to"], [sg["_eid"]], sg.get("vph", 0)))
    ids = {sg.get("id"): sg["_eid"] for sg in segs if sg.get("id")}
    for i, lr in enumerate(cfg.get("loop_routes", [])):
        via = [ids.get(v, v) for v in lr["via"]]
        out.append(flow(f'lr_{i}_{lr["from"]}_{lr["to"]}', lr["from"], lr["to"], via, lr.get("vph", 0)))
    return [f for f in out if f]


def _write_nohyeong_scenery(out_dir, cfg, arms, splits):
    import sumolib
    net = sumolib.net.readNet(os.path.join(out_dir, "highway.net.xml"))
    LANE_W = 3.2
    cw_w = float(cfg.get("crosswalk_width", 4.0))
    cw_cfg = cfg.get("crosswalks", {})
    lines = ["<additional>"]
    for isl in cfg.get("islands", []):
        lines.append(f'    <poly id="{isl["id"]}" color="{isl.get("color", "168,196,150")}" fill="1" '
                     f'layer="-1" shape="{_fmt_shape(isl["pts"])}"/>')

    def stripes(pid, r_center, angle_deg, half_width):
        """도로 방향 angle 에 수직으로, 교차로 중심에서 r_center 떨어진 곳에 폭 cw_w 의 횡단보도 줄무늬."""
        a = math.radians(angle_deg)
        ux, uy = math.cos(a), math.sin(a)          # 도로 방향 (길이 1 짜리 방향 화살표)
        nx, ny = -uy, ux                           # 수직 방향
        stripe, gap = 0.6, 0.6
        t, k = -half_width, 0
        while t + stripe <= half_width + 1e-6:
            cx, cy = ux * r_center, uy * r_center
            p0 = (cx + nx * t - ux * cw_w / 2,            cy + ny * t - uy * cw_w / 2)
            p1 = (cx + nx * (t + stripe) - ux * cw_w / 2, cy + ny * (t + stripe) - uy * cw_w / 2)
            p2 = (cx + nx * (t + stripe) + ux * cw_w / 2, cy + ny * (t + stripe) + uy * cw_w / 2)
            p3 = (cx + nx * t + ux * cw_w / 2,            cy + ny * t + uy * cw_w / 2)
            lines.append(f'    <poly id="{pid}_{k}" color="245,245,245" fill="1" layer="2" '
                         f'shape="{_fmt_shape([p0, p1, p2, p3])}"/>')
            t += stripe + gap
            k += 1

    for a in arms:
        aid = a["id"]
        half_w = int(a["lanes"]) * LANE_W + 0.3      # 양방향 전체 폭의 절반 (+여유)
        mode = cw_cfg.get(aid, "center")
        if isinstance(mode, (int, float)):
            r = float(mode)
        elif mode == "center":
            x, y = net.getEdge(f"{aid}_in").getShape()[-1]     # 교차로 경계
            r = math.hypot(x, y) + cw_w / 2 + 0.5
        else:
            r = None
        if r is not None:
            stripes(f"cw_{aid}", r, a["angle"], half_w)
        if cfg.get("ring_crosswalks", False):
            for k, rs in enumerate(splits[aid]):
                stripes(f"cwr_{aid}_{k}", rs + 7.0, a["angle"], half_w)   # 갈림점 바로 바깥
    lines.append("</additional>")
    with open(os.path.join(out_dir, "highway.scenery.xml"), "w") as f:
        f.write("\n".join(lines) + "\n")
