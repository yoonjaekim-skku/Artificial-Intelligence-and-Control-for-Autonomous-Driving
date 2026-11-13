"""Check physical bumper gaps and the recorded last stop in evaluation."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from env.sumo_env import SumoHighwayEnv
from utils.evaluator import _DrivingStats


class StopClearanceTests(unittest.TestCase):
    def setUp(self):
        self.env = SumoHighwayEnv.__new__(SumoHighwayEnv)
        self.env.ego_id = "ego"
        self.env._ego_is_present = lambda: True
        self.env._upcoming_signals = lambda: [("tls", 0, 10.0, "r")]
        self.vehicle = Mock()
        self.vehicle.getSpeed.side_effect = lambda vid: {"ego": 0.0, "leader": 0.0}[vid]
        self.vehicle.getLeader.return_value = ("leader", 4.5)
        self.vehicle.getMinGap.return_value = 2.5

    def test_waiting_leader_uses_physical_bumper_gap(self):
        with patch("env.sumo_env.traci", SimpleNamespace(vehicle=self.vehicle)):
            result = self.env.get_stop_clearance()
        self.assertEqual(result, {"stop_clearance_m": 7.0, "stop_reference": "queued_vehicle"})

    def test_nearer_stop_line_takes_priority(self):
        self.env._upcoming_signals = lambda: [("tls", 0, 3.0, "r")]
        with patch("env.sumo_env.traci", SimpleNamespace(vehicle=self.vehicle)):
            result = self.env.get_stop_clearance()
        self.assertEqual(result, {"stop_clearance_m": 3.0, "stop_reference": "stop_line"})

    def test_moving_ego_does_not_record_a_stop(self):
        self.vehicle.getSpeed.side_effect = lambda vid: 2.0
        with patch("env.sumo_env.traci", SimpleNamespace(vehicle=self.vehicle)):
            self.assertIsNone(self.env.get_stop_clearance())

    def test_no_reference_is_missing_instead_of_zero(self):
        self.env._upcoming_signals = lambda: []
        self.vehicle.getLeader.return_value = None
        with patch("env.sumo_env.traci", SimpleNamespace(vehicle=self.vehicle)):
            result = self.env.get_stop_clearance()
        self.assertEqual(result, {"stop_clearance_m": None, "stop_reference": None})

    def test_park_penalty_only_when_slow_and_short_of_target(self):
        env = self.env
        env.signal_visibility = 100.0
        env.rw = {"red_park_penalty_coef": 0.1, "red_park_scale": 10.0,
                  "red_park_max_speed": 2.0, "red_stop_margin": 1.5}
        self.vehicle.getLeader.return_value = None
        with patch("env.sumo_env.traci", SimpleNamespace(vehicle=self.vehicle)):
            self.assertAlmostEqual(env._red_park_penalty(("tls", 0, 6.5, "r"), 0.0), 0.05)
            self.assertAlmostEqual(env._red_park_penalty(("tls", 0, 40.0, "r"), 1.0), 0.1)   # 상한
            self.assertEqual(env._red_park_penalty(("tls", 0, 6.5, "r"), 5.0), 0.0)          # 접근 중
            self.assertEqual(env._red_park_penalty(("tls", 0, 1.0, "r"), 0.0), 0.0)          # 목표 안쪽
            self.assertEqual(env._red_park_penalty(("tls", 0, 6.5, "G"), 0.0), 0.0)          # 초록불

    def test_evaluation_keeps_last_stop_after_driving_again(self):
        stats = _DrivingStats(1)
        stats.record(0, 0, False, {"stop_clearance_m": 12.0, "stop_reference": "stop_line"})
        stats.record(0, 0, False, {"stop_clearance_m": 3.0, "stop_reference": "queued_vehicle"})
        stats.record(0, 2, False, {"arrived": True, "speed": 10.0})
        self.assertEqual(stats.episodes[0].last_stop_clearance_m, 3.0)
        self.assertEqual(stats.episodes[0].last_stop_reference, "queued_vehicle")


if __name__ == "__main__":
    unittest.main()
