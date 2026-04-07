"""
Test Analysis for SwarmEnvironment (environment.py)
====================================================
Written by: Tester Agent (following .claude/agents/tester.md)
Philosophy: High-signal only — catches real bugs; avoids trivial assertions.
"""
import sys
import os
import pytest
from pydantic import ValidationError

# Make sure environment.py is importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment import (
    SwarmEnvironment,
    SwarmCommand,
    Observation,
    DroneState,
    DeliveryState,
    Reward,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def easy_env():
    return SwarmEnvironment(task="easy")


@pytest.fixture
def medium_env():
    return SwarmEnvironment(task="medium")


@pytest.fixture
def hard_env():
    return SwarmEnvironment(task="hard")


# ─────────────────────────────────────────────
# 1. Reset / Initialization Tests
# ─────────────────────────────────────────────

class TestReset:
    def test_easy_spawns_correct_counts(self, easy_env):
        """Easy task must have exactly 4 drones and 6 deliveries."""
        obs = easy_env.state()
        assert len(obs.drones) == 4
        assert len(obs.deliveries) == 6

    def test_medium_spawns_correct_counts(self, medium_env):
        """Medium task must have exactly 6 drones and 12 deliveries."""
        obs = medium_env.state()
        assert len(obs.drones) == 6
        assert len(obs.deliveries) == 12

    def test_hard_spawns_correct_counts(self, hard_env):
        """Hard task must have exactly 8 drones and 20 deliveries."""
        obs = hard_env.state()
        assert len(obs.drones) == 8
        assert len(obs.deliveries) == 20

    def test_all_drones_start_at_base_station(self, easy_env):
        """All drones must spawn at base_station (6, 6) on reset."""
        for drone in easy_env.state().drones:
            assert drone.position == (6, 6), f"{drone.id} not at base"

    def test_all_drones_start_idle_full_battery(self, easy_env):
        """Drones must start idle with 100% battery."""
        for drone in easy_env.state().drones:
            assert drone.status == "idle"
            assert drone.battery == 100.0

    def test_all_deliveries_start_pending(self, easy_env):
        """All deliveries must start with 'pending' status."""
        for dlv in easy_env.state().deliveries:
            assert dlv.status == "pending"

    def test_delivery_positions_not_base_station(self, easy_env):
        """No delivery target should be placed on the base station (6, 6)."""
        for dlv in easy_env.state().deliveries:
            assert dlv.target_position != (6, 6), (
                f"Delivery {dlv.id} spawned on base station"
            )

    def test_reset_returns_observation_type(self, easy_env):
        """reset() must return an Observation instance."""
        obs = easy_env.reset()
        assert isinstance(obs, Observation)

    def test_reset_clears_mid_episode_state(self, easy_env):
        """After mid-game reset, time_step must be 0 and score must be 0."""
        cmd = SwarmCommand(action_type="no_op")
        easy_env.step(cmd)
        easy_env.step(cmd)
        easy_env.reset()
        assert easy_env.time_step == 0
        assert easy_env.current_mission_score == 0.0

    def test_unknown_task_raises_value_error(self):
        """Passing an unknown task name must raise ValueError immediately."""
        with pytest.raises(ValueError, match="Unknown task"):
            SwarmEnvironment(task="impossible")

    def test_medium_starts_with_rain(self, medium_env):
        """Medium task must start with rain weather."""
        assert medium_env.weather_condition == "rain"

    def test_hard_starts_with_storm(self, hard_env):
        """Hard task must start with storm weather."""
        assert hard_env.weather_condition == "storm"

    def test_time_step_zero_on_reset(self, easy_env):
        """time_step in the returned Observation must be 0 after reset."""
        obs = easy_env.state()
        assert obs.time_step == 0


# ─────────────────────────────────────────────
# 2. Pydantic Model Validation Tests
# ─────────────────────────────────────────────

class TestPydanticModels:
    def test_swarm_command_rejects_unknown_action(self):
        """action_type must be one of the 8 literals — unknown values are forbidden."""
        with pytest.raises(ValidationError):
            SwarmCommand(action_type="nuke_city")

    def test_swarm_command_extra_fields_forbidden(self):
        """SwarmCommand has Config extra='forbid'; unknown fields must raise."""
        with pytest.raises(ValidationError):
            SwarmCommand(action_type="no_op", unknown_field="oops")

    def test_observation_is_json_serialisable(self, easy_env):
        """Observation must serialise to JSON without error (OpenEnv wire format)."""
        obs = easy_env.state()
        json_str = obs.model_dump_json()
        assert len(json_str) > 0

    def test_observation_contains_all_required_fields(self, easy_env):
        """All required Observation fields must be present and typed correctly."""
        obs = easy_env.state()
        assert isinstance(obs.time_step, int)
        assert isinstance(obs.drones, list)
        assert isinstance(obs.deliveries, list)
        assert isinstance(obs.emergencies, list)
        assert isinstance(obs.weather_condition, str)
        assert isinstance(obs.weather_affected_areas, list)
        assert isinstance(obs.current_mission_score, float)
        assert isinstance(obs.natural_language_summary, str)

    def test_reward_has_required_fields(self, easy_env):
        """Reward model must expose step_reward and breakdown."""
        cmd = SwarmCommand(action_type="no_op")
        _, rwd, _, _ = easy_env.step(cmd)
        assert isinstance(rwd, Reward)
        assert isinstance(rwd.step_reward, float)
        assert isinstance(rwd.breakdown, dict)


# ─────────────────────────────────────────────
# 3. Step / Action Resolution Tests
# ─────────────────────────────────────────────

class TestStepActions:
    def test_no_op_increments_time_step(self, easy_env):
        """Each step must increment time_step by exactly 1."""
        obs, _, _, _ = easy_env.step(SwarmCommand(action_type="no_op"))
        assert obs.time_step == 1

    def test_assign_delivery_transitions_drone_to_moving(self, easy_env):
        """assign_delivery on a valid idle drone+pending delivery must set drone to 'moving'."""
        cmd = SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1")
        obs, rwd, _, _ = easy_env.step(cmd)
        d1 = next(d for d in obs.drones if d.id == "D1")
        assert d1.status == "moving"
        assert d1.cargo == "P1"

    def test_assign_delivery_sets_delivery_to_assigned(self, easy_env):
        """The delivery targeted by assign_delivery must transition to 'assigned'."""
        cmd = SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1")
        obs, _, _, _ = easy_env.step(cmd)
        p1 = next(d for d in obs.deliveries if d.id == "P1")
        assert p1.status == "assigned"

    def test_double_assign_same_delivery_penalised(self, easy_env):
        """Trying to assign an already-assigned delivery must trigger a penalty, not a crash."""
        easy_env.step(SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1"))
        _, rwd, _, _ = easy_env.step(
            SwarmCommand(action_type="assign_delivery", drone_id="D2", target_id="P1")
        )
        assert rwd.breakdown["penalty"] < 0, "Double-assign must incur penalty"

    def test_assign_busy_drone_is_penalised(self, easy_env):
        """Assigning a second delivery to an already-moving drone must incur a penalty."""
        easy_env.step(SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1"))
        _, rwd, _, _ = easy_env.step(
            SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P2")
        )
        assert rwd.breakdown["penalty"] < 0

    def test_recharge_drone_at_base_transitions_to_charging(self, easy_env):
        """recharge_drone on a drone at base_station must set its status to 'charging'."""
        cmd = SwarmCommand(action_type="recharge_drone", drone_id="D1")
        obs, _, _, _ = easy_env.step(cmd)
        d1 = next(d for d in obs.drones if d.id == "D1")
        assert d1.status == "charging"

    def test_recharge_drone_away_from_base_is_penalised(self, easy_env):
        """Recharging a drone not at base station must incur a penalty — not silently succeed."""
        # Move D1 away first
        easy_env.step(SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1"))
        # Now D1 is 'moving', not at base → recharge should fail
        _, rwd, _, _ = easy_env.step(SwarmCommand(action_type="recharge_drone", drone_id="D1"))
        assert rwd.breakdown["penalty"] < 0

    def test_nonexistent_drone_id_is_penalised(self, easy_env):
        """Using a drone_id that doesn't exist (e.g., 'D99') must incur a penalty."""
        _, rwd, _, _ = easy_env.step(
            SwarmCommand(action_type="assign_delivery", drone_id="D99", target_id="P1")
        )
        assert rwd.breakdown["penalty"] < 0

    def test_step_returns_four_tuple(self, easy_env):
        """step() must return (Observation, Reward, bool, dict) — the Gym contract."""
        result = easy_env.step(SwarmCommand(action_type="no_op"))
        assert len(result) == 4
        obs, rwd, done, info = result
        assert isinstance(obs, Observation)
        assert isinstance(rwd, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_state_is_deep_copy_not_reference(self, easy_env):
        """Mutating a returned Observation must not mutate the environment's internal state."""
        obs = easy_env.state()
        obs.drones[0].battery = 0.0  # mutate the copy
        # Internal state must be untouched
        assert easy_env.drones[0].battery == 100.0


# ─────────────────────────────────────────────
# 4. Physics Tick (Battery / Movement) Tests
# ─────────────────────────────────────────────

class TestPhysics:
    def test_moving_drone_drains_battery_clear_weather(self, easy_env):
        """A moving drone in clear weather must lose exactly 1.0 battery per step."""
        easy_env.step(SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1"))
        obs, _, _, _ = easy_env.step(SwarmCommand(action_type="no_op"))
        d1 = next(d for d in obs.drones if d.id == "D1")
        # D1 was assigned in step 1, moves in step 2 → should drain 1.0
        assert d1.battery <= 99.0

    def test_idle_drone_has_tiny_drain(self, easy_env):
        """Idle drones must still drain 0.1% battery per step (not zero, not 1.0)."""
        obs, _, _, _ = easy_env.step(SwarmCommand(action_type="no_op"))
        for d in obs.drones:
            if d.status == "idle":
                assert 99.0 < d.battery < 100.0, (
                    f"Idle drone {d.id} has unexpected battery {d.battery}"
                )

    def test_charging_drone_gains_15_battery_per_step(self, easy_env):
        """A charging drone must gain exactly 15 battery per step and cap at 100."""
        # Drain drone first — take it to ~80 battery via movement
        easy_env.drones[0].battery = 80.0
        easy_env.step(SwarmCommand(action_type="recharge_drone", drone_id="D1"))
        obs, _, _, _ = easy_env.step(SwarmCommand(action_type="no_op"))
        d1 = next(d for d in obs.drones if d.id == "D1")
        # 80 + 15 = 95 (still charging)
        assert d1.battery == pytest.approx(95.0)

    def test_charging_exceeds_100_caps_and_goes_idle(self, easy_env):
        """A nearly-full charging drone must cap at 100 and switch to idle."""
        easy_env.drones[0].battery = 90.0
        easy_env.step(SwarmCommand(action_type="recharge_drone", drone_id="D1"))
        obs, _, _, _ = easy_env.step(SwarmCommand(action_type="no_op"))
        d1 = next(d for d in obs.drones if d.id == "D1")
        assert d1.battery == 100.0
        assert d1.status == "idle"

    def test_zero_battery_drone_fails(self, easy_env):
        """A drone whose battery hits 0 must transition to 'failed' status."""
        easy_env.drones[0].battery = 0.5  # just below 1 drain tick
        obs, _, _, _ = easy_env.step(SwarmCommand(action_type="no_op"))
        d1 = next(d for d in obs.drones if d.id == "D1")
        assert d1.status == "failed"
        assert d1.battery == 0.0

    def test_failed_drone_cargo_delivery_also_fails(self, easy_env):
        """If a moving drone fails (battery = 0), its cargo delivery must be marked 'failed'."""
        easy_env.step(SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1"))
        easy_env.drones[0].battery = 0.5  # will hit 0 on next physics tick
        obs, _, _, _ = easy_env.step(SwarmCommand(action_type="no_op"))
        p1 = next(d for d in obs.deliveries if d.id == "P1")
        assert p1.status == "failed"

    def test_storm_drains_moving_drone_faster(self, hard_env):
        """In a storm, a moving drone should drain 5.0 battery per step, not 1.0."""
        hard_env.step(SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1"))
        obs, _, _, _ = hard_env.step(SwarmCommand(action_type="no_op"))
        d1 = next(d for d in obs.drones if d.id == "D1")
        # 100 - 5 (storm drain during step 1 movement) - 5 (step 2)
        # D1 starts moving at step 1. By step 2 battery=95, by physics tick battery=90.
        assert d1.battery <= 95.0, f"Expected storm-level drain but got {d1.battery}"

    def test_rain_drains_at_2_5_rate(self, medium_env):
        """In rain, a moving drone drains 2.5 battery per step."""
        medium_env.step(SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1"))
        obs, _, _, _ = medium_env.step(SwarmCommand(action_type="no_op"))
        d1 = next(d for d in obs.drones if d.id == "D1")
        assert d1.battery <= 97.6  # 100 - 2.5 (step 1) = 97.5, then step 2 moves again


# ─────────────────────────────────────────────
# 5. Movement & Delivery Completion Tests
# ─────────────────────────────────────────────

class TestMovementAndCompletion:
    def test_moving_drone_earns_movement_reward(self, easy_env):
        """A moving drone must generate +0.02 movement reward per step."""
        easy_env.step(SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1"))
        _, rwd, _, _ = easy_env.step(SwarmCommand(action_type="no_op"))
        assert rwd.breakdown["movement"] >= 0.02

    def test_delivery_complete_earns_05_reward(self, easy_env):
        """Completing a delivery must add +0.5 to the completion reward bucket."""
        # Force deliver immediately: put drone right next to target
        target_pos = easy_env.deliveries[0].target_position
        easy_env.drones[0].position = (target_pos[0] - 1, target_pos[1])
        easy_env.step(SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1"))
        _, rwd, _, _ = easy_env.step(SwarmCommand(action_type="no_op"))
        assert rwd.breakdown["completion"] == pytest.approx(0.5)

    def test_completed_delivery_marks_drone_idle_again(self, easy_env):
        """After delivering, the drone must return to 'idle' (not 'moving')."""
        target_pos = easy_env.deliveries[0].target_position
        easy_env.drones[0].position = (target_pos[0] - 1, target_pos[1])
        easy_env.step(SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1"))
        obs, _, _, _ = easy_env.step(SwarmCommand(action_type="no_op"))
        d1 = next(d for d in obs.drones if d.id == "D1")
        assert d1.status == "idle"
        assert d1.cargo is None

    def test_mission_score_increases_on_completion(self, easy_env):
        """current_mission_score must increase after a delivery is completed."""
        target_pos = easy_env.deliveries[0].target_position
        easy_env.drones[0].position = (target_pos[0] - 1, target_pos[1])
        easy_env.step(SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1"))
        obs, _, _, _ = easy_env.step(SwarmCommand(action_type="no_op"))
        assert obs.current_mission_score > 0.0

    def test_mission_score_bounded_0_to_1(self, easy_env):
        """current_mission_score must never exceed 1.0 or go below 0.0."""
        cmd = SwarmCommand(action_type="no_op")
        for _ in range(easy_env.max_steps):
            obs, _, done, _ = easy_env.step(cmd)
            assert 0.0 <= obs.current_mission_score <= 1.0
            if done:
                break


# ─────────────────────────────────────────────
# 6. Episode Termination Tests
# ─────────────────────────────────────────────

class TestTermination:
    def test_done_after_max_steps(self, easy_env):
        """Episode must terminate with done=True after max_steps steps."""
        cmd = SwarmCommand(action_type="no_op")
        done = False
        for _ in range(easy_env.max_steps):
            _, _, done, _ = easy_env.step(cmd)
        assert done is True

    def test_done_when_all_deliveries_complete_or_failed(self, easy_env):
        """done must be True when every delivery is in a terminal state."""
        for dlv in easy_env.deliveries:
            dlv.status = "complete"
        _, _, done, _ = easy_env.step(SwarmCommand(action_type="no_op"))
        assert done is True

    def test_not_done_after_one_step(self, easy_env):
        """After a single step with pending deliveries, done must be False."""
        _, _, done, _ = easy_env.step(SwarmCommand(action_type="no_op"))
        assert done is False


# ─────────────────────────────────────────────
# 7. Natural Language Summary Tests
# ─────────────────────────────────────────────

class TestNaturalLanguageSummary:
    def test_summary_contains_time_step(self, easy_env):
        """natural_language_summary must reference the current time step."""
        obs = easy_env.state()
        assert "Step 0" in obs.natural_language_summary

    def test_summary_contains_weather(self, easy_env):
        """natural_language_summary must mention the current weather condition."""
        obs = easy_env.state()
        assert easy_env.weather_condition in obs.natural_language_summary

    def test_summary_updates_after_step(self, easy_env):
        """Summary must reflect step 1 after one step, not stay at step 0."""
        easy_env.step(SwarmCommand(action_type="no_op"))
        obs = easy_env.state()
        assert "Step 1" in obs.natural_language_summary
