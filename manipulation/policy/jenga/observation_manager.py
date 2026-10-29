"""Stateful construction of the jenga actor observation for one control tick.

Episode lifecycle (the session's episode key drives it):
  idle    -> arm_episode()  -> armed
  armed   -> first tick whose trust is not untrusted -> active: the symmetry is picked
             against the training reset orientation, that pose is the first trusted pose,
             and the goal is fixed at its position + target_lift_height along base z
             (PerceivedObjectPose captures the actor goal the same way).
  armed / active -> end_episode() -> idle

The symmetry reference is the last trusted (SAM-verified) corrected pose, or the training
reset orientation before one exists. A re-registration therefore resolves against the last
trusted pose, never against a pose FoundationPose produced while it was failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .. import observations as obs
from ..robot_state import RobotState
from ..symmetry import BOX_SYMMETRY_NAMES, SymmetryChoice, box_symmetries, select_closest
from ..trust import Trust
from .config import JengaPolicyConfig

IDLE, ARMED, ACTIVE = "idle", "armed", "active"


@dataclass
class ObsResult:
    phase: str
    step: Optional[int]                  # 0-based control step within the episode
    trust: Trust
    T_raw: Optional[np.ndarray]          # FP original-mesh pose in the base frame
    choice: Optional[SymmetryChoice]
    reference: Optional[str]             # "training" or "last_trusted"
    goal: Optional[np.ndarray]
    terms: Dict[str, np.ndarray]
    vector: Optional[np.ndarray]
    events: List[dict] = field(default_factory=list)

    @property
    def T_corrected(self) -> Optional[np.ndarray]:
        return None if self.choice is None else self.choice.T


class JengaObservationManager:
    def __init__(self, cfg: JengaPolicyConfig):
        self.cfg = cfg
        self.symmetries = box_symmetries(cfg.box_center)
        self.R_training = cfg.T_base_initial[:3, :3]
        self.phase = IDLE
        self.episode = 0
        self.step = 0
        self.initial_pose: Optional[np.ndarray] = None
        self.goal: Optional[np.ndarray] = None
        self.last_trusted: Optional[np.ndarray] = None
        self.last_trusted_frame: Optional[int] = None

    # ----- episode control ------------------------------------------------------------

    def arm_episode(self) -> dict:
        self.phase = ARMED
        # The new episode resolves the symmetry against training, not the previous run.
        self.last_trusted = self.last_trusted_frame = None
        return {"event": "episode_armed", "episode": self.episode + 1}

    def end_episode(self, reason: str) -> dict:
        event = {"event": "episode_end", "episode": self.episode, "reason": reason,
                 "phase": self.phase, "steps": self.step if self.phase == ACTIVE else 0}
        if self.phase == ARMED:
            event["event"] = "episode_cancelled"
        self.phase = IDLE
        self.initial_pose = self.goal = None
        self.step = 0
        return event

    # ----- per frame and per tick -----------------------------------------------------

    def _reference(self):
        if self.last_trusted is not None:
            return self.last_trusted[:3, :3], "last_trusted"
        return self.R_training, "training"

    def note_verified(self, T_raw: np.ndarray, frame_number: int) -> None:
        """A SAM-verified frame (any frame, tick or not): its corrected pose becomes last trusted."""
        if self.phase == ARMED:
            return  # the episode start picks against the training pose first
        R_ref, _ = self._reference()
        self.last_trusted = select_closest(T_raw, self.symmetries, R_ref).T
        self.last_trusted_frame = frame_number

    def tick(self, robot: Optional[RobotState], T_raw: Optional[np.ndarray], trust: Trust,
             frame_number: int) -> ObsResult:
        events: List[dict] = []
        choice = reference = None
        if T_raw is not None:
            if self.phase == ARMED and trust is not Trust.UNTRUSTED:
                choice = select_closest(T_raw, self.symmetries, self.R_training)
                reference = "training"
                self.episode += 1
                self.phase, self.step = ACTIVE, 0
                self.initial_pose = choice.T
                self.last_trusted, self.last_trusted_frame = choice.T, frame_number
                self.goal = choice.T[:3, 3] + np.array([0.0, 0.0, self.cfg.target_lift_height])
                events.append({"event": "episode_start", "episode": self.episode, "frame_number": frame_number,
                               "trust": trust.value, "symmetry_index": choice.index,
                               "symmetry": BOX_SYMMETRY_NAMES[choice.index],
                               "symmetry_angles_deg": np.degrees(choice.angles_rad).tolist(),
                               "T_base_object_raw": T_raw.tolist(), "T_base_object_initial": choice.T.tolist(),
                               "goal": self.goal.tolist()})
            else:
                R_ref, reference = self._reference()
                choice = select_closest(T_raw, self.symmetries, R_ref)

        carriage_q = carriage_qd = None
        kwargs = {}
        if robot is not None:
            carriage_q = obs.carriages_from_gripper(robot.gripper_q)
            carriage_qd = obs.carriages_from_gripper(robot.gripper_qd)
            kwargs = {"arm_q": robot.arm_q, "arm_qd": robot.arm_qd, "T_base_tcp": robot.T_base_tcp}
        terms = obs.actor_terms(carriage_q=carriage_q, carriage_qd=carriage_qd,
                                T_base_obj=None if choice is None else choice.T,
                                goal_pos=self.goal if self.phase == ACTIVE else None,
                                gripper_open=self.cfg.gripper_open, gripper_closed=self.cfg.gripper_closed,
                                **kwargs)
        vector = obs.build_actor_obs(terms) if self.phase == ACTIVE else None

        step = None
        if self.phase == ACTIVE:
            step = self.step
            if step == self.cfg.episode_steps:
                events.append({"event": "episode_length_exceeded", "episode": self.episode,
                               "episode_steps": self.cfg.episode_steps,
                               "note": "the policy was trained on episodes of this many steps"})
            self.step += 1
        return ObsResult(phase=self.phase, step=step, trust=trust, T_raw=T_raw, choice=choice,
                         reference=reference, goal=None if self.goal is None else self.goal.copy(),
                         terms=terms, vector=vector, events=events)
