#!/usr/bin/env python3
import time
import sys
import math
import threading

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_ as LowCmdMsg,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    LowCmd_ as LowCmdType,
    LowState_ as LowStateType,
)
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

# ================= CONFIG =================

CONTROL_HZ = 100.0
DT = 1.0 / CONTROL_HZ

KP_ARM = 40.0
KD_ARM = 1.0
KP_WAIST = 150.0
KD_WAIST = 4.5

MAX_ARM_DEG_PER_SEC = 20.0     # adjust if you want slower/faster arms
MAX_WAIST_DEG_PER_SEC = 10.0   # waist more conservative


LOWSTATE_TOPIC = "rt/lowstate"
LOWCMD_TOPIC   = "rt/arm_sdk"

# ================= JOINT INDEX MAP =================

class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11

    # Waist
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll  = 16
    LeftShoulderYaw   = 17
    LeftElbow         = 18
    LeftWristRoll     = 19
    LeftWristPitch    = 20
    LeftWristYaw      = 21

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll  = 23
    RightShoulderYaw   = 24
    RightElbow         = 25
    RightWristRoll     = 26
    RightWristPitch    = 27
    RightWristYaw      = 28

    kNotUsedJoint = 29

# Names for interactive use
JOINT_NAME = {
    G1JointIndex.LeftShoulderPitch: "LSP",
    G1JointIndex.LeftShoulderRoll:  "LSR",
    G1JointIndex.LeftShoulderYaw:   "LSY",
    G1JointIndex.LeftElbow:         "LE",
    G1JointIndex.LeftWristRoll:     "LWR",
    G1JointIndex.LeftWristPitch:    "LWP",
    G1JointIndex.LeftWristYaw:      "LWY",

    G1JointIndex.RightShoulderPitch: "RSP",
    G1JointIndex.RightShoulderRoll:  "RSR",
    G1JointIndex.RightShoulderYaw:   "RSY",
    G1JointIndex.RightElbow:         "RE",
    G1JointIndex.RightWristRoll:     "RWR",
    G1JointIndex.RightWristPitch:    "RWP",
    G1JointIndex.RightWristYaw:      "RWY",

    G1JointIndex.WaistYaw:   "WY",
    G1JointIndex.WaistRoll:  "WR",
    G1JointIndex.WaistPitch: "WP",
}

NAME_TO_JOINT = {v: k for k, v in JOINT_NAME.items()}

# Left->Right mirror with sign symmetry (+1 same, -1 inverse)
BILATERAL_MAP = {
    "LSP": ("RSP", +1),  # Shoulder Pitch: same
    "LSR": ("RSR", -1),  # Shoulder Roll:  inverse
    "LSY": ("RSY", -1),  # Shoulder Yaw:   inverse
    "LE":  ("RE",  +1),  # Elbow:          same
    "LWR": ("RWR", -1),  # Wrist Roll:     inverse
    "LWP": ("RWP", +1),  # Wrist Pitch:    same
    "LWY": ("RWY", -1),  # Wrist Yaw:      inverse
}


ARM_JOINTS = [
    G1JointIndex.LeftShoulderPitch,
    G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,
    G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
    G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]

WAIST_JOINTS = [
    G1JointIndex.WaistYaw,
    G1JointIndex.WaistRoll,
    G1JointIndex.WaistPitch,
]

CONTROLLED_JOINTS = ARM_JOINTS + WAIST_JOINTS

def d2r(deg):
    return deg * math.pi / 180.0

def r2d(rad):
    return rad * 180.0 / math.pi

S0_OFFSETS = {  # home
    G1JointIndex.LeftShoulderPitch:  d2r(0),
    G1JointIndex.LeftShoulderRoll:  d2r(0),
    G1JointIndex.LeftShoulderYaw:  d2r(0),
    G1JointIndex.LeftElbow:  d2r(0),
    G1JointIndex.LeftWristRoll:  d2r(0),
    G1JointIndex.LeftWristPitch:  d2r(0),
    G1JointIndex.LeftWristYaw:  d2r(0),
    G1JointIndex.RightShoulderPitch:  d2r(0),
    G1JointIndex.RightShoulderRoll:  -d2r(0),
    G1JointIndex.RightShoulderYaw:  d2r(0),
    G1JointIndex.RightElbow:  d2r(0),
    G1JointIndex.RightWristRoll:  d2r(0),
    G1JointIndex.RightWristPitch:  d2r(0),
    G1JointIndex.RightWristYaw:  d2r(0),
}

S1_OFFSETS = {  # flare
    G1JointIndex.LeftShoulderRoll:  d2r(40),
    G1JointIndex.RightShoulderRoll: -d2r(40),
    G1JointIndex.LeftElbow:        -d2r(20),
    G1JointIndex.RightElbow:       -d2r(20),
    G1JointIndex.LeftShoulderYaw:   d2r(20),
    G1JointIndex.RightShoulderYaw: -d2r(20),
}

S1B_OFFSETS = {  # soft release
    G1JointIndex.LeftShoulderRoll:  d2r(10),
    G1JointIndex.RightShoulderRoll: -d2r(10),
    G1JointIndex.LeftShoulderYaw:   d2r(10),
    G1JointIndex.RightShoulderYaw: -d2r(10),
}

S1C_OFFSETS = {  # hands clear
    G1JointIndex.LeftElbow:         -d2r(100),
    G1JointIndex.RightElbow:        -d2r(100),
    G1JointIndex.LeftShoulderRoll:  d2r(20),
    G1JointIndex.RightShoulderRoll: -d2r(20),
    G1JointIndex.LeftShoulderPitch: d2r(50),
    G1JointIndex.RightShoulderPitch: d2r(50),
    G1JointIndex.LeftShoulderYaw:   d2r(20),
    G1JointIndex.RightShoulderYaw:   -d2r(20),
    G1JointIndex.LeftWristPitch:   -d2r(30),
    G1JointIndex.RightWristPitch:   -d2r(30),
    G1JointIndex.LeftWristYaw:    d2r(10),
    G1JointIndex.RightWristYaw:   -d2r(10),
}

S1C_OFFSETS = {  # hands fully clear
    G1JointIndex.LeftElbow:         -d2r(100),
    G1JointIndex.RightElbow:        -d2r(100),
    G1JointIndex.LeftShoulderRoll:  d2r(20),
    G1JointIndex.RightShoulderRoll: -d2r(20),
    G1JointIndex.LeftShoulderPitch: d2r(50),
    G1JointIndex.RightShoulderPitch: d2r(50),
    G1JointIndex.LeftShoulderYaw:   d2r(20),
    G1JointIndex.RightShoulderYaw:   -d2r(20),
    G1JointIndex.LeftWristPitch:   -d2r(30),
    G1JointIndex.RightWristPitch:   -d2r(30),
    G1JointIndex.LeftWristYaw:    d2r(10),
    G1JointIndex.RightWristYaw:   -d2r(10),
}


S2A_OFFSETS = {  # PRE grab
    G1JointIndex.LeftElbow:         -d2r(10),
    G1JointIndex.RightElbow:        -d2r(10),
    G1JointIndex.LeftShoulderRoll:  d2r(20),
    G1JointIndex.RightShoulderRoll: -d2r(20),
    G1JointIndex.LeftShoulderPitch: -d2r(40),
    G1JointIndex.RightShoulderPitch:-d2r(40),
    G1JointIndex.LeftShoulderYaw:   d2r(40),
    G1JointIndex.RightShoulderYaw:   -d2r(40),
    G1JointIndex.LeftWristPitch:   -d2r(30),
    G1JointIndex.RightWristPitch:   -d2r(30),
}


S2_OFFSETS = {  #  grab / clasp
    G1JointIndex.LeftElbow:         d2r(0),
    G1JointIndex.RightElbow:        d2r(0),
    G1JointIndex.LeftShoulderRoll:  -d2r(0),
    G1JointIndex.RightShoulderRoll: d2r(0),
    G1JointIndex.LeftShoulderPitch: -d2r(55),
    G1JointIndex.RightShoulderPitch:-d2r(55),
    G1JointIndex.LeftShoulderYaw:   -d2r(45),
    G1JointIndex.RightShoulderYaw:   d2r(45),
    G1JointIndex.LeftWristPitch:   -d2r(55),
    G1JointIndex.RightWristPitch:   -d2r(55),
    G1JointIndex.LeftWristRoll:   -d2r(60),
    G1JointIndex.RightWristRoll:   d2r(60),
    G1JointIndex.LeftWristYaw:    d2r(10),
    G1JointIndex.RightWristYaw:   -d2r(10),
    G1JointIndex.WaistPitch: -d2r(5),
}

S2B_OFFSETS = {  #  post grab = pick up
    G1JointIndex.LeftElbow:         -d2r(70),
    G1JointIndex.RightElbow:        -d2r(70),
    G1JointIndex.LeftShoulderRoll:  d2r(5),
    G1JointIndex.RightShoulderRoll: -d2r(5),
    G1JointIndex.LeftShoulderPitch: -d2r(20),
    G1JointIndex.RightShoulderPitch:-d2r(20),
    G1JointIndex.LeftShoulderYaw:   -d2r(20),
    G1JointIndex.RightShoulderYaw:   d2r(20),
    G1JointIndex.LeftWristPitch:   -d2r(60),
    G1JointIndex.RightWristPitch:   -d2r(60),
    G1JointIndex.LeftWristRoll:   -d2r(10),
    G1JointIndex.RightWristRoll:   d2r(10),
    G1JointIndex.LeftWristYaw:    d2r(20),
    G1JointIndex.RightWristYaw:   -d2r(20),
    G1JointIndex.WaistPitch: -d2r(15),
}


S2C_OFFSETS = {  #  post drop leave
    G1JointIndex.LeftElbow:         d2r(0),
    G1JointIndex.RightElbow:        d2r(0),
    G1JointIndex.LeftShoulderRoll:  d2r(20),
    G1JointIndex.RightShoulderRoll:  -d2r(20),
    G1JointIndex.LeftShoulderPitch: -d2r(55),
    G1JointIndex.RightShoulderPitch:-d2r(55),
    G1JointIndex.LeftShoulderYaw:   -d2r(45),
    G1JointIndex.RightShoulderYaw:   d2r(45),
    G1JointIndex.LeftWristPitch:   -d2r(45),
    G1JointIndex.RightWristPitch:   -d2r(45),
    G1JointIndex.LeftWristRoll:   -d2r(60),
    G1JointIndex.RightWristRoll:   d2r(60),
    G1JointIndex.LeftWristYaw:    d2r(30),
    G1JointIndex.RightWristYaw:   -d2r(10),
    G1JointIndex.WaistPitch: -d2r(5),
}




S4A_OFFSETS = {  # one hand clear
    G1JointIndex.LeftElbow:         -d2r(0),
    G1JointIndex.RightElbow:        -d2r(100),
    G1JointIndex.LeftShoulderRoll:  d2r(20),
    G1JointIndex.RightShoulderRoll: -d2r(20),
    G1JointIndex.LeftShoulderPitch: -d2r(55),
    G1JointIndex.RightShoulderPitch: d2r(50),
    G1JointIndex.LeftShoulderYaw:   -d2r(45),
    G1JointIndex.RightShoulderYaw:   -d2r(20),
    G1JointIndex.LeftWristPitch:   -d2r(45),
    G1JointIndex.RightWristPitch:   -d2r(30),
    G1JointIndex.LeftWristRoll:   -d2r(60),
    G1JointIndex.LeftWristYaw:    d2r(30),
    G1JointIndex.RightWristYaw:   -d2r(10),
    G1JointIndex.WaistPitch: -d2r(5),
}



S4B_OFFSETS = {  # one hand push
    G1JointIndex.LeftElbow:         -d2r(0),
    G1JointIndex.RightElbow:        -d2r(100),
    G1JointIndex.LeftShoulderRoll:  -d2r(20),
    G1JointIndex.RightShoulderRoll: -d2r(20),
    G1JointIndex.LeftShoulderPitch: -d2r(55),
    G1JointIndex.RightShoulderPitch: d2r(50),
    G1JointIndex.LeftShoulderYaw:   -d2r(45),
    G1JointIndex.RightShoulderYaw:   -d2r(20),
    G1JointIndex.LeftWristPitch:   -d2r(35),
    G1JointIndex.RightWristPitch:   -d2r(30),
    G1JointIndex.LeftWristRoll:   -d2r(60),
    G1JointIndex.LeftWristYaw:    d2r(30),
    G1JointIndex.RightWristYaw:   -d2r(10),
    G1JointIndex.WaistPitch: -d2r(5),
}




S4C_OFFSETS = {  #  grab / clasp
    G1JointIndex.LeftElbow:         -d2r(0),
    G1JointIndex.RightElbow:        -d2r(100),
    G1JointIndex.LeftShoulderRoll:  d2r(20),
    G1JointIndex.RightShoulderRoll: -d2r(20),
    G1JointIndex.LeftShoulderPitch: -d2r(55),
    G1JointIndex.RightShoulderPitch: d2r(50),
    G1JointIndex.LeftShoulderYaw:   -d2r(45),
    G1JointIndex.RightShoulderYaw:   -d2r(20),
    G1JointIndex.LeftWristPitch:   -d2r(45),
    G1JointIndex.RightWristPitch:   -d2r(30),
    G1JointIndex.LeftWristRoll:   -d2r(60),
    G1JointIndex.LeftWristYaw:    -d2r(40),
    G1JointIndex.RightWristYaw:   -d2r(10),
    G1JointIndex.WaistPitch: -d2r(5),
}

S4D_OFFSETS = {  #  grab / clasp
    G1JointIndex.LeftElbow:         -d2r(0),
    G1JointIndex.RightElbow:        -d2r(100),
    G1JointIndex.LeftShoulderRoll:  -d2r(20),
    G1JointIndex.RightShoulderRoll: -d2r(20),
    G1JointIndex.LeftShoulderPitch: -d2r(55),
    G1JointIndex.RightShoulderPitch: d2r(50),
    G1JointIndex.LeftShoulderYaw:   -d2r(45),
    G1JointIndex.RightShoulderYaw:   -d2r(20),
    G1JointIndex.LeftWristPitch:   -d2r(35),
    G1JointIndex.RightWristPitch:   -d2r(30),
    G1JointIndex.LeftWristRoll:   -d2r(60),
    G1JointIndex.LeftWristYaw:    -d2r(10),
    G1JointIndex.RightWristYaw:   -d2r(10),
    G1JointIndex.WaistPitch: -d2r(5),
}


# S2D_OFFSETS = {  #  post drop-leave - retract
#     G1JointIndex.LeftElbow:         d2r(0),
#     G1JointIndex.RightElbow:        d2r(0),
#     G1JointIndex.LeftShoulderRoll:  -d2r(20),
#     G1JointIndex.RightShoulderRoll: d2r(20),
#     G1JointIndex.LeftShoulderPitch: -d2r(55),
#     G1JointIndex.RightShoulderPitch:-d2r(55),
#     G1JointIndex.LeftShoulderYaw:   -d2r(45),
#     G1JointIndex.RightShoulderYaw:   d2r(45),
#     G1JointIndex.LeftWristPitch:   -d2r(45),
#     G1JointIndex.RightWristPitch:   -d2r(45),
#     G1JointIndex.LeftWristRoll:   -d2r(60),
#     G1JointIndex.RightWristRoll:   d2r(60),
#     G1JointIndex.LeftWristYaw:    d2r(10),
#     G1JointIndex.RightWristYaw:   -d2r(10),
#     G1JointIndex.WaistPitch: -d2r(5),
# }


S3_OFFSETS = {  # bring box closer / present
    G1JointIndex.LeftElbow:        -d2r(30),
    G1JointIndex.RightElbow:       -d2r(30),
    G1JointIndex.LeftShoulderPitch: d2r(5),
    G1JointIndex.RightShoulderPitch:d2r(5),
    G1JointIndex.LeftShoulderYaw:   -d2r(10),
    G1JointIndex.RightShoulderYaw:   d2r(10),
}






OFFSET_PRESETS = {
    "S0": S0_OFFSETS,
    "S1": S1_OFFSETS,
    "S2": S2_OFFSETS,
    "S3": S3_OFFSETS,
    "S1B": S1B_OFFSETS,
    "S1C": S1C_OFFSETS,
    "S2A": S2A_OFFSETS,
    "S2B": S2B_OFFSETS,
    "S2C": S2C_OFFSETS,
    "S4A": S4A_OFFSETS,
    "S4B": S4B_OFFSETS,
    "S4C": S4C_OFFSETS,
    "S4D": S4D_OFFSETS,
    # add more: "HOME": {}, etc.
}

ACTIONS = {
    # name: list of (offset_set_name, hold_time_after_apply)
    "PICK_FRONT": [
        ("S1", 0.0),
        ("S2", 0.0),
    ],
    "PLACE_FRONT": [
        ("S3", 0.0),
        ("S1", 0.0),
        ("HOME", 0.0),  # define HOME: {} in OFFSET_PRESETS if you want
    ],
}

# ================= LIVE TUNER =================

class LiveTuner:
    def __init__(self, gravity_comp=None):
        self.dt = DT
        self.low_cmd = LowCmdMsg()
        self.low_state = None
        self.crc = CRC()
        self.have_state = False

        self.init_q = {}
        self.target_q = {}

        self.lock = threading.Lock()
        self.motion_switcher = MotionSwitcherClient()

        # Optional gravity-compensation feedforward. If None, behavior is
        # identical to before. If set, the control loop reads each
        # controlled joint's measured q every tick, calls
        # gravity_comp.compute(q_dict), and writes the result into the
        # tau field of each cmd.motor_cmd entry. Joints not present in
        # gravity_comp.joint_indices() keep tau=0 (e.g. waist excluded
        # by default).
        self.gravity_comp = gravity_comp

    def low_state_cb(self, msg):
        with self.lock:
            self.low_state = msg
            self.have_state = True

    def init_dds(self):
        self.pub = ChannelPublisher(LOWCMD_TOPIC, LowCmdType)
        self.pub.Init()

        self.sub = ChannelSubscriber(LOWSTATE_TOPIC, LowStateType)
        self.sub.Init(self.low_state_cb, 10)

        self.motion_switcher.Init()
        # Optionally: set motion mode here if needed.

    def start(self):
        print("[TUNER] Waiting for first lowstate...")
        while not self.have_state:
            time.sleep(0.01)

        with self.lock:
            st = self.low_state
            for j in CONTROLLED_JOINTS:
                q = float(st.motor_state[j].q)
                self.init_q[j] = q
                self.target_q[j] = q

        print("[TUNER] Captured initial pose for arms+waist.")

        self.thread = RecurrentThread(
            interval=self.dt,
            target=self.control_loop,
            name="g1_arm_live_tuner"
        )
        self.thread.Start()
        print("[TUNER] Control loop started. arm_sdk enabled, safe PD on controlled joints.")

    def control_loop(self):
        with self.lock:
            st = self.low_state
            if st is None:
                return

            cmd = self.low_cmd

            # Mirror all joints as baseline
            n = len(st.motor_state)
            for i in range(n):
                cmd.motor_cmd[i].q   = st.motor_state[i].q
                cmd.motor_cmd[i].dq  = st.motor_state[i].dq
                cmd.motor_cmd[i].tau = 0.0
                cmd.motor_cmd[i].kp  = 0.0
                cmd.motor_cmd[i].kd  = 0.0

            # Enable arm_sdk and override CONTROLLED_JOINTS
            cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0

            for j in CONTROLLED_JOINTS:
                tq = self.target_q.get(j, self.init_q.get(j, st.motor_state[j].q))
                cmd.motor_cmd[j].q   = float(tq)
                cmd.motor_cmd[j].dq  = 0.0
                cmd.motor_cmd[j].tau = 0.0

                if j in WAIST_JOINTS:
                    cmd.motor_cmd[j].kp = KP_WAIST
                    cmd.motor_cmd[j].kd = KD_WAIST
                else:
                    cmd.motor_cmd[j].kp = KP_ARM
                    cmd.motor_cmd[j].kd = KD_ARM

            # Gravity-compensation feedforward. Overwrites the tau=0
            # baseline for any joint present in self.gravity_comp's
            # allow-set. When gravity_comp.scale == 0 (the default until
            # you ramp it up) this is a no-op even when enabled, since
            # the computed values are all 0. Reads each joint's *measured*
            # q (st.motor_state[j].q), not the commanded target — gravity
            # depends on actual configuration, not desired.
            if self.gravity_comp is not None:
                q_now = {}
                for j in CONTROLLED_JOINTS:
                    q_now[j] = float(st.motor_state[j].q)
                tau_ff = self.gravity_comp.compute(q_now)
                for j, tau in tau_ff.items():
                    cmd.motor_cmd[j].tau = tau

            cmd.crc = self.crc.Crc(cmd)

        self.pub.Write(cmd)

    # ---------- Interactive helpers ----------
    def run_action(self, name: str):
        key = name.upper()
        if key not in ACTIONS:
            print(f"[TUNER] Unknown action '{name}'. Known:", ", ".join(ACTIONS.keys()))
            return
        print(f"[TUNER] Running action '{key}'")
        for (offset_name, wait_t) in ACTIONS[key]:
            self.apply_offset_set(offset_name)
            if wait_t > 0:
                time.sleep(wait_t)
        print(f"[TUNER] Action '{key}' complete")

    def apply_offset_set(self, name: str):
        """Apply a named offset preset (e.g. S1, S2) smoothly to all controlled joints simultaneously."""
        key = name.upper()
        if key not in OFFSET_PRESETS:
            print(f"[TUNER] Unknown offset set '{name}'. Known:", ", ".join(OFFSET_PRESETS.keys()))
            return

        offsets = OFFSET_PRESETS[key]
        print(f"[TUNER] Applying offset set '{key}' (simultaneous move)")

        with self.lock:
            if self.low_state is None:
                print("[TUNER] No state yet.")
                return

            # Compute start + target offsets (deg) for each controlled joint
            start_deg = {}
            target_deg = {}
            max_steps = 1

            for j in CONTROLLED_JOINTS:
                base = self.init_q[j]
                cur = self.target_q.get(j, base)
                cur_off_deg = (cur - base) * 180.0 / math.pi

                off_rad = offsets.get(j, 0.0)          # preset offset (rad), 0 if not set
                tgt_off_deg = off_rad * 180.0 / math.pi

                start_deg[j] = cur_off_deg
                target_deg[j] = tgt_off_deg

                delta = abs(tgt_off_deg - cur_off_deg)
                if j in WAIST_JOINTS:
                    max_rate = MAX_WAIST_DEG_PER_SEC
                else:
                    max_rate = MAX_ARM_DEG_PER_SEC

                if max_rate > 0.0:
                    steps = int(math.ceil(delta / (max_rate * self.dt))) if delta > 1e-6 else 1
                    if steps > max_steps:
                        max_steps = steps

        # Interpolate all joints together
        max_steps = max(1, max_steps)
        for k in range(max_steps):
            s = float(k + 1) / float(max_steps)
            with self.lock:
                for j in CONTROLLED_JOINTS:
                    base = self.init_q[j]
                    sd = start_deg[j]
                    td = target_deg[j]
                    new_off_deg = sd + (td - sd) * s
                    self.target_q[j] = base + d2r(new_off_deg)
            time.sleep(self.dt)

        print(f"[TUNER] Finished applying '{key}'")

    def _slew_pair_to_offsets_deg(self, lj, rj, l_target_deg, r_target_deg):
        """Slew L and R joints simultaneously to target offsets (deg from init), speed-limited."""
        with self.lock:
            lb = self.init_q[lj]
            rb = self.init_q[rj]
            lcur = self.target_q.get(lj, lb)
            rcur = self.target_q.get(rj, rb)
            lcur_deg = (lcur - lb) * 180.0 / math.pi
            rcur_deg = (rcur - rb) * 180.0 / math.pi

        # choose max rate by joint type
        def rate(j):
            return MAX_WAIST_DEG_PER_SEC if j in WAIST_JOINTS else MAX_ARM_DEG_PER_SEC

        l_rate = rate(lj)
        r_rate = rate(rj)
        l_delta = abs(l_target_deg - lcur_deg)
        r_delta = abs(r_target_deg - rcur_deg)
        l_steps = int(math.ceil(l_delta / (l_rate * self.dt))) if l_delta > 1e-6 else 1
        r_steps = int(math.ceil(r_delta / (r_rate * self.dt))) if r_delta > 1e-6 else 1
        steps = max(1, l_steps, r_steps)

        for k in range(steps):
            s = float(k + 1) / float(steps)
            with self.lock:
                new_l_deg = lcur_deg + (l_target_deg - lcur_deg) * s
                new_r_deg = rcur_deg + (r_target_deg - rcur_deg) * s
                self.target_q[lj] = lb + d2r(new_l_deg)
                self.target_q[rj] = rb + d2r(new_r_deg)
            time.sleep(self.dt)

        # final snap to exact targets
        with self.lock:
            self.target_q[lj] = lb + d2r(l_target_deg)
            self.target_q[rj] = rb + d2r(r_target_deg)

    def bset_abs_offset_deg(self, left_name, deg):
        """Set both joints from a LEFT joint name to offsets (deg) with proper symmetry."""
        key = left_name.upper()
        if key not in BILATERAL_MAP:
            print("[TUNER] Unknown left joint for bilateral:", left_name)
            print("       Choose one of:", ", ".join(sorted(BILATERAL_MAP.keys())))
            return
        r_name, sign = BILATERAL_MAP[key]
        lj = NAME_TO_JOINT.get(key)
        rj = NAME_TO_JOINT.get(r_name)
        if lj is None or rj is None:
            print("[TUNER] Mapping error for", key, "->", r_name)
            return
        l_deg = float(deg)
        r_deg = sign * l_deg
        print(f"[TUNER] Bilateral set: {key}={l_deg} deg, {r_name}={r_deg} deg (offsets from init)")
        self._slew_pair_to_offsets_deg(lj, rj, l_deg, r_deg)

    def brel_offset_deg(self, left_name, ddeg):
        """Increment both joints by ddeg (deg), symmetry applied (relative to current)."""
        key = left_name.upper()
        if key not in BILATERAL_MAP:
            print("[TUNER] Unknown left joint for bilateral:", left_name)
            print("       Choose one of:", ", ".join(sorted(BILATERAL_MAP.keys())))
            return
        r_name, sign = BILATERAL_MAP[key]
        lj = NAME_TO_JOINT.get(key)
        rj = NAME_TO_JOINT.get(r_name)
        if lj is None or rj is None:
            print("[TUNER] Mapping error for", key, "->", r_name)
            return

        with self.lock:
            lb = self.init_q[lj]; rb = self.init_q[rj]
            lcur = self.target_q.get(lj, lb)
            rcur = self.target_q.get(rj, rb)
            lcur_deg = (lcur - lb) * 180.0 / math.pi
            rcur_deg = (rcur - rb) * 180.0 / math.pi

        l_target = lcur_deg + float(ddeg)
        r_target = rcur_deg + sign * float(ddeg)
        print(f"[TUNER] Bilateral rel: {key}+={ddeg} deg -> {key}={l_target} deg, {r_name}={r_target} deg")
        self._slew_pair_to_offsets_deg(lj, rj, l_target, r_target)


    def _slew_joint_to_offset_deg(self, j, target_deg):
        with self.lock:
            base = self.init_q[j]
            cur = self.target_q.get(j, base)
            # current offset (deg) from init
            cur_deg = (cur - base) * 180.0 / math.pi

        delta = target_deg - cur_deg
        if abs(delta) < 1e-3:
            return

        # pick rate based on joint type
        if j in WAIST_JOINTS:
            max_rate = MAX_WAIST_DEG_PER_SEC
        else:
            max_rate = MAX_ARM_DEG_PER_SEC

        max_step = max_rate * self.dt  # deg per control step
        # iterate in small steps towards target
        while True:
            with self.lock:
                base = self.init_q[j]
                cur = self.target_q.get(j, base)
                cur_deg = (cur - base) * 180.0 / math.pi
            delta = target_deg - cur_deg
            if abs(delta) <= max_step:
                # final step
                with self.lock:
                    self.target_q[j] = base + d2r(target_deg)
                break
            step = max_step if delta > 0 else -max_step
            new_deg = cur_deg + step
            with self.lock:
                self.target_q[j] = base + d2r(new_deg)
            time.sleep(self.dt)

    def set_abs_offset_deg(self, joint_name, deg):
        """Set joint to init_q + deg with limited speed."""
        name = joint_name.upper()
        if name not in NAME_TO_JOINT:
            print("[TUNER] Unknown joint name:", joint_name)
            return
        j = NAME_TO_JOINT[name]
        print("[TUNER] Slew", name, "to", deg, "deg offset from init.")
        self._slew_joint_to_offset_deg(j, deg)

    def add_rel_offset_deg(self, joint_name, ddeg):
        """Increment joint offset by ddeg degrees with limited speed."""
        name = joint_name.upper()
        if name not in NAME_TO_JOINT:
            print("[TUNER] Unknown joint name:", joint_name)
            return
        j = NAME_TO_JOINT[name]

        with self.lock:
            base = self.init_q[j]
            cur = self.target_q.get(j, base)
            cur_deg = (cur - base) * 180.0 / math.pi

        target_deg = cur_deg + ddeg
        print("[TUNER] Slew", name, "by", ddeg, "deg to", target_deg, "deg offset.")
        self._slew_joint_to_offset_deg(j, target_deg)


    def home(self):
        with self.lock:
            for j in CONTROLLED_JOINTS:
                self.target_q[j] = self.init_q[j]
        print("[TUNER] All controlled joints set back to init pose.")

    def dump_offsets(self):
        with self.lock:
            st = self.low_state
            if st is None:
                print("[TUNER] No state.")
                return
            print("====== CURRENT OFFSETS (deg from init) ======")
            for j in CONTROLLED_JOINTS:
                name = JOINT_NAME.get(j, str(j))
                q = float(st.motor_state[j].q)
                q0 = float(self.init_q[j])
                off = r2d(q - q0)
                print(f"{name:3s}: {off:7.2f} deg")
            print("=============================================")


# ================= MAIN (INTERACTIVE LOOP) =================

def main():
    print("WARNING: Use this only with the robot in a safe stance.")
    print("This script will take over arms + waist via arm_sdk.")
    raw_input = input  # for Python 3

    ChannelFactoryInitialize(0)

    tuner = LiveTuner()
    tuner.init_dds()
    tuner.start()

    print("\nInteractive commands:")
    print("  set NAME deg    -> set joint to init + deg (e.g. set LSP -40)")
    print("  rel NAME ddeg   -> add relative offset (e.g. rel LE 5)")
    print("  bset LJOINT deg  -> bilateral set (e.g. bset LSY -45)")
    print("    brel LJOINT ddeg -> bilateral relative (e.g. brel LWR 5)")
    print("  offset NAME     -> apply named offset set (e.g. offset S1)")
    print("  action NAME     -> run sequence using offset sets (e.g. action PICK_FRONT)")
    print("  dump            -> print current offsets (copy into S*_OFFSETS)")
    print("  home            -> return all controlled joints to init")
    print("  list            -> show available joint names")
    print("  quit            -> exit (releases control)")
    print("NOTE: All angles are OFFSETS FROM INIT (what you type == what goes into offsets).\n")

    try:
        while True:
            line = raw_input("> ").strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()

            if cmd == "quit" or cmd == "exit":
                break
            elif cmd == "list":
                print("Joints:", " ".join(sorted(NAME_TO_JOINT.keys())))
            elif cmd == "home":
                tuner.home()
            elif cmd == "dump":
                tuner.dump_offsets()
            elif cmd == "set" and len(parts) == 3:
                name = parts[1]
                try:
                    deg = float(parts[2])
                except ValueError:
                    print("[TUNER] Bad angle.")
                    continue
                tuner.set_abs_offset_deg(name, deg)
            elif cmd == "rel" and len(parts) == 3:
                name = parts[1]
                try:
                    ddeg = float(parts[2])
                except ValueError:
                    print("[TUNER] Bad angle.")
                    continue
                tuner.add_rel_offset_deg(name, ddeg)
            elif cmd == "bset" and len(parts) == 3:
                tuner.bset_abs_offset_deg(parts[1], float(parts[2]))
            elif cmd == "brel" and len(parts) == 3:
                tuner.brel_offset_deg(parts[1], float(parts[2]))
            elif cmd == "offset" and len(parts) == 2:
                tuner.apply_offset_set(parts[1])

            elif cmd == "action" and len(parts) == 2:
                tuner.run_action(parts[1])
            else:
                print("[TUNER] Unknown command.")
    except KeyboardInterrupt:
        pass
    finally:
        print("[TUNER] Exiting, disabling arm_sdk.")
        # Best-effort release: set flag 0 once
        with tuner.lock:
            tuner.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 0.0
            tuner.low_cmd.crc = tuner.crc.Crc(tuner.low_cmd)
            tuner.pub.Write(tuner.low_cmd)
        time.sleep(0.1)
        sys.exit(0)


if __name__ == "__main__":
    main()