import struct
import threading
import time
import can

GRIPPER_CMD_ID = 0x159
EFFORT = 0.4  # 0-1, maps to 0-5 Nm; lower = less grip force

# The nominal position=0.0/1.0 (fully closed/open) endpoints land a little short
# of the actual mechanical hard stops, so a "closed"/"open" command leaves a
# visible gap. Command slightly past each endpoint on full-travel requests so
# the jaws drive fully into the hard stop; safe because torque is still capped
# by EFFORT. The two gripper units differ in how short they land, so these are
# tunable per CanGripper instance rather than shared constants.
CLOSE_OVERTRAVEL = 0.10
OPEN_OVERTRAVEL = 0.10


def _make_frame(
    position: float, status_code: int, effort: float = EFFORT, inverted: bool = True,
    close_overtravel: float = CLOSE_OVERTRAVEL, open_overtravel: float = OPEN_OVERTRAVEL,
) -> bytes:
    p = (1.0 - position) if inverted else position
    if position <= 0.0:
        p += close_overtravel if inverted else -close_overtravel
    elif position >= 1.0:
        p += -open_overtravel if inverted else open_overtravel
    pos_ref = int(p * 0.07 * 1e6)              # 0-1 → meters → micrometers
    effort_ref = int(effort * 5000)             # 0-1 → mNm
    return struct.pack(">iH", pos_ref, effort_ref) + bytes([status_code, 0x00])


class CanGripper:
    """200 Hz raw-CAN gripper controller for the Nero integrated gripper.

    nerolib must be configured with gripper_on=False so its loop doesn't
    conflict. This class owns all 0x159 traffic.
    """

    def __init__(
        self, can_port: str, inverted: bool = True,
        close_overtravel: float = CLOSE_OVERTRAVEL, open_overtravel: float = OPEN_OVERTRAVEL,
    ):
        self._can_port = can_port
        self._inverted = inverted
        self._close_overtravel = close_overtravel
        self._open_overtravel = open_overtravel
        self._target: float | None = None
        self._running = False
        self._bus: can.interface.Bus | None = None
        self._thread: threading.Thread | None = None

    def _frame(self, position: float, status_code: int, effort: float = EFFORT) -> bytes:
        return _make_frame(
            position, status_code, effort=effort, inverted=self._inverted,
            close_overtravel=self._close_overtravel, open_overtravel=self._open_overtravel,
        )

    def enable(self):
        """Open the bus, kick with status=3 (enable+clear), start the loop."""
        if self._running:
            return
        self._bus = can.interface.Bus(channel=self._can_port, interface="socketcan")
        # Kick with ENABLE_AND_CLEAR_ERROR so the gripper activates immediately
        self._bus.send(can.Message(
            arbitration_id=GRIPPER_CMD_ID,
            data=self._frame(1.0, status_code=3),
            is_extended_id=False,
        ))
        time.sleep(0.05)
        self._target = 1.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def disable(self):
        """Stop the loop and send status=0 frames to let the gripper go limp."""
        self._target = None
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._bus:
            for _ in range(10):
                try:
                    self._bus.send(can.Message(
                        arbitration_id=GRIPPER_CMD_ID,
                        data=self._frame(0.0, status_code=0, effort=0.0),
                        is_extended_id=False,
                    ))
                except Exception:
                    pass
                time.sleep(0.005)
            self._bus.shutdown()
            self._bus = None

    def set_position(self, pos: float):
        """Set gripper target (0.0 = closed, 1.0 = open)."""
        self._target = max(0.0, min(1.0, pos))

    def get_position(self) -> float:
        return self._target if self._target is not None else 0.0

    def stop(self):
        self.disable()

    def _loop(self):
        while self._running:
            t = self._target
            if t is not None and self._bus:
                try:
                    self._bus.send(can.Message(
                        arbitration_id=GRIPPER_CMD_ID,
                        data=self._frame(t, status_code=1),
                        is_extended_id=False,
                    ))
                except Exception:
                    pass
            time.sleep(0.005)  # 200 Hz
