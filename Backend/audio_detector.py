"""Simple audio activity detector for proctoring.

Features:
- Captures short audio blocks from the system default microphone using sounddevice.
- Computes RMS (volume) and triggers an alert when volume exceeds a configurable threshold
  for a configurable number of consecutive blocks.
- Alerting is handled by a callback passed to the detector; a CLI mode prints alerts.

Notes / integration:
- This is intentionally minimal and dependency-light. It uses `sounddevice` + `numpy`.
- In your proctoring app you can call `AudioDetector(..., alert_callback=your_function)`
  to send alerts to your websocket or server (see `proctor_websocket.py`).
- Tune `threshold` and `block_duration` to match your environment.
"""
from __future__ import annotations

import time
import threading
import logging
from collections import deque
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioDetector:
    """Detect audio activity from microphone and call an alert callback.

    Contract (short):
    - Input: audio from default device (or given device), sample rate, block duration
    - Output: calls alert_callback(alert_info: dict) when sustained loud audio detected

    Parameters:
    - threshold: RMS amplitude to consider as "audio" (float, 0..1 for float32 audio)
    - sample_rate: sample rate for input (Hz)
    - block_duration: seconds per audio block to analyze
    - consecutive_blocks: number of consecutive blocks above threshold required to trigger
    - alert_callback: callable that receives a dict {"timestamp", "rms", "message"}
    - device: sounddevice device id or None
    """

    def __init__(
        self,
        threshold: float = 0.02,
        sample_rate: int = 16000,
        block_duration: float = 0.5,
        consecutive_blocks: int = 3,
        alert_callback: Optional[Callable[[dict], None]] = None,
        device: Optional[int | str] = None,
        channels: int = 1,
    ) -> None:
        self.threshold = float(threshold)
        self.sample_rate = int(sample_rate)
        self.block_duration = float(block_duration)
        self.consecutive_blocks = max(1, int(consecutive_blocks))
        self.alert_callback = alert_callback
        self.device = device
        self.channels = channels

        # internal
        self._stream: Optional[sd.InputStream] = None
        self._rms_history: deque[float] = deque(maxlen=self.consecutive_blocks)
        self._alerted = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def _rms(self, block: np.ndarray) -> float:
        # block expected shape: (frames, channels)
        if block.size == 0:
            return 0.0
        if block.ndim > 1:
            block = block.mean(axis=1)
        return float(np.sqrt(np.mean(np.square(block), dtype=np.float64)))

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.debug("Input stream status: %s", status)
        rms_val = self._rms(indata)
        with self._lock:
            self._rms_history.append(rms_val)
            # check if we have enough history
            if len(self._rms_history) >= self.consecutive_blocks:
                avg = sum(self._rms_history) / len(self._rms_history)
                logger.debug("RMS history=%s avg=%.6f", list(self._rms_history), avg)
                if avg >= self.threshold:
                    if not self._alerted:
                        self._alerted = True
                        self._trigger_alert(avg)
                else:
                    # reset alert when volume goes down
                    if self._alerted:
                        logger.debug("Volume dropped below threshold; clearing alerted state")
                    self._alerted = False

    def _trigger_alert(self, rms_value: float) -> None:
        info = {
            "timestamp": time.time(),
            "rms": rms_value,
            "message": "Loud audio detected",
        }
        logger.info("Audio alert triggered: rms=%.6f", rms_value)
        if self.alert_callback:
            try:
                self.alert_callback(info)
            except Exception:
                logger.exception("alert_callback raised an exception")
        else:
            # Default behavior: log and print
            print("ALERT: audio activity detected:", info)

    def start(self) -> None:
        """Start the audio detector (non-blocking)."""
        if self._stream is not None and self._stream.active:
            logger.debug("AudioDetector already running")
            return

        self._stop_event.clear()
        blocksize = int(self.sample_rate * self.block_duration)
        logger.info(
            "Starting AudioDetector: sample_rate=%s block_duration=%s blocksize=%s device=%s",
            self.sample_rate,
            self.block_duration,
            blocksize,
            self.device,
        )
        self._rms_history.clear()
        self._alerted = False

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=blocksize,
            device=self.device,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop the detector."""
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Error stopping stream")
        self._stream = None

    def run_blocking(self, duration: Optional[float] = None) -> None:
        """Start detector and block until duration seconds have passed (or KeyboardInterrupt)."""
        try:
            self.start()
            if duration is None:
                # block until stop called or Ctrl+C
                while not self._stop_event.is_set():
                    time.sleep(0.1)
            else:
                end = time.time() + float(duration)
                while time.time() < end and not self._stop_event.is_set():
                    time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()


def _cli_print_alert(info: dict) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.get("timestamp", time.time())))
    print(f"[{ts}] ALERT: {info.get('message')} (rms={info.get('rms'):.6f})")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run a simple audio detector and print alerts.")
    parser.add_argument("--threshold", type=float, default=0.02, help="RMS threshold (0..1)")
    parser.add_argument(
        "--block-duration",
        type=float,
        default=0.5,
        help="Seconds per block to compute RMS (shorter = more responsive)",
    )
    parser.add_argument("--consecutive", type=int, default=3, help="Consecutive blocks to trigger")
    parser.add_argument("--duration", type=float, default=None, help="How long to run (seconds)")
    parser.add_argument("--samplerate", type=int, default=16000, help="Sample rate in Hz")
    args = parser.parse_args()

    detector = AudioDetector(
        threshold=args.threshold,
        sample_rate=args.samplerate,
        block_duration=args.block_duration,
        consecutive_blocks=args.consecutive,
        alert_callback=_cli_print_alert,
    )

    print("Starting audio detector. Speak or make a noise to trigger an alert. Ctrl+C to stop.")
    detector.run_blocking(duration=args.duration)
