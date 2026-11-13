
import os
import socket
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import httpx

from UserSession.AIEval.RemoteAISessionEval import RemoteAISessionEval


class ServerManager:
    """Manages GPU server startup and readiness."""
    def __init__(self):
        self.worker_url = os.environ["AI_WORKER_URL"].rstrip("/")
        self.mac_address = os.environ["AI_WORKER_MAC"]
        self.broadcast_ip = os.environ["AI_WORKER_BROADCAST"]

        self.lock = Lock()
        self._ai_eval = None  # IMPORTANT: initialize this attribute

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.startup_future = None

    @property
    def ai_eval(self):
        with self.lock:
            if self._ai_eval is None:
                self._ai_eval = RemoteAISessionEval()
            return self._ai_eval

    def health_check(self):
        """True only when Server B reports AI readiness."""
        try:
            response = httpx.get(
                f"{self.worker_url}/health",
                timeout=2.0
            )

            if response.status_code != 200:
                return False

            return response.json().get("status") == "ready"

        except (httpx.HTTPError, ValueError):
            return False

    def wake_server(self):
        """Power on Server B using Wake-on-LAN."""
        mac = bytes.fromhex(
            self.mac_address.replace(":", "")
        )

        packet = b"\xff" * 6 + mac * 16

        with socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM
        ) as sock:
            sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_BROADCAST,
                1
            )
            sock.sendto(
                packet,
                (self.broadcast_ip, 9)
            )

        logging.info("Sent Wake-on-LAN packet to Server B")

    def ensure_server_ready(self, timeout=120):
        """Wake Server B and wait until inference is ready."""

        # Already running? Nothing to do.
        if self.health_check():
            logging.info("GPU worker already ready")
            return True

        logging.info("GPU worker unavailable; waking Server B")
        self.wake_server()

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self.health_check():
                logging.info("GPU worker is ready")
                return True

            time.sleep(3)

        logging.error("GPU worker startup timed out")
        return False

    def start_in_background(self):
        """Start one readiness check without blocking users."""
        with self.lock:
            if (
                self.startup_future is None
                or self.startup_future.done()
            ):
                self.startup_future = self.executor.submit(
                    self.ensure_server_ready
                )

            return self.startup_future

    def get_status(self):
        """Report the current worker state."""
        if self.health_check():
            return "ready"

        with self.lock:
            if (
                self.startup_future is not None
                and not self.startup_future.done()
            ):
                return "starting"

        return "unavailable"


# Shared across the entire backend process
server_manager = ServerManager()
