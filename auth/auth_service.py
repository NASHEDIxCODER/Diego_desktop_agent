"""
FaceAuthService — Mandatory face authentication with live popup + voice.

Redesigned authentication:
  * Small floating preview window showing camera feed, face rectangle,
    confidence, user name, lighting indicator, distance indicator,
    head alignment.
  * If no face: shows "No face detected" and keeps waiting — never exits,
    never times out, never closes Leo.
  * Voice prompts: "I can't see you yet", "Move a little closer",
    "Face detected", "Authentication successful".
  * Once authenticated: hides popup, continues normally.

Runs as a service so the supervisor can restart it if the camera fails.
"""

import asyncio
import logging
import threading
import time
from typing import Optional

from core.event_bus import bus
from core.service import BaseService
from core.metrics import metrics

logger = logging.getLogger(__name__)


class FaceAuthService(BaseService):
    """
    Face authentication service. Provides:
      * authenticate() — blocks until a face is verified (never times out)
      * continuous camera health monitoring
      * voice prompts state (driven by the conversation engine)
    """

    name = "face_auth"

    def __init__(self):
        super().__init__()
        self._auth_user: Optional[str] = None
        self._last_auth_time: float = 0.0
        self._auth_session_s: float = 600.0  # 10 minutes
        self._enabled = True
        self._popup_open = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def _start(self) -> bool:
        """Initialize the auth service (no camera open at boot)."""
        # Start the camera health monitor (self-healing: camera failure
        # emits watchdog.alert → supervisor restarts the auth service).
        try:
            loop = asyncio.get_running_loop()
            self._monitor_task = loop.create_task(self.monitor_camera())
        except Exception as e:
            logger.debug("[AUTH-SVC] camera monitor deferred: %s", e)
        self.set_health("face auth ready", {"enabled": self._enabled})
        return True

    async def _stop(self) -> None:
        """Stop the camera monitor and close the camera (idempotent)."""
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except (asyncio.CancelledError, Exception):
                pass
            self._monitor_task = None
        try:
            from auth.faceauth import close as _auth_close
            _auth_close()
        except Exception:
            pass

    # ── Public API ─────────────────────────────────────────

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable face authentication (dev mode --no-auth)."""
        self._enabled = enabled

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def authenticated_user(self) -> Optional[str]:
        return self._auth_user

    def set_authenticated(self, name: Optional[str]) -> None:
        """Mark an externally-verified session (e.g. --no-auth)."""
        self._auth_user = name
        self._last_auth_time = time.time()
        if name:
            metrics.set("face.authenticated", True)
        else:
            metrics.set("face.authenticated", False)

    def invalidate(self) -> None:
        """Force re-authentication on the next wake."""
        self._auth_user = None
        self._last_auth_time = 0.0
        metrics.set("face.authenticated", False)
        logger.info("[AUTH-SVC] Session invalidated")

    def needs_auth(self) -> bool:
        """True if we must authenticate before conversation."""
        if not self._enabled:
            return False
        if self._auth_user is None:
            return True
        return (time.time() - self._last_auth_time) >= self._auth_session_s

    # ── Authentication flow ────────────────────────────────

    async def authenticate(self) -> Optional[str]:
        """
        Run live face authentication with popup + voice prompts.

        Returns the verified user's name, or None if denied/cancelled.
        NEVER raises, NEVER terminates the runtime. Runs the blocking
        auth in an executor (it blocks on the camera).
        """
        if not self._enabled:
            return None
        try:
            from auth.live_auth import authenticate_live

            loop = asyncio.get_running_loop()
            # Voice prompt: "I can't see you yet" plays as the popup
            # opens (the live_auth loop also provides its own prompts).
            await bus.emit("conv.speak", data={
                "text": "Looking for you. Please look at the camera.",
            }, source="auth_service")

            # Run the blocking auth flow in a thread.
            name = await loop.run_in_executor(None, authenticate_live)

            if name:
                self._auth_user = name
                self._last_auth_time = time.time()
                metrics.set("face.authenticated", True)
                metrics.set("face.confidence", 1.0)  # verified via majority
                metrics.set("face.user", name)
                await bus.emit("auth.success", data={"user": name},
                               source="auth_service")
                await bus.emit("conv.speak", data={
                    "text": f"Authentication successful. Welcome back, {name}.",
                }, source="auth_service")
            else:
                metrics.set("face.authenticated", False)
                await bus.emit("auth.denied", data={}, source="auth_service")
                await bus.emit("conv.speak", data={
                    "text": "I couldn't verify your identity. Please try again.",
                }, source="auth_service")

            return name
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[AUTH-SVC] authentication error: %s", e)
            metrics.set("face.authenticated", False)
            await bus.emit("auth.error", data={"error": str(e)},
                           source="auth_service")
            return None

    # ── Camera health monitoring ───────────────────────────

    async def monitor_camera(self) -> None:
        """
        Watchdog: if the camera fails during auth, emit a watchdog.alert
        so the supervisor can restart the auth service.
        """
        from core.service import ServiceState
        try:
            # Runs while the service is active (STARTING→RUNNING→DEGRADED).
            # Exits only on STOPPING/STOPPED (cancelled by _stop()).
            while self._state in (ServiceState.STARTING,
                                  ServiceState.RUNNING,
                                  ServiceState.DEGRADED,
                                  ServiceState.FAILED):
                try:
                    from auth.faceauth import _get_camera
                    cam = _get_camera()
                    if cam is None:
                        raise RuntimeError("camera not available")
                    ok = cam.isOpened()
                    if not ok:
                        raise RuntimeError("camera not open")
                    metrics.set("camera.ok", True)
                except Exception as e:
                    metrics.set("camera.ok", False)
                    logger.warning("[AUTH-SVC] camera issue: %s", e)
                    await bus.emit("watchdog.alert", data={
                        "name": self.name, "reason": "camera failure",
                        "error": str(e),
                    }, source="auth_service")
                await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            pass


# Global singleton
face_auth_service = FaceAuthService()