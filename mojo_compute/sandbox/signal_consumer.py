"""WebSocket client that consumes real-time trading signals from intraday-engine.

Connects to the intraday-engine WebSocket at ws://localhost:6007/ws and routes
incoming signal messages to registered async callback handlers. Supports
auto-reconnect with exponential backoff and keep-alive pings.

Message types handled:
    - NEW_SIGNAL:    A new trading signal has been generated.
    - BATCH_SIGNALS: Multiple signals delivered at once.
    - SIGNAL_CLOSED: A signal has been closed (target/stoploss hit).
    - SIGNAL_UPDATE: An existing signal has been updated.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidStatusCode,
)

from . import config

logger = logging.getLogger(__name__)

# Type alias for async callback handlers
SignalCallback = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]
SignalClosedCallback = Callable[[Dict[str, Any], str], Coroutine[Any, Any, None]]


class SignalConsumer:
    """Async WebSocket client that consumes trading signals from intraday-engine.

    Usage::

        consumer = SignalConsumer()

        async def handle_new(signal):
            print(f"New signal: {signal['symbol']}")

        async def handle_closed(signal, result):
            print(f"Signal closed: {signal['symbol']} -> {result}")

        async def handle_update(signal):
            print(f"Signal updated: {signal['symbol']}")

        consumer.set_handlers(handle_new, handle_closed, handle_update)
        await consumer.start()  # runs forever with auto-reconnect
    """

    def __init__(self) -> None:
        self.connected: bool = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running: bool = False
        self._reconnect_delay: float = config.WS_RECONNECT_DELAY
        self._ping_task: Optional[asyncio.Task] = None
        self._consume_task: Optional[asyncio.Task] = None

        # Callback handlers (async functions)
        self._on_new_signal: Optional[SignalCallback] = None
        self._on_signal_closed: Optional[SignalClosedCallback] = None
        self._on_signal_update: Optional[SignalCallback] = None

    def set_handlers(
        self,
        on_new_signal: SignalCallback,
        on_signal_closed: SignalClosedCallback,
        on_signal_update: SignalCallback,
    ) -> None:
        """Set callback handlers for signal events.

        Args:
            on_new_signal: Called with (signal_dict) on NEW_SIGNAL and each
                           signal in BATCH_SIGNALS.
            on_signal_closed: Called with (signal_dict, result_str) on
                              SIGNAL_CLOSED. Result is "TARGET_HIT" or
                              "STOPLOSS_HIT".
            on_signal_update: Called with (signal_dict) on SIGNAL_UPDATE.
        """
        self._on_new_signal = on_new_signal
        self._on_signal_closed = on_signal_closed
        self._on_signal_update = on_signal_update

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start consuming signals. Runs forever with auto-reconnect.

        This coroutine blocks indefinitely. Call :meth:`stop` from another
        task or a signal handler to shut it down gracefully.
        """
        self._running = True
        logger.info(
            "SignalConsumer starting — will connect to %s", config.INTRADAY_WS_URL
        )

        while self._running:
            try:
                await self._connect_and_consume()
            except asyncio.CancelledError:
                logger.info("SignalConsumer cancelled")
                break
            except Exception as exc:
                logger.error(
                    "SignalConsumer unexpected error: %s", exc, exc_info=True
                )

            if not self._running:
                break

            # Exponential backoff before reconnect
            logger.info(
                "Reconnecting in %.1f seconds...", self._reconnect_delay
            )
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * 2, config.WS_MAX_RECONNECT_DELAY
            )

        logger.info("SignalConsumer stopped")

    async def stop(self) -> None:
        """Stop the consumer and close the WebSocket connection."""
        logger.info("SignalConsumer stopping...")
        self._running = False

        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()

        self.connected = False
        logger.info("SignalConsumer stopped cleanly")

    # ------------------------------------------------------------------
    # Internal connection logic
    # ------------------------------------------------------------------

    async def _connect_and_consume(self) -> None:
        """Establish a WebSocket connection and consume messages until disconnect."""
        try:
            async with websockets.connect(
                config.INTRADAY_WS_URL,
                ping_interval=None,  # we handle pings ourselves
                ping_timeout=None,
                close_timeout=5,
            ) as ws:
                self._ws = ws
                self.connected = True
                self._reconnect_delay = config.WS_RECONNECT_DELAY  # reset backoff
                logger.info(
                    "Connected to intraday-engine at %s", config.INTRADAY_WS_URL
                )

                # Start keep-alive ping task
                self._ping_task = asyncio.create_task(self._keep_alive(ws))

                try:
                    await self._consume_messages(ws)
                finally:
                    self.connected = False
                    self._ws = None
                    if self._ping_task and not self._ping_task.done():
                        self._ping_task.cancel()
                        try:
                            await self._ping_task
                        except asyncio.CancelledError:
                            pass

        except ConnectionRefusedError:
            logger.warning(
                "Connection refused — intraday-engine not running at %s",
                config.INTRADAY_WS_URL,
            )
        except InvalidStatusCode as exc:
            logger.warning(
                "WebSocket handshake failed (HTTP %s) at %s",
                exc.status_code,
                config.INTRADAY_WS_URL,
            )
        except OSError as exc:
            logger.warning("Network error connecting to intraday-engine: %s", exc)
        except ConnectionClosedOK:
            logger.info("WebSocket closed normally by server")
        except ConnectionClosedError as exc:
            logger.warning("WebSocket closed with error: code=%s reason=%s", exc.code, exc.reason)

    async def _consume_messages(
        self, ws: websockets.WebSocketClientProtocol
    ) -> None:
        """Read and dispatch messages from the WebSocket."""
        async for raw_message in ws:
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.warning("Received non-JSON message, ignoring: %.200s", raw_message)
                continue

            msg_type = message.get("type")
            if not msg_type:
                logger.debug("Message without type field, ignoring: %.200s", raw_message)
                continue

            try:
                await self._dispatch(msg_type, message)
            except Exception as exc:
                logger.error(
                    "Error handling %s message: %s", msg_type, exc, exc_info=True
                )

    async def _dispatch(self, msg_type: str, message: Dict[str, Any]) -> None:
        """Route a parsed message to the appropriate callback handler."""
        if msg_type == "NEW_SIGNAL":
            signal = message.get("signal", {})
            logger.info(
                "NEW_SIGNAL received: %s %s @ %.2f (confidence: %.2f)",
                signal.get("signal_type", "?"),
                signal.get("symbol", "?"),
                signal.get("entry_price", 0),
                signal.get("confidence", 0),
            )
            if self._on_new_signal:
                await self._on_new_signal(signal)

        elif msg_type == "BATCH_SIGNALS":
            signals: List[Dict[str, Any]] = message.get("signals", [])
            count = message.get("count", len(signals))
            logger.info("BATCH_SIGNALS received: %d signals", count)
            if self._on_new_signal:
                for signal in signals:
                    try:
                        await self._on_new_signal(signal)
                    except Exception as exc:
                        logger.error(
                            "Error processing batch signal %s: %s",
                            signal.get("symbol", "?"),
                            exc,
                            exc_info=True,
                        )

        elif msg_type == "SIGNAL_CLOSED":
            signal = message.get("signal", {})
            result = message.get("result", "UNKNOWN")
            logger.info(
                "SIGNAL_CLOSED: %s %s -> %s",
                signal.get("symbol", "?"),
                signal.get("signal_type", "?"),
                result,
            )
            if self._on_signal_closed:
                await self._on_signal_closed(signal, result)

        elif msg_type == "SIGNAL_UPDATE":
            signal = message.get("signal", {})
            logger.info(
                "SIGNAL_UPDATE: %s %s (price: %.2f)",
                signal.get("symbol", "?"),
                signal.get("signal_type", "?"),
                signal.get("current_price", 0),
            )
            if self._on_signal_update:
                await self._on_signal_update(signal)

        else:
            logger.debug("Unhandled message type: %s", msg_type)

    # ------------------------------------------------------------------
    # Keep-alive
    # ------------------------------------------------------------------

    async def _keep_alive(
        self, ws: websockets.WebSocketClientProtocol
    ) -> None:
        """Send periodic ping messages to keep the connection alive."""
        try:
            while True:
                await asyncio.sleep(config.WS_PING_INTERVAL)
                try:
                    await ws.send(json.dumps({"type": "ping"}))
                    logger.debug("Keep-alive ping sent")
                except ConnectionClosed:
                    logger.debug("Connection closed during ping, exiting keep-alive")
                    break
        except asyncio.CancelledError:
            pass
