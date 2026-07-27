"""
Tests for the async EventBus.
"""

import pytest
from core.event_bus import EventBus, Event


@pytest.mark.asyncio
async def test_emit_and_receive():
    bus = EventBus()
    received = []

    async def handler(event: Event):
        received.append(event)

    bus.on("test_event", handler)
    await bus.emit("test_event", {"key": "value"}, "test")

    assert len(received) == 1
    assert received[0].type == "test_event"
    assert received[0].data == {"key": "value"}
    assert received[0].source == "test"


@pytest.mark.asyncio
async def test_wildcard_handler():
    bus = EventBus()
    received = []

    async def handler(event: Event):
        received.append(event.type)

    bus.on("*", handler)
    await bus.emit("event_a", source="test")
    await bus.emit("event_b", source="test")

    assert received == ["event_a", "event_b"]


@pytest.mark.asyncio
async def test_multiple_handlers():
    bus = EventBus()
    results = []

    async def handler1(event: Event):
        results.append("h1")

    async def handler2(event: Event):
        results.append("h2")

    bus.on("evt", handler1)
    bus.on("evt", handler2)
    await bus.emit("evt", source="test")

    assert "h1" in results
    assert "h2" in results


@pytest.mark.asyncio
async def test_handler_error_does_not_block():
    bus = EventBus()
    results = []

    async def bad_handler(event: Event):
        raise ValueError("oops")

    async def good_handler(event: Event):
        results.append("ok")

    bus.on("evt", bad_handler)
    bus.on("evt", good_handler)
    await bus.emit("evt", source="test")

    assert results == ["ok"]


@pytest.mark.asyncio
async def test_off():
    bus = EventBus()
    results = []

    async def handler(event: Event):
        results.append("got")

    bus.on("evt", handler)
    bus.off("evt", handler)
    await bus.emit("evt", source="test")

    assert results == []