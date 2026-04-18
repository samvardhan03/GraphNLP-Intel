"""Async generator: consume from Kafka / webhook stream as document sources."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


async def kafka_stream(
    topic: str,
    bootstrap_servers: str,
    *,
    group_id: str = "graphnlp-consumer",
) -> AsyncIterator[str]:
    """Yield raw document strings from a Kafka topic.

    Requires ``aiokafka`` to be installed. If not available, raises
    ``ImportError`` with a helpful message.

    Parameters
    ----------
    topic : str
        Kafka topic to consume from.
    bootstrap_servers : str
        Comma-separated Kafka broker addresses.
    group_id : str
        Consumer group ID.

    Yields
    ------
    str
        Decoded document string from each Kafka message.
    """
    try:
        from aiokafka import AIOKafkaConsumer
    except ImportError:
        raise ImportError(
            "aiokafka is required for Kafka streaming. "
            "Install with: pip install aiokafka"
        )

    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )

    await consumer.start()
    try:
        async for msg in consumer:
            if msg.value is not None:
                try:
                    text = msg.value.decode("utf-8")
                    if text.strip():
                        yield text
                except UnicodeDecodeError:
                    logger.warning("Skipping message with invalid UTF-8 encoding")
    finally:
        await consumer.stop()


async def webhook_stream(queue: asyncio.Queue) -> AsyncIterator[str]:
    """Yield document strings from an asyncio.Queue.

    This is fed by the ``/v1/stream`` WebSocket endpoint.  The queue acts
    as a bridge between the HTTP/WebSocket handler and the processing pipeline.

    Parameters
    ----------
    queue : asyncio.Queue
        Queue from which raw document strings are consumed.

    Yields
    ------
    str
        Raw document string placed into the queue.
    """
    while True:
        item = await queue.get()
        if item is None:
            # Sentinel value — stop iteration
            break
        if isinstance(item, str) and item.strip():
            yield item
        queue.task_done()
