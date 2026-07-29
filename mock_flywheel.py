#!/usr/bin/env python3
"""
Minimal mock of the Dexmate data-collection flywheel (readme §4).

No Kafka, no Flink, no Iceberg — just stdlib. In-memory queues stand in for
each hop so the *shape* of the pipeline is runnable and testable:

    interaction  ->  [topic]  ->  Flink(scrub+consent)  ->  bronze lake
                     (Kafka)      (stream processor)        (Iceberg)

Run:  python3 mock_flywheel.py
"""
from __future__ import annotations
import queue
import re
import time
from dataclasses import dataclass, field, asdict

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")


@dataclass
class Interaction:
    """One raw event/content/telemetry/agent-trace item (Protobuf stand-in)."""
    kind: str
    text: str
    consent: bool = True
    provenance: str = "unknown"
    ts: float = field(default_factory=time.time)


class Topic:
    """In-memory stand-in for a Kafka topic."""
    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()

    def produce(self, item: Interaction) -> None:
        self._q.put(item)

    def drain(self):
        while not self._q.empty():
            yield self._q.get()


def scrub_pii(text: str) -> str:
    text = EMAIL.sub("<email>", text)
    text = PHONE.sub("<phone>", text)
    return text


def flink(topic: Topic) -> list[dict]:
    """Stream processor: consent-filter + PII scrub + provenance stamp."""
    bronze: list[dict] = []
    dropped = 0
    for item in topic.drain():
        if not item.consent:          # consent-filter
            dropped += 1
            continue
        rec = asdict(item)
        rec["text"] = scrub_pii(item.text)   # PII scrub
        rec["stamped"] = True                # provenance + consent stamped
        bronze.append(rec)
    print(f"[flink] consent-dropped={dropped} promoted={len(bronze)}")
    return bronze


def gdpr_erase(bronze: list[dict], provenance: str) -> list[dict]:
    """Erasure propagates to the lake (readme §4)."""
    kept = [r for r in bronze if r["provenance"] != provenance]
    print(f"[gdpr] erased provenance={provenance} removed={len(bronze) - len(kept)}")
    return kept


def main() -> None:
    topic = Topic()
    for it in [
        Interaction("event", "user alice@corp.com tapped start", provenance="app-web"),
        Interaction("telemetry", "joint torque 4.2Nm call 555-123-4567", provenance="robot-7"),
        Interaction("content", "no-consent note", consent=False, provenance="app-web"),
        Interaction("agent_trace", "planned grasp; contacted bob@x.io", provenance="agent-3"),
    ]:
        topic.produce(it)

    bronze = flink(topic)
    for r in bronze:
        print("  bronze <-", r["kind"], "|", r["text"])

    bronze = gdpr_erase(bronze, provenance="app-web")
    print(f"[lake] bronze rows = {len(bronze)}")

    # Flywheel closes: bronze feeds labeling/training (§5/§7) downstream.
    assert all("@" not in r["text"] for r in bronze), "PII leaked!"
    assert all(r["consent"] for r in bronze), "consent violation!"
    print("[ok] flywheel invariants hold")


if __name__ == "__main__":
    main()
