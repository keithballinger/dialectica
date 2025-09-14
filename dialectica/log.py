from __future__ import annotations

def step(msg: str) -> None:
    print(f"[step] {msg}")


def info(msg: str) -> None:
    print(f"  - {msg}")


def done(msg: str) -> None:
    print(f"[done] {msg}")


def warn(msg: str) -> None:
    print(f"[warn] {msg}")

