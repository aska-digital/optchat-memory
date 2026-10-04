"""Tests for the append-only log (optchat/store.py)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optchat.store import ChatLog, LogLocked  # noqa: E402


def test_append_read_roundtrip(tmp_path):
    log = ChatLog(tmp_path / "home" / "optchat")
    try:
        i0 = log.append("user", "hello")
        i1 = log.append("talk", "hi there")
        assert (i0, i1) == (0, 1)
        assert len(log) == 2
        m = log.read(0)
        assert m["kind"] == "user" and m["text"] == "hello" and m["i"] == 0
        assert m["date"] and m["size"] == len("user: hello".encode())
        assert log.read(1)["kind"] == "talk"
    finally:
        log.close()


def test_ids_survive_reload(tmp_path):
    root = tmp_path / "home" / "optchat"
    log = ChatLog(root)
    log.append("user", "first")
    log.close()
    log2 = ChatLog(root)
    try:
        assert len(log2) == 1
        assert log2.append("user", "second") == 1
        assert log2.read(0)["text"] == "first"
    finally:
        log2.close()


def test_torn_line_skipped(tmp_path):
    root = tmp_path / "home" / "optchat"
    log = ChatLog(root)
    log.append("user", "ok")
    log.close()
    day_file = next((root / "main").glob("*.jsonl"))
    with open(day_file, "ab") as fh:
        fh.write(b'{"i": 999, "kind": "user", "text": "BROKEN\n')
    log2 = ChatLog(root)
    try:
        assert len(log2) == 1  # torn line skipped, ids stay dense
        assert log2.append("user", "after") == 1
    finally:
        log2.close()


def test_second_writer_fails_loudly(tmp_path):
    root = tmp_path / "home" / "optchat"
    log = ChatLog(root)
    try:
        with pytest.raises(LogLocked):
            ChatLog(root)
    finally:
        log.close()
    # After close, a new writer works.
    log2 = ChatLog(root)
    log2.close()


def test_unknown_kind_rejected(tmp_path):
    log = ChatLog(tmp_path / "home" / "optchat")
    try:
        with pytest.raises(ValueError):
            log.append("thought", "must not be logged")
    finally:
        log.close()


def test_day_files_are_valid_jsonl(tmp_path):
    root = tmp_path / "home" / "optchat"
    log = ChatLog(root)
    log.append("user", "x")
    log.close()
    for path in (root / "main").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            assert {"i", "kind", "text", "size", "date"} <= set(obj)
