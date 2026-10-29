"""WhatsApp export parsing and local dataset preparation."""

from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable



MEDIA_ONLY_RE = re.compile(
    r"^(?:(?:<[^>]+ omitted>|<unknown message>|this message was deleted)\s*)+$", re.I
)
MEDIA_TAG_RE = re.compile(
    r"<(?:image|gif|video|sticker|audio|document|unknown message)[^>]*>", re.I
)
MINI_START_RE = re.compile(r"^\s*# Mini convo\s+\d+\s*$", re.I)
_DATE = r"(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2})"
_TIME = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp]\.?(?:\s*[Mm])\.?)?"
_STAMP = rf"{_DATE},?\s+{_TIME}"
RAW_BRACKET_RE = re.compile(
    rf"^\s*\[({_STAMP})\]\s*(?:[-–—]\s*)?([^:]+?):\s*(.*)$", re.I
)
RAW_PLAIN_RE = re.compile(rf"^\s*({_STAMP})\s*[-–—]\s*([^:]+?):\s*(.*)$", re.I)
ANY_STAMP_RE = re.compile(rf"^\s*(?:\[{_STAMP}\]|{_STAMP}(?:\s*[-–—]|$))", re.I)


@dataclass
class PreparationResult:
    dataset_dir: Path
    report: dict


def _normalize_speaker(label: str, target: str, aliases: set[str]) -> str:
    label = " ".join(label.replace("\u200e", "").split()).strip()
    return target if label in aliases else label


def _clean_text(text: str) -> str:
    text = (
        text.replace("\u200e", "")
        .replace("\ufeff", "")
        .replace("\u202f", " ")
        .replace("\xa0", " ")
        .strip()
    )
    if not text or MEDIA_ONLY_RE.fullmatch(text):
        return ""
    text = MEDIA_TAG_RE.sub("", text).strip()
    if not text or MEDIA_ONLY_RE.fullmatch(text) or len(text) > 1800:
        return ""
    return text


def _parse_datetime(value: str, date_order: str) -> datetime | None:
    try:
        date_part, time_part = re.split(r",?\s+", value.replace("\u202f", " ").replace("\xa0", " ").strip(), maxsplit=1)
        date_fields = re.split(r"[/.-]", date_part)
        if len(date_fields) != 3:
            return None
        if len(date_fields[0]) == 4:
            year, month, day = map(int, date_fields)
        else:
            first, second, year = map(int, date_fields)
            if year < 100:
                year += 2000 if year < 69 else 1900
            month, day = (second, first) if date_order.upper() == "DMY" else (first, second)

        time_match = re.fullmatch(
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
            r"(?:\s*(?P<ampm>[ap]\.?(?:\s*m)\.?))?",
            time_part,
            re.I,
        )
        if not time_match:
            return None
        hour = int(time_match.group("hour"))
        minute = int(time_match.group("minute"))
        second = int(time_match.group("second") or 0)
        ampm = (time_match.group("ampm") or "").replace(".", "").replace(" ", "").upper()
        if ampm:
            if not 1 <= hour <= 12:
                return None
            hour = hour % 12 + (12 if ampm == "PM" else 0)
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def parse_raw_text(text: str, target: str, aliases: set[str], date_order: str):
    messages: list[dict] = []
    current = None
    ignored = raw_headers = bad_timestamps = 0

    for line in text.splitlines():
        line = line.replace("\u202f", " ").replace("\xa0", " ").replace("\u200e", "")
        match = RAW_BRACKET_RE.match(line) or RAW_PLAIN_RE.match(line)
        if match:
            raw_headers += 1
            stamp, speaker, body = match.groups()
            when = _parse_datetime(stamp, date_order)
            if when is None:
                current = None
                bad_timestamps += 1
                ignored += 1
                continue
            speaker = _normalize_speaker(speaker, target, aliases)
            if not speaker or speaker.startswith("-"):
                current = None
                ignored += 1
                continue
            current = {"speaker": speaker, "text": body.rstrip(), "time": when, "stamp": stamp.strip()}
            messages.append(current)
        elif ANY_STAMP_RE.match(line):
            current = None
            ignored += 1
        elif current is not None:
            current["text"] += "\n" + line

    for message in messages:
        message["text"] = _clean_text(message["text"])
    messages = [message for message in messages if message["text"]]
    return messages, {
        "raw_headers": raw_headers,
        "invalid_timestamps": bad_timestamps,
        "kept_text_messages": len(messages),
        "ignored_system_events": ignored,
    }


def parse_mini_text(text: str, target: str, aliases: set[str]):
    chunks: list[list[dict]] = []
    current: list[dict] | None = None
    who = None
    lines: list[str] = []
    active = False
    windows = 0

    def finish_message():
        nonlocal who, lines
        if who is not None and current is not None:
            body = _clean_text("\n".join(lines))
            if body:
                current.append({"speaker": who, "text": body, "time": None, "stamp": ""})
        who, lines = None, []

    def finish_chunk():
        nonlocal current, active
        if active:
            finish_message()
            if current:
                chunks.append(current)
        current, active = None, False

    for line in text.splitlines():
        if MINI_START_RE.match(line):
            finish_chunk()
            current = []
            active = False
        elif current is not None and line.strip().lower() == "start":
            active = True
        elif current is not None and line.strip().lower() == "end":
            finish_chunk()
        elif active:
            if line.startswith("context:"):
                continue
            if line.endswith(":") and line.strip():
                finish_message()
                who = _normalize_speaker(line[:-1], target, aliases)
            elif who is not None:
                lines.append(line)
    finish_chunk()

    runs: list[list[dict]] = []
    for chunk in chunks:
        if not runs:
            runs.append(chunk[:])
            continue
        previous = runs[-1]
        overlap = 0
        for count in range(min(len(previous), len(chunk)), 0, -1):
            left = [(item["speaker"], item["text"]) for item in previous[-count:]]
            right = [(item["speaker"], item["text"]) for item in chunk[:count]]
            if left == right:
                overlap = count
                break
        if overlap:
            previous.extend(chunk[overlap:])
        else:
            runs.append(chunk[:])
    return runs, len(chunks)


def _read_sources(inputs: Iterable[Path]):
    """Yield (display name, text, sha256) without extracting ZIP members to disk."""
    found = 0
    for input_path in inputs:
        path = Path(input_path)
        if path.is_dir():
            for child in sorted(path.rglob("*.txt")):
                raw = child.read_bytes()
                found += 1
                yield str(child.relative_to(path)), raw.decode("utf-8-sig", errors="replace"), hashlib.sha256(raw).hexdigest()
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.is_dir() or not member.filename.lower().endswith(".txt"):
                        continue
                    # ZIP members are read directly; no member path is written to disk.
                    raw = archive.read(member)
                    found += 1
                    yield f"{path.name}/{PurePosixPath(member.filename).name}", raw.decode("utf-8-sig", errors="replace"), hashlib.sha256(raw).hexdigest()
        elif path.suffix.lower() == ".txt":
            raw = path.read_bytes()
            found += 1
            yield path.name, raw.decode("utf-8-sig", errors="replace"), hashlib.sha256(raw).hexdigest()
        else:
            raise ValueError(f"Choose .txt or .zip input files: {path.name}")
    if not found:
        raise ValueError("No .txt files were found in the selected inputs.")


def _segments_from_run(run: list[dict], gap_minutes: int = 60, max_messages: int = 100):
    segments, current = [], []
    previous_time = None
    for message in run:
        when = message["time"]
        gap = (when - previous_time).total_seconds() / 60 if when is not None and previous_time is not None else None
        if current and ((gap is not None and (gap > gap_minutes or gap < -1)) or len(current) >= max_messages):
            segments.append(current)
            current = []
        current.append(message)
        if when is not None:
            previous_time = when
    if current:
        segments.append(current)
    return segments


def _make_examples(segment: list[dict], target: str, max_context: int = 5, max_target_replies: int = 3):
    examples, index = [], 0
    while index < len(segment):
        if segment[index]["speaker"] != target:
            index += 1
            continue
        end = index
        while end < len(segment) and segment[end]["speaker"] == target:
            end += 1
        preceding = segment[max(0, index - max_context):index]
        if preceding and any(item["speaker"] != target for item in preceding):
            replies = segment[index:min(end, index + max_target_replies)]
            response = "\n".join(item["text"] for item in replies)
            prompt = []
            for item in preceding:
                role = "assistant" if item["speaker"] == target else "user"
                content = item["text"] if role == "assistant" else f"{item['speaker']}: {item['text']}"
                if prompt and prompt[-1]["role"] == role:
                    prompt[-1]["content"] += "\n" + content
                else:
                    prompt.append({"role": role, "content": content})
            while prompt and prompt[0]["role"] == "assistant":
                prompt.pop(0)
            if prompt and prompt[-1]["role"] == "user" and response:
                examples.append({"prompt": prompt, "completion": [{"role": "assistant", "content": response}]})
        index = end
    return examples


def _example_key(example: dict) -> str:
    packed = json.dumps(example, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def _split_groups(groups: list[list[dict]], seed: int, require_minimum: bool = True):
    owner: dict[str, int] = {}
    parents = list(range(len(groups)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for group_index, group in enumerate(groups):
        for example in group:
            key = _example_key(example)
            if key in owner:
                union(group_index, owner[key])
            else:
                owner[key] = group_index

    clusters: dict[int, list[dict]] = collections.OrderedDict()
    for index, group in enumerate(groups):
        clusters.setdefault(find(index), []).extend(group)
    unique_clusters, used, duplicates = [], set(), 0
    for cluster in clusters.values():
        clean = []
        for example in cluster:
            key = _example_key(example)
            if key in used:
                duplicates += 1
            else:
                used.add(key)
                clean.append(example)
        if clean:
            unique_clusters.append(clean)
    if require_minimum and len(unique_clusters) < 10:
        raise ValueError(
            f"Only {len(unique_clusters)} independent conversation groups remain after deduplication; "
            "at least 10 are needed for a train/validation/test split."
        )

    rng = random.Random(seed)
    rng.shuffle(unique_clusters)
    total = sum(map(len, unique_clusters))
    sets = {name: [] for name in ("train", "validation", "test")}
    for index, cluster in enumerate(unique_clusters):
        remaining = len(unique_clusters) - index
        if not sets["validation"] and remaining <= 2:
            bucket = "validation"
        elif not sets["test"] and remaining <= 1:
            bucket = "test"
        else:
            counts = {name: len(values) for name, values in sets.items()}
            goals = {"train": .8 * total, "validation": .1 * total, "test": .1 * total}
            bucket = min(sets, key=lambda name: counts[name] / goals[name])
        sets[bucket].extend(cluster)

    for name, examples in sets.items():
        rng.shuffle(examples)
        keys = [_example_key(example) for example in examples]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate examples remained in {name} data.")
    if require_minimum and not all(sets.values()):
        raise ValueError("Could not produce non-empty training, validation, and test splits.")
    return sets, len(unique_clusters), duplicates


def _source_is_mini(text: str, input_format: str) -> bool:
    lines = text.splitlines()[:400]
    has_mini = any(MINI_START_RE.match(line) for line in lines)
    return input_format == "mini" or (input_format == "auto" and has_mini)


def scan_speakers(inputs: Iterable[Path], input_format: str = "auto", date_order: str = "MDY") -> dict:
    """Parse selected exports and return speaker message counts for the UI picker."""
    if input_format not in {"auto", "whatsapp", "mini"}:
        raise ValueError("Format must be auto, whatsapp, or mini.")
    if date_order.upper() not in {"MDY", "DMY"}:
        raise ValueError("Date order must be MDY or DMY.")
    counts = collections.Counter()
    seen_raw = set()
    source_count = 0
    for source_name, text, _digest in _read_sources(inputs):
        source_count += 1
        if _source_is_mini(text, input_format):
            runs, _windows = parse_mini_text(text, "", set())
            rows = [message for run in runs for message in run]
            unique = []
            for message in rows:
                key = ("mini", message["speaker"], message["text"])
                if key not in seen_raw:
                    seen_raw.add(key)
                    unique.append(message)
        else:
            rows, _metrics = parse_raw_text(text, "", set(), date_order)
            if not rows:
                raise ValueError(
                    f"No WhatsApp messages were parsed from {source_name}. Check the export format and date order."
                )
            unique = []
            for message in rows:
                key = (message["stamp"], message["speaker"], message["text"])
                if key not in seen_raw:
                    seen_raw.add(key)
                    unique.append(message)
        counts.update(message["speaker"] for message in unique if message["speaker"])
    return {"speakers": counts, "source_files": source_count, "messages": sum(counts.values())}


def _collect_examples(
    inputs: Iterable[Path],
    target_username: str,
    target_aliases: Iterable[str],
    input_format: str,
    date_order: str,
    seed: int,
    gap_minutes: int,
    max_segment_messages: int,
    max_context_messages: int,
    max_target_replies: int,
):
    target = target_username.strip()
    if not target:
        raise ValueError("Enter the exact speaker name from the chat export.")
    if input_format not in {"auto", "whatsapp", "mini"}:
        raise ValueError("Format must be auto, whatsapp, or mini.")
    if date_order.upper() not in {"MDY", "DMY"}:
        raise ValueError("Date order must be MDY or DMY.")
    aliases = {target, *(alias.strip() for alias in target_aliases if alias.strip())}

    all_runs, raw_seen, source_reports = [], set(), []
    speaker_counts = collections.Counter()
    for source_name, text, digest in _read_sources(inputs):
        is_mini = _source_is_mini(text, input_format)
        metrics = None
        if is_mini:
            runs, windows = parse_mini_text(text, target, aliases)
            info = f"{windows} windows → {len(runs)} reconstructed runs"
            kind = "mini-conversations"
        else:
            rows, metrics = parse_raw_text(text, target, aliases, date_order)
            if not rows:
                raise ValueError(
                    f"No WhatsApp messages were parsed from {source_name}. "
                    "Check the export format and date order."
                )
            unique = []
            for message in rows:
                key = (message["stamp"], message["speaker"], message["text"])
                if key not in raw_seen:
                    raw_seen.add(key)
                    unique.append(message)
            runs = [unique] if unique else []
            info = (
                f"{len(rows)} text messages → {len(unique)} unique; "
                f"{metrics['raw_headers']} message headers, {metrics['ignored_system_events']} system events"
            )
            kind = "raw-chat"

        appearances = sum(message["speaker"] == target for run in runs for message in run)
        for run in runs:
            speaker_counts.update(message["speaker"] for message in run)
        source_reports.append({
            "source": source_name,
            "sha256": digest,
            "kind": kind,
            "info": info,
            "target_appearances": appearances,
            "raw_metrics": metrics,
        })
        all_runs.extend((source_name, run) for run in runs)

    if not speaker_counts.get(target):
        top_speakers = ", ".join(name for name, _ in speaker_counts.most_common(12)) or "none"
        raise ValueError(f"No messages matched {target!r}. Parsed speaker labels include: {top_speakers}")

    groups = []
    for _source, run in all_runs:
        for segment in _segments_from_run(run, gap_minutes, max_segment_messages):
            examples = _make_examples(segment, target, max_context_messages, max_target_replies)
            if examples:
                groups.append(examples)
    sets, cluster_count, duplicate_count = _split_groups(groups, seed, require_minimum=False)
    report = {
        "target_username": target,
        "target_aliases": sorted(aliases - {target}),
        "input_format": input_format,
        "date_order": date_order.upper(),
        "source_files": source_reports,
        "independent_clusters": cluster_count,
        "conversation_groups": len(groups),
        "duplicate_examples_removed": duplicate_count,
        "sets": {name: len(examples) for name, examples in sets.items()},
        "example_count": sum(len(examples) for examples in sets.values()),
        "settings": {
            "seed": seed,
            "gap_minutes": gap_minutes,
            "max_segment_messages": max_segment_messages,
            "max_context_messages": max_context_messages,
            "max_target_replies": max_target_replies,
        },
        "warning": "Group-based splitting reduces related-context overlap, but cannot remove it completely when original conversation IDs are unavailable.",
    }
    return target, sets, report


def analyze_dataset(
    inputs: Iterable[Path],
    target_username: str,
    target_aliases: Iterable[str] = (),
    input_format: str = "auto",
    date_order: str = "MDY",
    seed: int = 3407,
    gap_minutes: int = 60,
    max_segment_messages: int = 100,
    max_context_messages: int = 5,
    max_target_replies: int = 3,
) -> dict:
    _target, _sets, report = _collect_examples(
        inputs, target_username, target_aliases, input_format, date_order, seed,
        gap_minutes, max_segment_messages, max_context_messages, max_target_replies,
    )
    return report


def prepare_dataset(
    inputs: Iterable[Path],
    target_username: str,
    target_aliases: Iterable[str] = (),
    input_format: str = "auto",
    date_order: str = "MDY",
    output_parent: Path | None = None,
    seed: int = 3407,
    gap_minutes: int = 60,
    max_segment_messages: int = 100,
    max_context_messages: int = 5,
    max_target_replies: int = 3,
) -> PreparationResult:
    target, sets, report = _collect_examples(
        inputs, target_username, target_aliases, input_format, date_order, seed,
        gap_minutes, max_segment_messages, max_context_messages, max_target_replies,
    )
    if report["independent_clusters"] < 10:
        raise ValueError(
            f"Only {report['independent_clusters']} independent conversation groups remain after deduplication; "
            "at least 10 are needed for a train/validation/test split."
        )
    if not all(sets.values()):
        raise ValueError("Could not produce non-empty training, validation, and test splits.")
    dataset_root = Path(output_parent or (Path.home() / "FriendLLM" / "datasets"))
    dataset_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = dataset_root / f"dataset_{stamp}"
    suffix = 1
    while dataset_dir.exists():
        dataset_dir = dataset_root / f"dataset_{stamp}_{suffix}"
        suffix += 1
    dataset_dir.mkdir()

    for name, examples in sets.items():
        with (dataset_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as output:
            for example in examples:
                output.write(json.dumps(example, ensure_ascii=False) + "\n")
    (dataset_dir / "preparation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return PreparationResult(dataset_dir=dataset_dir, report=report)
