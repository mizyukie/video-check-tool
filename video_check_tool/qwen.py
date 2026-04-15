from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import re
import shutil
import sys
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
SEVERITY = {"low": 0, "medium": 1, "high": 2}
DEFAULT_POLICY = {
    "platform_name": "general_sns",
    "categories": [
        {"id": "explicit_nudity", "description": "Visible genitals or explicit sexual acts."},
        {"id": "sexual_content", "description": "Sexualized posing or strongly suggestive imagery."},
        {"id": "graphic_violence", "description": "Gore, severe wounds, or blood-heavy injury."},
        {"id": "self_harm", "description": "Self-harm or content encouraging self-harm."},
        {"id": "weapons", "description": "Prominent weapons or direct weapon threats."},
        {"id": "drugs", "description": "Drug use or glamorized substance abuse."},
        {"id": "hate_or_harassment", "description": "Hate symbols, slurs, or targeted abuse."},
        {"id": "personal_information", "description": "IDs, phone numbers, addresses, plates, or private docs."},
        {"id": "illegal_activity", "description": "Crime, vandalism, theft, or dangerous illegal behavior."},
        {"id": "shocking_content", "description": "Corpses, abuse, medical trauma, or disturbing scenes."},
        {"id": "other", "description": "Anything else likely to violate SNS or brand safety rules."},
    ],
    "decision_rules": {
        "reject_categories": ["explicit_nudity", "graphic_violence", "self_harm"],
    },
}
CATEGORY_ALIASES = {
    "nudity": "explicit_nudity",
    "sexual": "sexual_content",
    "violence": "graphic_violence",
    "gore": "graphic_violence",
    "selfharm": "self_harm",
    "weapon": "weapons",
    "drug": "drugs",
    "hate": "hate_or_harassment",
    "harassment": "hate_or_harassment",
    "pii": "personal_information",
    "illegal": "illegal_activity",
    "shocking": "shocking_content",
}
ACTION_ALIASES = {
    "allow": "allow",
    "ok": "allow",
    "safe": "allow",
    "blur": "blur",
    "cut": "cut",
    "trim": "cut",
    "review": "manual_review",
    "manual_review": "manual_review",
    "reject": "reject",
    "block": "reject",
}


@dataclass(slots=True)
class Sample:
    label: str
    timestamp_sec: float
    frame_index: int
    file_path: Path


def require_module(name: str, hint: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise SystemExit(f"Missing dependency '{name}'. Install it first: {hint}") from exc


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check local videos for SNS-unsuitable visual content with Qwen VL.")
    p.add_argument("input_path", nargs="?", default="videos/input")
    p.add_argument("--output")
    p.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--cache-dir")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--attn-implementation")
    p.add_argument("--sample-every-sec", type=float, default=3.0)
    p.add_argument("--max-frames", type=int, default=36)
    p.add_argument("--frames-per-batch", type=int, default=6)
    p.add_argument("--max-side", type=int, default=960)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--max-new-tokens", type=int, default=700)
    p.add_argument("--policy-file")
    p.add_argument("--keep-raw-output", action="store_true")
    return p


def load_policy(path: str | None) -> dict:
    if not path:
        return json.loads(json.dumps(DEFAULT_POLICY))
    policy_path = Path(path).expanduser().resolve()
    if not policy_path.is_file():
        raise SystemExit(f"Policy file not found: {policy_path}")
    try:
        return json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Policy file is not valid JSON: {policy_path}") from exc


def collect_videos(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTS:
            raise SystemExit(f"Unsupported file extension: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Input path does not exist: {path}")
    videos = sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"No video files found in: {path}")
    return videos


def _load_error(model_id: str, local_only: bool, exc: Exception):
    detail = str(exc).strip()
    if "Torchvision library" in detail or "torchvision" in detail.lower():
        raise SystemExit(
            "Could not load the Qwen image processor because torchvision is missing. "
            "Install a torchvision build that matches your PyTorch version, then rerun.\n"
            f"{detail}"
        ) from exc
    if "protobuf library" in detail or "protobuf" in detail.lower():
        raise SystemExit(
            "Could not load the Qwen processor because protobuf is missing. "
            "Install protobuf, then rerun.\n"
            f"{detail}"
        ) from exc
    if local_only:
        raise SystemExit(
            f"Could not load '{model_id}' from local cache only. "
            f"Download the model first or pass a fully downloaded local path with --model-id.\n{detail}"
        ) from exc
    raise SystemExit(
        f"Could not load '{model_id}'. The model is not fully cached and this environment could not reach Hugging Face. "
        f"Download it once with network access, or pass a fully downloaded local path with --model-id, or rerun later with --local-files-only.\n{detail}"
    ) from exc


def _resolve_cached_model_path(model_id: str, cache_dir: str | None, local_only: bool) -> str:
    model_path = Path(model_id).expanduser()
    if model_path.exists() or not local_only or "/" not in model_id:
        return str(model_path if model_path.exists() else model_id)
    cache_root = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    ref_path = repo_dir / "refs" / "main"
    if ref_path.is_file():
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot = repo_dir / "snapshots" / revision
        if snapshot.is_dir():
            return str(snapshot)
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.is_dir():
        snapshots = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
        if snapshots:
            return str(snapshots[0])
    return model_id


def load_model(model_id: str, device: str, dtype_name: str, attn: str | None, cache_dir: str | None, local_files_only: bool):
    torch = require_module("torch", "install PyTorch from https://pytorch.org/get-started/locally/")
    transformers = require_module("transformers", "pip install -U transformers accelerate pillow")
    source = _resolve_cached_model_path(model_id, cache_dir, local_files_only)
    common = {"local_files_only": local_files_only}
    if cache_dir:
        common["cache_dir"] = cache_dir
    try:
        processor = transformers.AutoProcessor.from_pretrained(source, **common)
    except Exception as exc:
        _load_error(model_id, local_files_only, exc)
    dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
    lower = model_id.lower()
    names = []
    if "qwen3-vl" in lower:
        names.append("Qwen3VLForConditionalGeneration")
    if "qwen2.5-vl" in lower or "qwen2_5_vl" in lower:
        names.append("Qwen2_5_VLForConditionalGeneration")
    if "qwen2-vl" in lower:
        names.append("Qwen2VLForConditionalGeneration")
    names.extend(["Qwen3VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration", "AutoModelForVision2Seq"])
    cls = next((getattr(transformers, name, None) for name in names if getattr(transformers, name, None)), None)
    if cls is None:
        raise SystemExit("No supported Qwen VL model class found. Update transformers.")
    kwargs = {"torch_dtype": dtype}
    if attn:
        kwargs["attn_implementation"] = attn
    use_auto_map = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if use_auto_map:
        kwargs["device_map"] = "auto"
    kwargs.update(common)
    try:
        model = cls.from_pretrained(source, **kwargs)
    except Exception as exc:
        _load_error(model_id, local_files_only, exc)
    if not use_auto_map:
        model = model.to("cuda" if device == "cuda" else "cpu")
    model.eval()
    return torch, processor, model


def get_device(model):
    return getattr(model, "device", next(model.parameters()).device)


def timecode(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    minutes, rest = divmod(ms, 60000)
    secs, ms = divmod(rest, 1000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def extract_frames(video_path: Path, temp_dir: Path, sample_sec: float, max_frames: int, max_side: int, quality: int):
    cv2 = require_module("cv2", "pip install opencv-python-headless")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if frame_count > 0 else 0.0
    step = max(1, int(round(sample_sec * fps)))
    indices = list(range(0, frame_count, step)) if frame_count > 0 else [0]
    if max_frames > 0 and len(indices) > max_frames:
        positions = [int(round(i * max(len(indices) - 1, 0) / max(max_frames - 1, 1))) for i in range(max_frames)]
        indices = [indices[pos] for pos in positions]
    indices = list(dict.fromkeys(indices)) or [0]
    samples: list[Sample] = []
    try:
        for i, frame_idx in enumerate(indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            if max_side > 0 and max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                frame = cv2.resize(frame, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
            out = temp_dir / f"frame_{i:04d}_{frame_idx:08d}.jpg"
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok:
                continue
            out.write_bytes(encoded.tobytes())
            samples.append(Sample(f"F{i:03d}", frame_idx / fps, frame_idx, out))
    finally:
        cap.release()
    if not samples:
        raise RuntimeError(f"No frames could be sampled from: {video_path}")
    meta = {"fps": round(fps, 3), "frame_count": frame_count, "duration_sec": round(duration, 3), "width": width, "height": height}
    return meta, samples


def prompt_schema() -> dict:
    return {
        "batch_judgement": "safe | manual_review | reject",
        "overall_reason": "short reason",
        "findings": [
            {
                "frame_label": "F001",
                "timestamp_sec": 0.0,
                # textには「画面に映ってる文字」を入れてね
                "text": "detected on-screen text",
                "category": "one of the category ids",
                "risk_level": "low | medium | high",
                "suggested_action": "allow | blur | cut | manual_review | reject",
                "reason": "short concrete reason",
            }
        ],
    }


def build_messages(samples: list[Sample], policy: dict) -> list[dict]:
    categories = "\n".join(f"- {item['id']}: {item['description']}" for item in policy["categories"])
    refs = "\n".join(f"- {sample.label}: {timecode(sample.timestamp_sec)}" for sample in samples)
    prompt = textwrap.dedent(
        f"""
        Review the following sampled video frames for a social-media preflight moderation check.
        Inspect both visuals and any on-screen text.
        Only flag frames that are clearly unsafe or should be manually reviewed.
        If evidence is ambiguous, prefer suggested_action="manual_review".

        Categories:
        {categories}

        Frame references:
        {refs}

        Return JSON only with this schema:
        {json.dumps(prompt_schema(), indent=2)}

        Rules:
        - frame_label must exactly match one of the provided labels.
        - batch_judgement must be safe, manual_review, or reject.
        - Always return the main readable on-screen text for each frame in findings, even if the frame is otherwise safe.
        - If no readable text is visible, set text to an empty string.
        """
    ).strip()
    content = [{"type": "text", "text": prompt}]
    for sample in samples:
        content.append({"type": "text", "text": f"Frame {sample.label} at {timecode(sample.timestamp_sec)}."})
        content.append({"type": "image", "image": str(sample.file_path.resolve())})
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a careful multimodal moderator. Return strict JSON only."}]},
        {"role": "user", "content": content},
    ]


def extract_json(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def normalize_text(value) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def sanitize_findings(items, samples: list[Sample], policy: dict) -> list[dict]:
    if items is None:
        return []
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return []
    by_label = {sample.label: sample for sample in samples}
    allowed = {item["id"] for item in policy["categories"]}
    findings = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("frame_label", "")).strip()
        if label not in by_label:
            continue
        category = CATEGORY_ALIASES.get(normalize_text(item.get("category")), normalize_text(item.get("category")))
        if category not in allowed:
            category = "other" if "other" in allowed else next(iter(allowed))
        action = ACTION_ALIASES.get(normalize_text(item.get("suggested_action")), "manual_review")
        risk = normalize_text(item.get("risk_level"))
        risk = risk if risk in SEVERITY else "medium"
        reason = str(item.get("reason", "")).strip() or "The model flagged this frame for manual review."
        sample = by_label[label]
        findings.append(
            {
                "frame_label": sample.label,
                "timestamp_sec": round(sample.timestamp_sec, 3),
                "frame_index": sample.frame_index,
                "text": str(item.get("text", "")).strip(),
                "category": category,
                "risk_level": risk,
                "suggested_action": action,
                "reason": reason,
            }
        )
    seen = set()
    deduped = []
    for finding in sorted(findings, key=lambda x: x["timestamp_sec"]):
        key = (finding["frame_label"], finding["category"], finding["suggested_action"], finding["reason"])
        if key not in seen:
            seen.add(key)
            deduped.append(finding)
    return deduped


def analyze_batch(processor, model, samples: list[Sample], policy: dict, max_new_tokens: int) -> dict:
    try:
        inputs = processor.apply_chat_template(
            build_messages(samples, policy),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(get_device(model))
    except Exception as exc:
        raise RuntimeError(
            "processor.apply_chat_template failed. Update transformers to a version that supports Qwen VL chat templates."
        ) from exc
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    block = extract_json(raw)
    if not block:
        reason = "The model response was not valid JSON. Review this batch manually."
        return {
            "batch_judgement": "manual_review",
            "overall_reason": reason,
            "findings": [
                {
                    "frame_label": samples[0].label,
                    "timestamp_sec": round(samples[0].timestamp_sec, 3),
                    "frame_index": samples[0].frame_index,
                    "category": "other",
                    "risk_level": "medium",
                    "suggested_action": "manual_review",
                    "reason": reason,
                }
            ],
            "raw_output": raw,
        }
    parsed = json.loads(block)
    judgement = normalize_text(parsed.get("batch_judgement"))
    judgement = "manual_review" if judgement not in {"safe", "manual_review", "reject"} else judgement
    findings = sanitize_findings(parsed.get("findings"), samples, policy)
    # if judgement == "safe":
        # findings = []
    return {
        "batch_judgement": judgement,
        "overall_reason": str(parsed.get("overall_reason", "")).strip() or "Batch moderation completed.",
        "findings": findings,
        "raw_output": raw,
    }


def merge_segments(findings: list[dict], gap_sec: float) -> list[dict]:
    segments = []
    for finding in findings:
        if segments:
            last = segments[-1]
            if (
                last["category"] == finding["category"]
                and last["suggested_action"] == finding["suggested_action"]
                and finding["timestamp_sec"] - last["end_sec"] <= gap_sec
            ):
                last["end_sec"] = finding["timestamp_sec"]
                last["end_timecode"] = timecode(finding["timestamp_sec"])
                last["frame_labels"].append(finding["frame_label"])
                if finding["reason"] not in last["reasons"]:
                    last["reasons"].append(finding["reason"])
                if SEVERITY[finding["risk_level"]] > SEVERITY[last["risk_level"]]:
                    last["risk_level"] = finding["risk_level"]
                continue
        segments.append(
            {
                "category": finding["category"],
                "risk_level": finding["risk_level"],
                "suggested_action": finding["suggested_action"],
                "start_sec": finding["timestamp_sec"],
                "end_sec": finding["timestamp_sec"],
                "start_timecode": timecode(finding["timestamp_sec"]),
                "end_timecode": timecode(finding["timestamp_sec"]),
                "frame_labels": [finding["frame_label"]],
                "reasons": [finding["reason"]],
            }
        )
    return segments

# 最終判断してる部分はここ↓
def overall_decision(findings: list[dict], policy: dict) -> tuple[str, str]:
    if not findings:
        return "allow", "No risky frames were flagged in the sampled visual checks."
    reject_categories = set(policy.get("decision_rules", {}).get("reject_categories", []))
    if any(item["category"] in reject_categories or item["suggested_action"] == "reject" for item in findings):
        return "reject", "At least one sampled frame looks severe enough to block or cut before upload."
    return "manual_review", "Potentially risky frames were found. A human review is recommended before posting."


def report_for(video_path: Path, meta: dict, samples: list[Sample], findings: list[dict], segments: list[dict], batches: list[dict], model_id: str, policy: dict, sample_sec: float) -> dict:
    decision, reason = overall_decision(findings, policy)
    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": {"model_id": model_id, "mode": "visual_frame_sampling"},
        "video": {"path": str(video_path.resolve()), **meta},
        "sampling": {
            "sample_every_sec": sample_sec,
            "sampled_frames": len(samples),
            "frame_labels": [sample.label for sample in samples],
            "coverage_note": "Only sampled frames are checked. Audio is not checked in this version.",
        },
        "coverage": {
            "visual_frames_checked": True,
            "on_screen_text_checked": True,
            "audio_checked": False,
            "spoken_words_checked": False,
        },
        "policy": policy,
        "decision": {"overall": decision, "reason": reason},
        "segments": segments,
        "findings": findings,
        "batch_summaries": [{"batch_judgement": b["batch_judgement"], "overall_reason": b["overall_reason"], "flagged_frames": [f["frame_label"] for f in b["findings"]]} for b in batches],
    }


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def load_text_rules() -> dict:
    project_root = Path(__file__).resolve().parent.parent
    rules_path = project_root / "rules.json"

    if not rules_path.exists():
        return {"banned_terms": {}, "name_rules": {}}

    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    return {
        "banned_terms": rules.get("banned_terms", {}),
        "name_rules": rules.get("name_rules", {})
    }


def apply_text_rules(report: dict, rules: dict) -> dict:
    banned_terms = rules.get("banned_terms", {})
    banned_patterns = rules.get("banned_patterns", {})
    name_rules = rules.get("name_rules", {})
    name_patterns = rules.get("name_patterns", {})

    for finding in report.get("findings", []):
        text = finding.get("text", "")
        violations = []
        suggested_text = text

        for ng_word, replacement in banned_terms.items():
            if ng_word in text:
                violations.append({
                    "rule_type": "banned_term",
                    "matched_text": ng_word,
                    "suggested_replacement": replacement
                })
                suggested_text = suggested_text.replace(ng_word, replacement)

        for pattern, replacement in banned_patterns.items():
            if re.search(pattern, text):
                violations.append({
                    "rule_type": "banned_pattern",
                    "matched_text": pattern,
                    "suggested_replacement": replacement
                })
                suggested_text = re.sub(pattern, replacement, suggested_text)

        for wrong_name, correct_name in name_rules.items():
            if wrong_name in text:
                violations.append({
                    "rule_type": "name_rule",
                    "matched_text": wrong_name,
                    "suggested_replacement": correct_name
                })
                suggested_text = suggested_text.replace(wrong_name, correct_name)

        for pattern, replacement in name_patterns.items():
            if re.search(pattern, text):
                violations.append({
                    "rule_type": "name_pattern",
                    "matched_text": pattern,
                    "suggested_replacement": replacement
                })
                suggested_text = re.sub(pattern, replacement, suggested_text)

        finding["rule_violations"] = violations
        finding["has_rule_violation"] = len(violations) > 0
        finding["suggested_text"] = suggested_text

    rule_violation_count = sum(
        1 for finding in report.get("findings", [])
        if finding.get("has_rule_violation")
    )

    report["text_rule_summary"] = {
        "checked": True,
        "rule_violation_count": rule_violation_count
    }

    return report

def output_targets(input_path: Path, videos: list[Path], output: str | None) -> dict[Path, Path]:
    project_root = Path(__file__).resolve().parent.parent
    default_reports_dir = project_root / "reports"

    if len(videos) == 1:
        video = videos[0]
        if not output:
            default_reports_dir.mkdir(parents=True, exist_ok=True)
            return {video: default_reports_dir / f"{video.stem}_moderation_report.json"}

        target = Path(output).expanduser().resolve()
        if target.suffix.lower() == ".json":
            return {video: target}

        target.mkdir(parents=True, exist_ok=True)
        return {video: target / f"{video.stem}_moderation_report.json"}

    if output and Path(output).suffix.lower() == ".json":
        raise SystemExit("When checking multiple videos, --output must be a directory, not a .json file.")

    out_dir = Path(output).expanduser().resolve() if output else default_reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return {video: out_dir / f"{video.stem}_moderation_report.json" for video in videos}


def analyze_video(video_path: Path, out_path: Path, args, processor, model, policy: dict) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_path.parent / f"_qwen_vl_frames_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        meta, samples = extract_frames(
            video_path,
            temp_dir,
            args.sample_every_sec,
            args.max_frames,
            args.max_side,
            args.jpeg_quality
        )
        batches = []
        findings = []

        for start in range(0, len(samples), args.frames_per_batch):
            batch = analyze_batch(
                processor,
                model,
                samples[start : start + args.frames_per_batch],
                policy,
                args.max_new_tokens
            )
            batches.append(batch)
            findings.extend(batch["findings"])

        findings = sorted(findings, key=lambda x: x["timestamp_sec"])
        segments = merge_segments(findings, max(args.sample_every_sec * 1.5, 1.0))
        report = report_for(
            video_path,
            meta,
            samples,
            findings,
            segments,
            batches,
            args.model_id,
            policy,
            args.sample_every_sec
        )

        rules = load_text_rules()
        report = apply_text_rules(report, rules)

        if args.keep_raw_output:
            report["raw_batch_outputs"] = [
                {
                    "batch_judgement": b["batch_judgement"],
                    "overall_reason": b["overall_reason"],
                    "raw_output": b["raw_output"]
                }
                for b in batches
            ]

        write_json(out_path, report)
        return report
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    args = parser().parse_args()
    if args.sample_every_sec <= 0 or args.max_frames <= 0 or args.frames_per_batch <= 0:
        raise SystemExit("sample_every_sec, max_frames, and frames_per_batch must be positive.")
    input_path = Path(args.input_path).expanduser().resolve()
    policy = load_policy(args.policy_file)
    videos = collect_videos(input_path)
    targets = output_targets(input_path, videos, args.output)
    torch, processor, model = load_model(
        args.model_id,
        args.device,
        args.torch_dtype,
        args.attn_implementation,
        args.cache_dir,
        args.local_files_only,
    )
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. A 4B VL model will be slow on CPU.", file=sys.stderr)
    summary = []
    for video in videos:
        report = analyze_video(video, targets[video], args, processor, model, policy)

        overall = report["decision"]["overall"]
        reason = report["decision"]["reason"]
        findings_count = len(report["findings"])

        print(f"\n=== {video.name} ===")
        print(f"overall   : {overall}")
        print(f"findings  : {findings_count}")
        print(f"report    : {targets[video]}")
        print(f"reason    : {reason}")

        summary.append({
            "video_path": str(video.resolve()),
            "report_path": str(targets[video].resolve()),
            "overall": overall,
            "findings": findings_count,
        })

    if len(summary) > 1:
        summary_path = (Path(args.output).expanduser().resolve() if args.output else input_path / "reports") / "moderation_summary.json"
        write_json(summary_path, {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "videos": summary
        })
        print(f"[summary] {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

for pattern, replacement in rules.get("banned_patterns", {}).items():
    if re.search(pattern, text):
        violations.append({
            "rule_type": "banned_pattern",
            "matched_text": pattern,
            "suggested_replacement": replacement
        })
        suggested_text = re.sub(pattern, replacement, suggested_text)

for pattern, replacement in rules.get("name_patterns", {}).items():
    if re.search(pattern, text):
        violations.append({
            "rule_type": "name_pattern",
            "matched_text": pattern,
            "suggested_replacement": replacement
        })
        suggested_text = re.sub(pattern, replacement, suggested_text)