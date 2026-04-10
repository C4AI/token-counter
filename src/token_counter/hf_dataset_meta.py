"""
Helpers for retrieving Hugging Face dataset metadata used by the terminal UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from datasets import get_dataset_infos, load_dataset_builder
from huggingface_hub import HfApi


@dataclass(frozen=True)
class HFSplitSize:
    num_examples: Optional[int]
    source: Optional[str] = None
    note: Optional[str] = None


def _card_data_to_dict(card_data: Any) -> dict[str, Any]:
    if card_data is None:
        return {}
    if isinstance(card_data, dict):
        return dict(card_data)
    if hasattr(card_data, "to_dict"):
        data = card_data.to_dict()
        if isinstance(data, dict):
            return data
    if hasattr(card_data, "__dict__"):
        return dict(card_data.__dict__)
    return {}


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _collect_split_counts(node: Any, split: str, matches: list[int]) -> None:
    if isinstance(node, dict):
        direct_split = node.get(split)
        if isinstance(direct_split, dict) and "num_examples" in direct_split:
            count = _coerce_int(direct_split.get("num_examples"))
            if count is not None:
                matches.append(count)

        if node.get("name") == split and "num_examples" in node:
            count = _coerce_int(node.get("num_examples"))
            if count is not None:
                matches.append(count)

        for value in node.values():
            if isinstance(value, (dict, list)):
                _collect_split_counts(value, split, matches)
        return

    if isinstance(node, list):
        for item in node:
            _collect_split_counts(item, split, matches)


def _split_num_examples_from_splits(splits: Any, split: str) -> Optional[int]:
    if splits is None:
        return None

    split_info = None
    if hasattr(splits, "get"):
        split_info = splits.get(split)
    elif isinstance(splits, dict):
        split_info = splits.get(split)

    if split_info is None:
        return None
    return _coerce_int(getattr(split_info, "num_examples", None))


def _resolve_from_builder(
    dataset_id: str,
    split: str,
    *,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> tuple[Optional[int], Optional[str]]:
    builder = load_dataset_builder(
        dataset_id,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    num_examples = _split_num_examples_from_splits(getattr(builder.info, "splits", None), split)
    if num_examples is None:
        return None, None

    config_name = getattr(getattr(builder, "config", None), "name", None)
    if config_name:
        return num_examples, f"dataset builder metadata ({config_name})"
    return num_examples, "dataset builder metadata"


def _resolve_from_dataset_infos(
    dataset_id: str,
    split: str,
    *,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> tuple[Optional[int], Optional[str], Optional[str]]:
    infos = get_dataset_infos(
        dataset_id,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    matches: list[int] = []
    config_hits: list[str] = []
    for config_name, info in infos.items():
        num_examples = _split_num_examples_from_splits(getattr(info, "splits", None), split)
        if num_examples is None:
            continue
        matches.append(num_examples)
        config_hits.append(config_name)

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        config_display = ", ".join(config_hits[:3])
        source = "dataset infos metadata"
        if config_display:
            source += f" ({config_display})"
        return unique_matches[0], source, None
    if len(unique_matches) > 1:
        return (
            max(unique_matches),
            "dataset infos metadata",
            (
                "Multiple configs publish different split sizes; "
                f"using the largest match for '{split}'."
            ),
        )
    return None, None, None


def resolve_dataset_split_size(
    dataset_id: str,
    split: str,
    *,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> HFSplitSize:
    """
    Best-effort lookup of `num_examples` for a dataset split.

    The Hub may expose this under `card_data.dataset_info` with slightly different
    shapes depending on how the dataset card was generated, so this helper keeps the
    parsing loose and falls back gracefully when metadata is unavailable.
    """

    errors: list[str] = []

    try:
        num_examples, source = _resolve_from_builder(
            dataset_id,
            split,
            token=token,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        errors.append(f"builder metadata failed: {type(exc).__name__}: {exc}")
    else:
        if num_examples is not None:
            return HFSplitSize(num_examples=num_examples, source=source)

    try:
        num_examples, source, note = _resolve_from_dataset_infos(
            dataset_id,
            split,
            token=token,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        errors.append(f"dataset infos failed: {type(exc).__name__}: {exc}")
    else:
        if num_examples is not None:
            return HFSplitSize(num_examples=num_examples, source=source, note=note)

    try:
        info = HfApi().dataset_info(dataset_id, token=token)
    except Exception as exc:
        errors.append(f"dataset card lookup failed: {type(exc).__name__}: {exc}")
        return HFSplitSize(
            num_examples=None,
            note="Dataset size metadata unavailable. " + " | ".join(errors),
        )

    matches: list[int] = []
    card_data = _card_data_to_dict(getattr(info, "card_data", None))
    dataset_info_payload = card_data.get("dataset_info")
    if dataset_info_payload is not None:
        _collect_split_counts(dataset_info_payload, split, matches)
    elif card_data:
        _collect_split_counts(card_data, split, matches)

    extra_dataset_info = getattr(info, "dataset_info", None)
    if extra_dataset_info is not None:
        _collect_split_counts(extra_dataset_info, split, matches)

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return HFSplitSize(
            num_examples=unique_matches[0],
            source="dataset card metadata",
            note=" | ".join(errors) if errors else None,
        )
    if len(unique_matches) > 1:
        return HFSplitSize(
            num_examples=max(unique_matches),
            source="dataset card metadata",
            note=(
                (" | ".join(errors) + " | ") if errors else ""
            )
            + (
                "Multiple split sizes were published in the card metadata; "
                f"using the largest match for '{split}'."
            ),
        )
    return HFSplitSize(
        num_examples=None,
        note=(
            (" | ".join(errors) + " | ") if errors else ""
        )
        + f"Dataset card metadata does not expose num_examples for split '{split}'.",
    )
