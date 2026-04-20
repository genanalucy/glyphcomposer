from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from glyphgen.types import GlyphSample
from glyphgen.utils.io import ensure_dir, read_jsonl, write_jsonl


@dataclass(frozen=True, slots=True)
class ManifestBundle:
    train_random: Path
    val_random: Path
    test_random: Path
    train_compositional: Path
    val_compositional: Path
    test_compositional: Path
    oov_extension: Path | None = None


def read_manifest(path: str | Path) -> list[GlyphSample]:
    return [GlyphSample.from_dict(row) for row in read_jsonl(path)]


def _ratio_split(items: list[GlyphSample], seed: int) -> tuple[list[GlyphSample], list[GlyphSample], list[GlyphSample]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total == 0:
        return [], [], []
    train_end = max(1, int(total * 0.8))
    val_end = max(train_end + 1, int(total * 0.9)) if total >= 3 else total
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def _char_holdout_split(
    items: list[GlyphSample], seed: int
) -> tuple[list[GlyphSample], list[GlyphSample], list[GlyphSample]]:
    grouped: dict[str, list[GlyphSample]] = defaultdict(list)
    for item in items:
        grouped[item.target_char].append(item)
    chars = sorted(grouped)
    random.Random(seed).shuffle(chars)
    total = len(chars)
    if total == 0:
        return [], [], []
    train_end = max(1, int(total * 0.8))
    val_end = max(train_end + 1, int(total * 0.9)) if total >= 3 else total
    train_chars = set(chars[:train_end])
    val_chars = set(chars[train_end:val_end])
    test_chars = set(chars[val_end:])
    train = [item for item in items if item.target_char in train_chars]
    val = [item for item in items if item.target_char in val_chars]
    test = [item for item in items if item.target_char in test_chars]
    return train, val, test


def write_manifest_bundle(
    samples: list[GlyphSample],
    output_dir: str | Path,
    *,
    seed: int = 42,
    oov_samples: list[GlyphSample] | None = None,
) -> ManifestBundle:
    manifest_dir = ensure_dir(Path(output_dir) / "manifests")

    train_random, val_random, test_random = _ratio_split(samples, seed)
    train_comp, val_comp, test_comp = _char_holdout_split(samples, seed)

    paths = ManifestBundle(
        train_random=manifest_dir / "train_random.jsonl",
        val_random=manifest_dir / "val_random.jsonl",
        test_random=manifest_dir / "test_random.jsonl",
        train_compositional=manifest_dir / "train_compositional.jsonl",
        val_compositional=manifest_dir / "val_compositional.jsonl",
        test_compositional=manifest_dir / "test_compositional.jsonl",
        oov_extension=manifest_dir / "oov_extension.jsonl" if oov_samples else None,
    )

    write_jsonl(paths.train_random, [item.to_dict() | {"split": "train_random"} for item in train_random])
    write_jsonl(paths.val_random, [item.to_dict() | {"split": "val_random"} for item in val_random])
    write_jsonl(paths.test_random, [item.to_dict() | {"split": "test_random"} for item in test_random])
    write_jsonl(
        paths.train_compositional,
        [item.to_dict() | {"split": "train_compositional"} for item in train_comp],
    )
    write_jsonl(
        paths.val_compositional,
        [item.to_dict() | {"split": "val_compositional"} for item in val_comp],
    )
    write_jsonl(
        paths.test_compositional,
        [item.to_dict() | {"split": "test_compositional"} for item in test_comp],
    )
    if oov_samples and paths.oov_extension:
        write_jsonl(paths.oov_extension, [item.to_dict() | {"split": "oov_extension"} for item in oov_samples])

    return paths

