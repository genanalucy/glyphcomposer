from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from glyphgen.structures import normalize_structure_label
from glyphgen.types import StructureType


@dataclass(frozen=True, slots=True)
class DecompositionRecord:
    target_char: str
    structure: StructureType
    component_a: str
    component_b: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "DecompositionRecord":
        required = {"target_char", "structure", "component_a", "component_b"}
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Decomposition row is missing columns: {sorted(missing)}")

        target_char = (row["target_char"] or "").strip()
        component_a = (row["component_a"] or "").strip()
        component_b = (row["component_b"] or "").strip()
        structure = normalize_structure_label(row["structure"])
        if not target_char:
            raise ValueError("target_char must not be empty.")
        if not component_a or not component_b:
            raise ValueError(f"{target_char}: component_a and component_b must both be present.")
        return cls(
            target_char=target_char,
            structure=structure,
            component_a=component_a,
            component_b=component_b,
        )

    def to_row(self) -> dict[str, str]:
        return {
            "target_char": self.target_char,
            "structure": self.structure.value,
            "component_a": self.component_a,
            "component_b": self.component_b,
        }


def load_decomposition_csv(path: str | Path) -> list[DecompositionRecord]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [DecompositionRecord.from_row(row) for row in reader]


def dump_decomposition_csv(path: str | Path, records: list[DecompositionRecord]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target_char", "structure", "component_a", "component_b"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def filter_two_component_records(records: list[DecompositionRecord]) -> list[DecompositionRecord]:
    return [
        item
        for item in records
        if len(item.component_a.strip()) > 0 and len(item.component_b.strip()) > 0
    ]

