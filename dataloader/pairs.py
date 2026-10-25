"""
Pre/Post Pair Index for SpaceNet 8.

Reads every ``*_label_image_mapping.csv`` under the dataset root so that a
pre-event image can be resolved to its post-event image(s) automatically.
Used by the API to auto-select the post image when a pre image is chosen.
"""

import csv
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def load_alignment_stats(path: Union[str, Path]) -> dict[tuple[str, str, int], float]:
    """
    Load residual misalignment per (aoi, tile, post_index) from the CSV
    written by scripts/analyze_alignment.py.

    Returns:
        Mapping to the residual shift in pixels (NaN when not estimable).
    """
    path = Path(path)
    stats = {}
    if not path.exists():
        return stats
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            try:
                stats[(row["aoi"], row["tile"], int(row["post_index"]))] = float(row["shift_px"])
            except (KeyError, ValueError):
                continue
    return stats


class PairIndex:
    """
    Index of all pre/post image pairs found in the dataset mapping CSVs.

    When alignment statistics are available, the post images of each pair
    are ordered best-aligned first, so the default selection is the post
    image with the smallest residual pre/post misalignment.

    Args:
        data_root: Root directory containing the AOI folders
            (e.g. ./CP1-DATASET/Germany_Training_Public/...).
        alignment_stats: Optional CSV from scripts/analyze_alignment.py.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        alignment_stats: Optional[Union[str, Path]] = None,
    ):
        self.data_root = Path(data_root)
        self.pairs: list[dict] = []
        self._by_pre: dict[str, dict] = {}
        self._shifts = load_alignment_stats(alignment_stats) if alignment_stats else {}
        self._build()

    def _build(self):
        if not self.data_root.exists():
            logger.warning(f"Dataset root not found: {self.data_root}")
            return

        for mapping_csv in sorted(self.data_root.glob("*/*_label_image_mapping.csv")):
            aoi_dir = mapping_csv.parent
            has_labels = (aoi_dir / "annotations").exists()

            with open(mapping_csv, "r") as f:
                for row in csv.DictReader(f):
                    pre = (row.get("pre-event image") or "").strip()
                    if not pre:
                        continue
                    label = (row.get("label") or "").strip()
                    tile = label.replace(".geojson", "")
                    posts = []
                    for idx, key in enumerate(("post-event image 1", "post-event image 2"), start=1):
                        name = (row.get(key) or "").strip()
                        if name:
                            shift = self._shifts.get((aoi_dir.name, tile, idx))
                            posts.append({
                                "name": name,
                                "shift_px": None if shift is None or shift != shift else round(shift, 1),
                            })
                    # Best-aligned first; unknown shifts keep CSV order after known ones
                    posts.sort(key=lambda p: (p["shift_px"] is None, p["shift_px"] or 0.0))

                    entry = {
                        "aoi": aoi_dir.name,
                        "label": label,
                        "pre_image": pre,
                        "post_images": [p["name"] for p in posts],
                        "post_shift_px": [p["shift_px"] for p in posts],
                        "has_labels": has_labels,
                    }
                    self.pairs.append(entry)
                    self._by_pre[pre.lower()] = entry

        logger.info(f"Pair index: {len(self.pairs)} pre/post pairs from {self.data_root}")

    def aois(self) -> list[dict]:
        """List AOIs with their pair counts."""
        summary: dict[str, dict] = {}
        for p in self.pairs:
            s = summary.setdefault(
                p["aoi"], {"aoi": p["aoi"], "num_pairs": 0, "has_labels": p["has_labels"]}
            )
            s["num_pairs"] += 1
        return list(summary.values())

    def search(
        self,
        aoi: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """
        Filter pairs by AOI and a substring of the pre image or tile name.

        Returns:
            Tuple of (page of matching pairs, total number of matches).
        """
        q = (query or "").strip().lower()
        matches = [
            p for p in self.pairs
            if (aoi is None or p["aoi"] == aoi)
            and (not q or q in p["pre_image"].lower() or q in p["label"].lower())
        ]
        return matches[offset:offset + limit], len(matches)

    def lookup(self, pre_image: str) -> Optional[dict]:
        """Find the pair for a pre image filename (path components ignored)."""
        return self._by_pre.get(Path(pre_image).name.lower())

    def annotation_paths(self, entry: dict) -> Optional[tuple[Path, Optional[Path]]]:
        """
        Locate a labelled pair's annotation GeoJSON and AOI reference CSV.

        Returns:
            (geojson_path, reference_csv_path or None), or None if unlabelled.
        """
        if not entry.get("has_labels") or not entry.get("label"):
            return None
        aoi_dir = self.data_root / entry["aoi"]
        geojson = aoi_dir / "annotations" / entry["label"]
        if not geojson.exists():
            return None
        ref = next(iter(aoi_dir.glob("*_reference.csv")), None)
        return geojson, ref

    def resolve(self, aoi: str, kind: str, name: str) -> Optional[Path]:
        """
        Resolve an image filename to its path on disk.

        Only names that appear in the mapping CSVs are resolvable, which
        prevents arbitrary file access through the API.

        Args:
            aoi: AOI folder name.
            kind: 'pre' or 'post'.
            name: Image filename.

        Returns:
            Path to the GeoTIFF, or None if it is not part of the index.
        """
        entry = self.lookup(name) if kind == "pre" else None
        if kind == "post":
            entry = next(
                (p for p in self.pairs if p["aoi"] == aoi and name in p["post_images"]),
                None,
            )
        if entry is None or entry["aoi"] != aoi:
            return None

        folder = "PRE-event" if kind == "pre" else "POST-event"
        path = self.data_root / aoi / folder / (entry["pre_image"] if kind == "pre" else name)
        return path if path.exists() else None
