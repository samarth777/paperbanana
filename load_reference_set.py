"""
Utility to load the reference set built from MinerU-parsed NeurIPS papers.

Usage:
    from load_reference_set import load_reference_set
    ref_set = load_reference_set()  # loads from reference_set.json
"""
import json
from pathlib import Path
from typing import List, Dict, Any


def load_reference_set(path: str = "data/spotlight_reference_set.json") -> List[Dict[str, Any]]:
    """
    Load reference set from JSON file.
    
    Args:
        path: Path to spotlight_reference_set.json (default) or reference_set.json
        
    Returns:
        List of reference examples compatible with RetrieverAgent:
        [
            {
                "id": "ref_0001",
                "domain": "Computer Vision",
                "diagram_type": "Architecture Diagram",
                "description": "Figure caption text...",
                "image_path": "reference_images/ref_0001_paper_name.jpg",
                "paper_title": "Paper Title",
                "source_file": "paper_stem",
                "page_idx": 2,
                "bbox": [x0, y0, x1, y1]
            },
            ...
        ]
    """
    ref_path = Path(path)
    if not ref_path.exists():
        print(f"Warning: Reference set not found at {ref_path}")
        return []

    with open(ref_path) as f:
        reference_set = json.load(f)

    # Validate image paths exist
    valid_refs = []
    missing = 0
    for ref in reference_set:
        img_path = ref.get("image_path", "")
        if img_path and Path(img_path).exists():
            valid_refs.append(ref)
        else:
            missing += 1

    if missing > 0:
        print(f"Warning: {missing}/{len(reference_set)} reference images not found on disk")

    print(f"Loaded {len(valid_refs)} reference examples from {ref_path}")
    return valid_refs


def get_reference_set_stats(reference_set: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Print and return statistics about the reference set."""
    domains = {}
    diagram_types = {}
    papers = set()

    for ref in reference_set:
        d = ref.get("domain", "Unknown")
        dt = ref.get("diagram_type", "Unknown")
        p = ref.get("source_file", "Unknown")
        domains[d] = domains.get(d, 0) + 1
        diagram_types[dt] = diagram_types.get(dt, 0) + 1
        papers.add(p)

    stats = {
        "total_references": len(reference_set),
        "unique_papers": len(papers),
        "domains": dict(sorted(domains.items(), key=lambda x: -x[1])),
        "diagram_types": dict(sorted(diagram_types.items(), key=lambda x: -x[1])),
    }

    print(f"\nReference Set Statistics:")
    print(f"  Total references: {stats['total_references']}")
    print(f"  Unique papers: {stats['unique_papers']}")
    print(f"  Domains:")
    for d, c in stats["domains"].items():
        print(f"    {d}: {c}")
    print(f"  Diagram Types:")
    for dt, c in stats["diagram_types"].items():
        print(f"    {dt}: {c}")

    return stats


if __name__ == "__main__":
    ref_set = load_reference_set()
    if ref_set:
        get_reference_set_stats(ref_set)
