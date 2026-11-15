"""
One-off build script: scans every Boxer CSV to produce a small
sem_id -> canonical LVIS name lookup, cached at data/boxer_class_vocab.json.

This is the same (sem_id, name) vocabulary already implicitly used by
model.py's `class_embedding` table for env tokens -- Boxer CSVs carry both
columns per detection row, so no separate LVIS taxonomy file is needed. This
script just collapses ~150 videos' worth of rows into one small mapping.
Each sem_id maps 1:1 to a single name string across the whole dataset (no
ambiguity found when this was built).

Re-run only if boxer_obbs/ changes. object_class_vocab.py reads the cached
JSON at import time; nothing else needs to re-scan the CSVs.
"""

import collections
import glob
import json
import os

BOXER_OBBS_ROOT = os.environ.get(
    "BOXER_OBBS_ROOT",
    os.environ.get("HDEPIC_ROOT", "/data/video_datasets/HD-EPIC") + "/boxer_obbs")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "boxer_class_vocab.json")


def build():
    import pandas as pd

    csvs = sorted(glob.glob(os.path.join(BOXER_OBBS_ROOT, "*", "boxer_3dbbs.csv")))
    print(f"scanning {len(csvs)} Boxer CSVs...")
    sem_to_names = collections.defaultdict(collections.Counter)
    for i, path in enumerate(csvs):
        df = pd.read_csv(path, usecols=["name", "sem_id"])
        for name, sem_id in zip(df["name"], df["sem_id"]):
            sem_to_names[int(sem_id)][str(name)] += 1
        if i % 20 == 0:
            print(f"  {i}/{len(csvs)}")

    inconsistent = {sid: c for sid, c in sem_to_names.items() if len(c) > 1}
    if inconsistent:
        print(f"WARNING: {len(inconsistent)} sem_ids have >1 distinct name string "
              f"(taking the most common per id): {list(inconsistent.items())[:5]}")

    vocab = {str(sid): counter.most_common(1)[0][0] for sid, counter in sem_to_names.items()}
    print(f"{len(vocab)} unique classes, sem_id range "
          f"[{min(sem_to_names)}, {max(sem_to_names)}]")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(vocab, f, indent=0, sort_keys=True)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
