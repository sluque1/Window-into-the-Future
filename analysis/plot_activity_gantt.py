"""
Per-participant horizontal stacked bar ("Gantt") chart of high-level activity
timelines: one bar per video, one stacked segment per activity row in
<...>/high-level/activities/<pid>_recipe_timestamps.csv. Segment length =
end_time - start_time (an "end" end_time is resolved against the video's
duration in Videos/<pid>/durations.txt). Segment color = recipe_id (a fixed
color per recipe across that participant's videos); rows with no recipe_id
are colored black. Each bar is annotated with the % of the video covered by
colored (non-black, i.e. recipe-tagged) segments.

Videos are ordered top-to-bottom by that video's mean per-episode PR-AUC
(best first) from the "_traj" LOOCV checkpoints -- see
plot_pr_auc_vs_video_length.py, which this reuses for the model side. Videos
with no scored test episodes (e.g. missing Boxer/SLAM data, or no valid
anchors) sort to the bottom, labeled "no episodes".

Writes one PNG per participant: analysis/out/activity_gantt_<PID>.png

Usage:
  python analysis/plot_activity_gantt.py
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pr_auc_vs_video_length import collect_participant_video_pr_auc  # noqa: E402

ACTIVITIES_DIR = os.environ.get(
    "ACTIVITIES_DIR",
    os.environ.get("HDEPIC_ROOT", "/data/video_datasets/HD-EPIC")
    + "/hd-epic-annotations/high-level/activities")
VIDEOS_DIR = os.environ.get(
    "VIDEOS_DIR", os.environ.get("HDEPIC_ROOT", "/data/video_datasets/HD-EPIC") + "/Videos")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")

PARTICIPANTS = [f"P{i:02d}" for i in range(1, 10)]

# 8-hue categorical order (validated for adjacent-pair CVD safety); cycles if
# a participant somehow has more than 8 distinct recipes.
CATEGORICAL = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
              "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
NO_RECIPE_COLOR = "#000000"


def load_durations(pid):
    """video_id -> duration in seconds, from Videos/<pid>/durations.txt lines
    like 'P01-20240202-110250.mp4 396.266667'."""
    path = os.path.join(VIDEOS_DIR, pid, "durations.txt")
    durations = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, dur = line.split()
            video_id = name[:-4] if name.endswith(".mp4") else name
            durations[video_id] = float(dur)
    return durations


def load_activities(pid):
    """video_id -> ordered list of {'recipe_id','high_level_activity_label',
    'start_time','end_time'} rows, in CSV appearance order."""
    path = os.path.join(ACTIVITIES_DIR, f"{pid}_recipe_timestamps.csv")
    rows_by_video = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows_by_video.setdefault(row["video_id"], []).append(row)
    return rows_by_video


def _video_duration(video_id, durations):
    """durations[video_id], tolerating the occasional annotation typo where the
    date/time component drops a leading zero (e.g. 'P09-2024623-154547' for
    'P09-20240623-154547', confirmed against durations.txt)."""
    if video_id in durations:
        return durations[video_id]
    parts = video_id.split("-")
    if len(parts) == 3:
        pid, date, time = parts
        # date is YYYYMMDD (8 digits); a 7-digit date lost the month's leading
        # zero (e.g. 'YYYY' + '6' + 'DD' instead of 'YYYY' + '06' + 'DD') --
        # the day is always 2 digits, so re-zero-pad the month in the middle.
        if len(date) == 7:
            date = date[:4] + date[4].zfill(2) + date[5:]
        padded = f"{pid}-{date.zfill(8)}-{time.zfill(6)}"
        if padded in durations:
            return durations[padded]
    raise KeyError(f"{video_id} not found in durations.txt (even after zero-pad fallback)")


def segment_span(row, durations):
    start = float(row["start_time"])
    end_raw = row["end_time"].strip()
    end = _video_duration(row["video_id"], durations) if end_raw.lower() == "end" else float(end_raw)
    return start, end


def _ordered_video_ids(video_ids, video_pr_auc):
    """Best mean PR-AUC first; videos with no scored episodes sort last, alphabetically."""
    scored = sorted((v for v in video_ids if v in video_pr_auc),
                    key=lambda v: video_pr_auc[v], reverse=True)
    unscored = sorted(v for v in video_ids if v not in video_pr_auc)
    return scored + unscored


def plot_participant(pid):
    durations = load_durations(pid)
    rows_by_video = load_activities(pid)
    video_pr_auc = collect_participant_video_pr_auc(pid)
    video_ids = _ordered_video_ids(sorted(rows_by_video), video_pr_auc)

    recipe_ids = sorted({row["recipe_id"].strip() for rows in rows_by_video.values()
                         for row in rows if row["recipe_id"].strip()})
    color_map = {rid: CATEGORICAL[i % len(CATEGORICAL)] for i, rid in enumerate(recipe_ids)}

    fig_height = max(3.0, 0.32 * len(video_ids) + 1.5)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    tick_labels = []
    max_duration = 0.0
    for y, video_id in enumerate(video_ids):
        total = _video_duration(video_id, durations)
        max_duration = max(max_duration, total)
        colored = 0.0
        for row in rows_by_video[video_id]:
            start, end = segment_span(row, durations)
            rid = row["recipe_id"].strip()
            color = color_map[rid] if rid else NO_RECIPE_COLOR
            if rid:
                colored += end - start
            ax.barh(y, end - start, left=start, color=color, height=0.7,
                    edgecolor="white", linewidth=0.4, zorder=3)

        pct = 100.0 * colored / total if total > 0 else 0.0
        ax.text(total, y, f"  {pct:.0f}% covered", va="center", ha="left",
                fontsize=6.5, color="#52514e", zorder=4)

        if video_id in video_pr_auc:
            tick_labels.append(f"{video_id}  (PR-AUC {video_pr_auc[video_id]:.2f})")
        else:
            tick_labels.append(f"{video_id}  (no episodes)")

    ax.set_xlim(0, max_duration * 1.22)
    ax.set_yticks(range(len(video_ids)))
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Time in video (s)")
    ax.set_title(f"{pid}: activity timeline per video, ordered by mean episode PR-AUC "
                f"(color = recipe_id)")
    ax.xaxis.grid(True, linestyle=":", color="#e1e0d9", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
    labels = list(color_map.keys())
    handles.append(plt.Rectangle((0, 0), 1, 1, color=NO_RECIPE_COLOR))
    labels.append("(none)")
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
             frameon=False, title="recipe_id", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"activity_gantt_{pid}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  ({len(video_ids)} videos, {len(recipe_ids)} recipes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=",".join(PARTICIPANTS))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    for pid in args.participants.split(","):
        plot_participant(pid)


if __name__ == "__main__":
    main()
