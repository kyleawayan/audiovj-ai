"""Are the Raveform training labels a bar late?

dataset.py generate_labels assigns each downbeat the latest cue with
``cue_time <= t``. Section starts come from structures/segments.json; downbeats
come from a SEPARATE file, structures/beats/<key>.beat.csv. If a section start
lands even microseconds AFTER its own downbeat, the <= test fails there and the
label first appears at the NEXT downbeat -- a full bar (4 beats) late.

A model trained on labels that are sometimes a bar late would learn to fire a
bar late. That is exactly the observed defect, so this is worth ruling in or out
before anyone considers retraining. Needs only raveform.zip (479MB, local) --
not the 149GB corpus.

Usage: uv run python experiments/_labelalign.py <path-to-raveform.zip>
"""
import csv, io, json, statistics, sys, zipfile
from collections import Counter

TOL_EXACT = 0.001  # 1 ms -> "same instant, float noise apart"

z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
seg_name = next(n for n in names if n.endswith("structures/segments.json"))
tracks = json.loads(z.read(seg_name))
beat_paths = {n.rsplit("/", 1)[-1].replace(".beat.csv", ""): n
              for n in names if n.endswith(".beat.csv")}
print(f"tracks in segments.json: {len(tracks)}   beat files: {len(beat_paths)}")

offs_all, offs_drop = [], []
late_all = late_drop = tot_all = tot_drop = 0
near_miss_drop = 0          # start is after its downbeat by < 1 ms
per_label_late = Counter(); per_label_tot = Counter()
n_done = 0

for tr in tracks:
    key = tr["key"]
    p = beat_paths.get(key)
    if not p:
        continue
    rows = list(csv.DictReader(io.StringIO(z.read(p).decode())))
    db = [float(r["time"]) for r in rows if r["downbeat"] == "1"]
    if len(db) < 8:
        continue
    n_done += 1
    bpm = float(tr.get("average_bpm") or 0) or 128.0
    beat = 60.0 / bpm
    for s in tr["sections"]:
        st = float(s["start"])
        nearest = min(db, key=lambda d: abs(d - st))
        off = st - nearest                     # + = start is AFTER its downbeat
        # Does the <= test admit this cue at its own downbeat?
        is_late = st > nearest
        offs_all.append(off); tot_all += 1; per_label_tot[s["name"]] += 1
        if is_late:
            late_all += 1; per_label_late[s["name"]] += 1
        if s["name"] == "drop":
            offs_drop.append(off); tot_drop += 1
            if is_late:
                late_drop += 1
                if off < TOL_EXACT:
                    near_miss_drop += 1

def q(xs, p):
    s = sorted(xs); return s[min(int(p * (len(s) - 1)), len(s) - 1)]

print(f"tracks processed: {n_done}\n")
print("signed offset: section start - nearest downbeat  (+ = start lands AFTER)")
for lab, v in (("all sections", offs_all), ("drop sections", offs_drop)):
    print(f"  {lab:<15} n={len(v):5d}  median {statistics.median(v)*1000:+8.3f} ms"
          f"   p05 {q(v,0.05)*1000:+8.2f}   p95 {q(v,0.95)*1000:+8.2f}"
          f"   |median| {abs(statistics.median(v))*1000:6.3f} ms")

print(f"\nsections whose start is AFTER its own downbeat (=> label pushed one bar late):")
print(f"  all sections : {late_all}/{tot_all} = {late_all/max(tot_all,1)*100:.1f}%")
print(f"  drop sections: {late_drop}/{tot_drop} = {late_drop/max(tot_drop,1)*100:.1f}%")
print(f"  of those drops, within 1 ms of the downbeat (pure float race): {near_miss_drop}"
      f" ({near_miss_drop/max(late_drop,1)*100:.1f}% of the late ones)")

print("\nper-label share pushed a bar late:")
for lab in sorted(per_label_tot, key=lambda k: -per_label_tot[k]):
    t = per_label_tot[lab]
    print(f"  {lab:<12} {per_label_late[lab]:5d}/{t:5d} = {per_label_late[lab]/t*100:5.1f}%")

big = [o for o in offs_drop if abs(o) > 0.05]
print(f"\ndrop starts more than 50 ms off ANY downbeat: {len(big)}/{len(offs_drop)}"
      f" = {len(big)/max(len(offs_drop),1)*100:.1f}%  (genuinely off-grid, not a float race)")
