# Raveform predictor — feedback-loop findings (KA-233 / KA-234)

> **⚠️ 2026-06-01 UPDATE — full-scale clean-fold certification supersedes the
> "both goals met" claim below.** The full 1,391-track set is now local, so the
> model was retrained from scratch and certified on a held-out fold it never saw
> (train folds 0–5 / val 6 / **test 7**). Result: **macro-F1 holds (~0.63) and
> "in-a-drop" detection is solid (~80%), but the 70%-recall goals are NOT met on
> clean data (~58% LB-transition + warning recall).** The 150-subset overstated
> recall (leaky, class-skewed val). Crucially, lowering the onset threshold does
> NOT raise recall (flat 58%) → the missed boundaries are a **model-signal
> ceiling, not a tuning/SM problem**; more data confirmed F1 but did not lift
> recall. See "## Full-scale certification" at the bottom.

Experiments run on the GCP L4 box against the **150-track subset** (features +
tracks only; 120 train / 30 val via `create_splits`). All numbers are on the
30-track held-out val split. **Caveat:** the subset is class-skewed and the val
split overlaps the seeded checkpoint's training data — directional results are
robust, absolute numbers need the full 1,391-track set to certify.

Scripts cross-import by sibling name; run from this dir, e.g.
`uv run python experiments/_unified_eval.py`. Checkpoints/caches live under
`/mnt/scratch/data/loop/` (regenerable; not durable).

## Headline result

A **longer-context sequence model** (per-downbeat CNN window encoder → LSTM
*across* downbeats, sequence-labeled over whole tracks) fixes the core
limitation of the committed model, whose LSTM only saw inside one 8-beat window.

| | committed 8-beat model | seq model | unified (final) |
|---|---|---|---|
| load-bearing macro-F1 | 0.46 | **0.62** | 0.60 |
| buildup F1 | 0.37 | 0.50 | — |
| outro F1 | 0.35 | 0.57 | — |
| drop on/off frame acc | — | 75% | **76%** |
| countdown MAE (final 2 bars) | ~30 (broken) | — | **1.9** |

The seq model beats the baseline on held-out tracks the baseline *trained on*.

## Two goals, both met on val (with caveats)

**Detection** (`_arch.py`, `_drop.py`): seq model + re-tuned State Manager
(correction_threshold 0.4, **sticky_beats 8** not 32, warmup 0 — the baseline's
32-beat sticky hold over-suppressed the smoother model). LB recall **70.4%**,
matched cueing latency **3.5 beats**, macro-F1 52→62.

**Prediction/anticipation** (`_antic_train.py`, `_antic_sm.py`): the broken
`beats_until` head (PhraseLoss weighted regression at 0.01) → retrain with
**w_reg≈1.0 + capped target**. Countdown MAE 30→**~2 beats**; a
**monotone-clamped countdown** gives **100% monotonicity**; drop flagged ≥1 bar
early on ~60–71% (recall/precision trade via consensus threshold).

**Unified model** (`_unified.py`): a **dedicated, non-detached beats branch**
(deeper MLP, w_reg 0.3) gives ONE model good at both — F1 0.60 + countdown MAE
1.9 — resolving the detection-vs-countdown tension. This is the production
candidate.

## Key metric correction

`evaluate-pipeline`'s old `mean_timing_error` (~30 beats, the number that drove
"timing is broken" across KA-233) is an **artifact** — it averaged over all
fires incl. mid-phrase false fires. The real cueing latency (`matched_latency`,
boundary→nearest fire on detected boundaries) is **~3–4 beats (<1 bar)**. The
binding constraints are **recall and false-fire rate, not timing**. Fixed in
committed `evaluate.py` (commit be33bb0).

## Negative results (ruled out)

- **Vocab merge** (altoutro+end→outro): HURTS recall (`end` is undetectable,
  pooling it drags outro). Keep 10-class. (`_merge.py`)
- **Context-gating the drop warning** (require recent buildup / not-intro):
  no help; buildup-gating hurts recall (buildup itself unreliable). (`_grind.py`)
- **Detached beats branch**: worse than a lightly-coupled dedicated branch.

## VJ-usability snapshot (unified model)

- drop on/off state: ~76% accurate frame-to-frame.
- drop START: ~57% recall, ~1-bar latency, ~37% precision.
- drop END / buildup: weaker (data-limited).
- pre-drop warning: ~57–71% recall, over-eager (low precision).
- offline timeline tool (`_timeline.py`) writes per-track event JSONs
  (`drop_incoming`/`drop_start`/`drop_end`/`buildup`, times in seconds →
  `<id>.wav`) for ear-checking; 30 in `gs://<bucket>/timelines/`.

## Next

1. **Full-scale features** — the one real unlock for drop-end + buildup
   reliability and for the countdown to time its start for far drops.
2. **Productionize** — promote `UnifiedSeq` + sequence training into the
   package; add streaming/stateful inference + the debounced drop/buildup state
   machine + `drop_incoming/start/end/buildup` OSC events to `run-live`
   (needs audio hardware to test end-to-end).

## Full-scale certification (2026-06-01)

The full **1,391-track** set is now staged locally (`/mnt/scratch/data`: audio +
**pre-computed features** + labels, all present; 47GB features). Driver:
`experiments/_full.py` (RAM-safe — 47GB features can't fit in 31GB RAM, so the
batch=1 seq model lazy-loads each track's windows from NVMe in the loop; eval
folds ~174 tracks fit eagerly). Split uses Raveform's official 8-fold field:
**train = folds 0–5 (1043), val = fold 6 (174, selection), test = fold 7 (174,
HELD OUT, scored once)**. This is the first leakage-free number — fixes the core
caveat (150-subset val overlapped the seed's training data).

`UnifiedSeq` (FINDINGS winner config: detach=0, w_reg=0.3, cap=12), 40 epochs,
~52s/epoch, best by val mF1+countdown saved at ep7 (val loss climbs after →
mild overfit). Held-out **test fold (fold 7)**:

| metric | clean test | goal | leaky 150-subset (prior) |
|---|---|---|---|
| LB macro-F1 | **0.628** | don't regress ✓ | 0.62 |
| "in-a-drop" frame (acc / recall) | **80% / 75%** | — | ~76% |
| LB **transition** recall (≤2 bars) | **~58%** | ≥70% ✗ | "70.4%" |
| pre-drop warning (≥1 bar early) | **58%** | ≥70% ✗ | ~61% |
| countdown MAE (final 2 bars) | **3.4b** | ≤4b ✓ | ~2b |
| countdown monotonicity | **100%** | ≥90% ✓ | 100% |
| per-class recall (frame) | intro 79 / drop 76 / **buildup 46 / outro 46** | ≥70 | — |

**Key finding — recall is a model-signal ceiling, not tuning.** The onset
detector's LB recall is **flat at ~58% as the threshold drops 0.40→0.25**
(`_full_sm.py`): the missing ~42% of boundaries never cross even 0.25 prob for
the LB class — the model doesn't surface them at all, so no threshold/SM tuning
recovers them. The State Manager configs that "hit" 77–99% recall are the same
inflation artifact as the old `mean_timing_error`: **18% precision, 9k–24k fires
across ~39k downbeats** (firing a third of all downbeats). Not usable. Onset@0.4
is the honest usable operating point: **58% recall, 49% precision, ~1-bar
latency**.

**Implication:** the earlier "both goals met" was an artifact of the leaky,
class-skewed subset. With the real data: F1 generalizes (~0.63, no overfit gap),
the countdown is genuinely good, "are we in a drop" is solid (~80%, usable for
sustained visual state) — but rare-boundary recall (buildup/outro/drop-onset/
warning) sits ~46–58%. More data did not lift it → the next lever is
architecture/signal or rare-class learning, NOT data volume or SM tuning.

**In flight:** `train_v2` (`_full.py`) — rare-class reweight (wp 0.5→0.75,
cw_cap 5→8) + anti-overfit (dropout 0.3→0.4, wd 1e-4→3e-4) → `seq_unified_full_v2`,
testing whether the model can be pushed to actually surface buildup/outro.

## Recall-ceiling verdict (2026-06-01) — goal's "prove it / lock it" branch

Goal: lift LB transition recall ~58%→≥70% at ≤2-bar timing & ≥45% fire-precision,
or PROVE it's a hard ceiling and lock the best honest operating point. Verdict:
**70% at the strict 2-bar tolerance is not reachable on this data — proven — but
the binding limiter is TIMING PRECISION + label quantization, not detection
blindness.**

Evidence (all on held-out test fold 7, `_cue.py` / `_full_sm.py` / `_bidir.py`):
1. **Reweight (v2):** lifted buildup *frame* recall 46→61 and F1 0.63→0.65, but
   LB *transition* recall stayed ~58%. Frame ≠ cueing.
2. **Threshold:** flat — onset LB recall 55.7%→58.0% across 0.40→0.25.
3. **Cueing method:** onset (prob rising-edge) ≈ debounced argmax-flip (~54-58%);
   not a measurement-tool artifact.
4. **Full future context (bidir, the upper bound):** does NOT beat causal —
   transition recall ~50% (it *under-segments*: 825 fires vs causal 1350, higher
   precision 64%). Future context helps *frame* F1 (0.63→0.70) and *outro* frame
   recall (48→64) — outro is genuinely causality-limited — but it does not raise
   transition recall. So a better/bidir model can't close the 2-bar gap.
5. **Tolerance sweep (causal v2, onset@0.3):** recall climbs 27%(1bar) →
   **58%(2bar)** → 64%(3bar) → 67%(4bar) → **72%(6bar)**. The model fires near
   ~72% of boundaries within 6 bars — the misses are mostly **near-misses with
   2-4 bar timing scatter**, not blindness. Labels are quantized to clean 32/64
   positions, so part of the "error" is likely label-offset vs the actual audio
   change (best confirmed by ear-checking the timeline tool).

**Per-class:** drops are caught tightest (69% @2-bar, 77% @6-bar); outro/intro
have the most scatter. No single model wins all classes (causal best on drop,
bidir best on outro).

**LOCKED best honest operating point (live, causal, deployable) — CERTIFIED FLOOR:**
`seq_unified_full_v2.safetensors`, cue = onset on current-phrase LB probs @0.30:
**LB transition recall 58% @2-bar / 64% @3-bar, ~50% fire-precision, matched
latency ~1 bar; in-drop frame 80% acc / 75% recall; countdown MAE 3.2b, 100%
monotone.** This is the production cueing config. These are the certified
held-out numbers — the floor; the operating point is locked here.

**Label-offset upside — UNCONFIRMED (do not bank on it).** Hypothesis: the
2-bar misses are mostly label-quantization offset (true recall ~72% @6-bar). A
quick feature-derived novelty check (`_validate.py`) was **inconclusive**: model
fires align with mel-novelty peaks 73% vs labels 61% (mildly supportive), BUT
absolute novelty at boundaries was below random and both sit ~1.8 bars from any
peak — the overlapping-8-beat-window novelty is too crude to settle it. Proper
validation = human ear-check via `_timeline.py`, or raw-audio structural novelty
(checkerboard kernel on a self-similarity matrix / librosa onset envelope). Until
then the locked number stays at the conservative 58% @2-bar floor.

**Levers ruled out** for 2-bar recall: thresholds, cueing method, rare-class
reweight, full future context (bidir), absolute phrase-grid features (killed —
not live-deployable). **Untried but argued-moot:** wider context window — bidir
already has unbounded temporal context and doesn't help, so wider local audio is
very unlikely to. **Real remaining lever** (different problem): sharpen the
transition *timing* (dedicated change-point head) — but likely label-limited;
validate against ear-check before investing. Trick/fake-out robustness: not in
the data (idealized labels) → needs Kyle's own labeled tracks or simulation +
the inference-time energy-confirmation gate (see memory).

## Productionized into the package (2026-06-01)

The winning seq model + locked operating point are now wired into `run-live`
(previously it ran the old 8-beat stateless PhrasePredictor):
- `model.py`: added `UnifiedSeqPredictor` (causal cross-downbeat LSTM) with a
  `step()` method for stateful streaming. Verified bit-exact vs the offline
  full-sequence forward — 100% argmax agreement, |Δbeats| 3e-5
  (`experiments/_steptest.py`); the ~1e-3 logit drift is cuDNN
  full-seq-vs-stepped numerics, decision-irrelevant.
- `live/inference.py`: `SeqInferenceEngine` — stateful, carries the LSTM hidden
  state across downbeats; `PredictionResult` gained `current_probs` for onset
  cueing.
- `live/cue.py`: `OnsetCueTracker` — the locked onset@0.30 cueing → `transition`
  / `drop_start` / `drop_end` / `buildup` events. The State Manager is kept ONLY
  for its (good) mechanical countdown / `anticipate` (drop-incoming); its
  consensus-transition core spams this model so we don't emit it.
- `live/pipeline.py`, `live/osc.py`, `cli.py`: wired through; `run-live` now
  defaults to `data/models/seq_unified.safetensors` + an `--onset-threshold`
  flag; OSC emits the new event kinds.
- New `audiovj evaluate-seq [--fold 7]` command = offline twin of the live path
  (same seq inference + cue components). On fold 7 it reproduces the locked
  numbers exactly: **LB transition recall 57.7%, prec 50.6%, lat 4.3b, per-class
  intro 38 / buildup 50 / drop 69 / outro 32, 574 drop_start / 557 drop_end.**

Production checkpoint `data/models/seq_unified.safetensors` (copied from
`loop/seq_unified_full_v2`) is backed up to `gs://<bucket>/models/`.
NOT yet tested: live audio capture + Carabiner/Ableton-Link beat sync + OSC
transport (needs hardware — Kyle's Mac). Everything that *decides* (model,
windowing, statefulness, cueing) is verified offline.
