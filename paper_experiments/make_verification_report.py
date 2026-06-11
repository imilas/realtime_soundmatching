"""
Generate a standalone HTML verification report for one search method.

Compares NPSK P-loss rankings (computed from experiment results) against
published rankings from the IEEE 2025 and ISMIR in-domain papers.

Usage:
    python paper_experiments/make_verification_report.py                 # GD
    python paper_experiments/make_verification_report.py --method CMA-ES
    python paper_experiments/make_verification_report.py --method RandomSearch
    python paper_experiments/make_verification_report.py --out my_report.html
"""

from __future__ import annotations
import argparse, pickle, sys, re
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = Path(__file__).parent / "results"

METHODS = ["GD", "CMA-ES", "RandomSearch", "CMA-ES-evosax", "LES"]

# ---------------------------------------------------------------------------
# Published ranks — P-Loss (NPSK) column
# ---------------------------------------------------------------------------

IEEE_PUBLISHED = {
    "bandpass_noise_v1": {"SIMSE_Spec": 1, "L1_Spec": 1, "JTFS": 4, "DTW_Envelope": 3},
    "add_sinesaw":       {"SIMSE_Spec": 4, "L1_Spec": 2, "JTFS": 1, "DTW_Envelope": 3},
    "am_noise":          {"SIMSE_Spec": 4, "L1_Spec": 2, "JTFS": 3, "DTW_Envelope": 1},
    "sine_mod_saw":      {"SIMSE_Spec": 2, "L1_Spec": 3, "JTFS": 4, "DTW_Envelope": 1},
}

# chirplet_pulse excluded — only the "no delay" variant is in scope.
ISMIR_PUBLISHED = {
    "chirplet": {"SIMSE_Spec": 3, "L1_Spec": 1, "JTFS": 1, "DTW_Envelope": 4},
}

PUBLISHED = {**IEEE_PUBLISHED, **ISMIR_PUBLISHED}

SYNTH_LABELS = {
    "bandpass_noise_v1": "BP-Noise",
    "add_sinesaw":       "Add-SineSaw",
    "am_noise":          "Noise-AM",
    "sine_mod_saw":      "SineSaw-AM",
    "chirplet":          "Chirp (no delay)",
}
SYNTH_PAPER = {s: "IEEE" for s in IEEE_PUBLISHED} | {s: "ISMIR" for s in ISMIR_PUBLISHED}
LOSSES = ["SIMSE_Spec", "L1_Spec", "JTFS", "DTW_Envelope"]
SYNTHS = list(PUBLISHED)
RANK_COLORS = {1: "#27ae60", 2: "#f39c12", 3: "#e67e22", 4: "#e74c3c"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _slug(loss: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", loss).strip("_")


def load_trials(synth: str, loss: str, method: str) -> list[dict]:
    path = RES / f"{synth}_{_slug(loss)}_{method}.pkl"
    if not path.exists():
        return []
    try:
        return pickle.load(open(path, "rb")).get("trials", [])
    except Exception:
        return []


def extract_scores(trials: list[dict], method: str) -> tuple[list[float], list[float]]:
    """Returns (returned_p_loss, best_p_loss) for every trial.

    returned_p_loss:
      GD          — final gradient step (matches paper's methodology)
      Black-box   — P-loss at the step with the lowest audio loss
                    (the candidate the method would actually return)
    best_p_loss:
      All methods — minimum P-loss ever visited (oracle upper bound)
    """
    returned, bests = [], []
    for t in trials:
        pl = np.asarray(t.get("history_p_loss", []), dtype=float)
        if len(pl) == 0:
            continue
        best = float(np.nanmin(pl))
        if np.isfinite(best):
            bests.append(best)
        if method == "GD":
            ret = float(pl[-1])
        else:
            al = np.asarray(t.get("history_audio_loss", []), dtype=float)
            ret = float(pl[int(np.nanargmin(al))]) if len(al) else float(pl[-1])
        if np.isfinite(ret):
            returned.append(ret)
    return returned, bests

# ---------------------------------------------------------------------------
# NPSK
# ---------------------------------------------------------------------------

def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    count = sum(np.sum(ai > b) - np.sum(ai < b) for ai in a)
    return count / (len(a) * len(b))


def npsk_ranks(groups: dict[str, list[float]],
               alpha: float = 0.05, min_delta: float = 0.147) -> dict[str, int]:
    valid = {k: np.asarray(v, float) for k, v in groups.items() if len(v) >= 2}
    valid = {k: v[np.isfinite(v)] for k, v in valid.items() if len(v[np.isfinite(v)]) >= 2}
    if len(valid) < 2:
        return {k: 1 for k in valid}
    sorted_names = sorted(valid, key=lambda g: float(valid[g].mean()))

    def split(names):
        if len(names) <= 1:
            return [list(names)]
        all_v = np.concatenate([valid[n] for n in names])
        gm = all_v.mean()
        best_bss, best_i = -1.0, 1
        for i in range(1, len(names)):
            lv = np.concatenate([valid[n] for n in names[:i]])
            rv = np.concatenate([valid[n] for n in names[i:]])
            bss = (len(lv) * (lv.mean() - gm) ** 2
                   + len(rv) * (rv.mean() - gm) ** 2)
            if bss > best_bss:
                best_bss, best_i = bss, i
        lv = np.concatenate([valid[n] for n in names[:best_i]])
        rv = np.concatenate([valid[n] for n in names[best_i:]])
        _, p = mannwhitneyu(lv, rv, alternative="two-sided")
        d = _cliffs_delta(lv, rv)
        if p < alpha and abs(d) >= min_delta:
            return split(names[:best_i]) + split(names[best_i:])
        return [list(names)]

    partitions = split(sorted_names)
    return {n: r for r, grp in enumerate(partitions, 1) for n in grp}


def bootstrap_medians(scores: list[float], n: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores, float)
    return np.array([np.median(rng.choice(arr, size=len(arr), replace=True))
                     for _ in range(n)])

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def rank_badge(rank, muted=False):
    if not isinstance(rank, int):
        return '<span class="badge badge-na">—</span>'
    color = RANK_COLORS.get(rank, "#95a5a6")
    if muted:
        color = "#bdc3c7"
    return f'<span class="badge" style="background:{color}">{rank}</span>'


def match_cell(pub: int, comp, highlight_rank1: bool = False) -> str:
    r1 = ' class="rank1"' if (pub == 1 and highlight_rank1) else ''
    if not isinstance(comp, int):
        return f'<td{r1} class="cell-na">—<br><small>no data</small></td>'
    if pub == comp:
        return f'<td{r1} class="cell-match">{rank_badge(pub)} ✓</td>'
    return f'<td{r1} class="cell-mismatch">{rank_badge(comp)} ✗ <small>(pub={pub})</small></td>'

# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

def main(out_path: Path, method: str) -> None:
    is_gd = (method == "GD")
    returned_label = "Final P-loss<br><small>step 200</small>" if is_gd else "Returned P-loss<br><small>best-audio step</small>"

    # Collect all data
    data: dict[str, dict[str, dict]] = {}
    for synth in SYNTHS:
        data[synth] = {}
        for loss in LOSSES:
            trials = load_trials(synth, loss, method)
            returned, bests = extract_scores(trials, method)
            n = len(trials)
            data[synth][loss] = dict(n=n, returned=returned, bests=bests)

    # Compute ranks on both returned and best P-loss
    ret_ranks: dict[str, dict[str, int | str]] = {}
    best_ranks: dict[str, dict[str, int | str]] = {}
    for synth in SYNTHS:
        rg, bg = {}, {}
        for loss in LOSSES:
            r, b = data[synth][loss]["returned"], data[synth][loss]["bests"]
            if len(r) >= 2:
                rg[loss] = bootstrap_medians(r)
            if len(b) >= 2:
                bg[loss] = bootstrap_medians(b)
        rr = npsk_ranks(rg) if len(rg) >= 2 else {}
        br = npsk_ranks(bg) if len(bg) >= 2 else {}
        ret_ranks[synth] = {l: rr.get(l, "—") for l in LOSSES}
        best_ranks[synth] = {l: br.get(l, "—") for l in LOSSES}

    # Summary counts (rank-1 match is the headline metric)
    n_match = n_miss = n_fail = 0
    n_rank1_pub = n_rank1_match = 0
    for synth in SYNTHS:
        for loss in LOSSES:
            pub = PUBLISHED[synth][loss]
            comp = ret_ranks[synth].get(loss, "—")
            if not isinstance(comp, int):
                n_miss += 1
            elif pub == comp:
                n_match += 1
            else:
                n_fail += 1
            if pub == 1:
                n_rank1_pub += 1
                if isinstance(comp, int) and comp == 1:
                    n_rank1_match += 1
    n_total = len(SYNTHS) * len(LOSSES)

    total_trials = sum(data[s][l]["n"] for s in SYNTHS for l in LOSSES)

    css = """
body{font-family:'Segoe UI',Arial,sans-serif;margin:40px auto;max-width:1200px;
     color:#2c3e50;background:#f8f9fa;padding:0 20px}
h1{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:8px}
h2{color:#34495e;margin-top:40px}
h3{color:#555;margin-top:24px}
.summary-box{display:flex;gap:16px;margin:20px 0;flex-wrap:wrap}
.stat{background:white;border-radius:8px;padding:14px 20px;
      box-shadow:0 2px 6px rgba(0,0,0,.1);text-align:center;flex:1;min-width:120px}
.stat .num{font-size:2em;font-weight:bold}
.stat .lbl{font-size:.82em;color:#7f8c8d;margin-top:4px}
.green .num{color:#27ae60} .amber .num{color:#e67e22}
.red .num{color:#e74c3c}   .blue .num{color:#2980b9}
table{border-collapse:collapse;width:100%;background:white;border-radius:8px;
      overflow:hidden;box-shadow:0 2px 6px rgba(0,0,0,.1);margin-bottom:24px}
th{background:#2c3e50;color:white;padding:9px 12px;text-align:center;font-size:.85em}
th.left{text-align:left}
td{padding:8px 12px;border-bottom:1px solid #ecf0f1;text-align:center;font-size:.85em}
td.left{text-align:left;font-weight:600}
tr:last-child td{border-bottom:none}
tr:hover td{background:#f0f4f8}
.badge{display:inline-block;width:22px;height:22px;border-radius:50%;color:white;
       font-weight:bold;font-size:.82em;line-height:22px;text-align:center}
.badge-na{background:#bdc3c7;color:#555;font-size:.72em}
.cell-match{color:#27ae60;font-weight:600}
.cell-mismatch{color:#e74c3c}
.cell-na{color:#95a5a6;font-style:italic}
.note{background:#eaf4fb;border-left:4px solid #3498db;
      padding:12px 16px;border-radius:4px;margin:16px 0;font-size:.88em;line-height:1.6}
.warn{background:#fef9e7;border-left:4px solid #f39c12;
      padding:12px 16px;border-radius:4px;margin:16px 0;font-size:.88em;line-height:1.6}
.rank1{outline:2px solid #f1c40f;outline-offset:-1px;font-weight:700}
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<title>{method} Verification Report</title>
<style>{css}</style>
</head>
<body>

<h1>{method} — Verification Report</h1>
<p style="color:#7f8c8d;font-size:.88em">
  Comparing NPSK P-loss rankings ({method}, {n_total // len(LOSSES)} synths × {len(LOSSES)} losses)
  against published in-domain rankings.
  Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}.
</p>

<div class="note">
  <strong>IEEE paper</strong> (Salimi et al., 200 trials):
  BP-Noise → <code>bandpass_noise_v1</code>,
  Add-SineSaw → <code>add_sinesaw</code>,
  Noise-AM → <code>am_noise</code>,
  SineSaw-AM → <code>sine_mod_saw</code><br>
  <strong>ISMIR paper</strong> (in-domain, 200 trials):
  Chirp (no delay) → <code>chirplet</code>
</div>
"""

    html += """
<div class="note" style="font-size:.85em">
  <span style="background:#f1c40f;padding:1px 5px;border-radius:3px;font-size:.9em">■ yellow border</span>
  = published rank-1 cell (most important).
</div>
"""

    html += f"""
<h2>Summary</h2>
<div class="summary-box">
  <div class="stat {'green' if n_rank1_match == n_rank1_pub else 'red'}">
    <div class="num">{n_rank1_match}/{n_rank1_pub}</div>
    <div class="lbl">rank-1 match</div>
  </div>
  <div class="stat {'green' if n_fail == 0 else 'amber'}">
    <div class="num">{n_match}/{n_total}</div>
    <div class="lbl">all ranks match</div>
  </div>
  <div class="stat {'red' if n_fail else 'green'}">
    <div class="num">{n_fail}</div>
    <div class="lbl">disagreements</div>
  </div>
  <div class="stat amber">
    <div class="num">{n_miss}</div>
    <div class="lbl">missing data</div>
  </div>
  <div class="stat blue">
    <div class="num">{total_trials}</div>
    <div class="lbl">total trials</div>
  </div>
</div>
"""

    def rank_table(synth_list, ranks):
        t = '<table>\n<tr><th class="left">Synth</th>'
        t += "".join(f'<th>{l.replace("_","<br>")}</th>' for l in LOSSES)
        t += "</tr>\n"
        for synth in synth_list:
            t += f'<tr><td class="left">{SYNTH_LABELS[synth]}</td>'
            for loss in LOSSES:
                pub = PUBLISHED[synth][loss]
                comp = ranks[synth].get(loss, "—")
                t += match_cell(pub, comp, highlight_rank1=True)
            t += "</tr>\n"
        t += "</table>\n"
        return t

    html += f'<h2>Rank Comparison: Returned P-loss <small style="font-size:.6em;color:#888">({"final step" if is_gd else "best-audio step"})</small></h2>\n'
    html += '<p style="font-size:.85em;color:#7f8c8d">Yellow border = published rank 1 (best loss per synth). This is the headline metric.</p>\n'
    html += "<h3>IEEE paper</h3>\n" + rank_table(list(IEEE_PUBLISHED), ret_ranks)
    html += "<h3>ISMIR paper (in-domain)</h3>\n" + rank_table(list(ISMIR_PUBLISHED), ret_ranks)

    html += '<h2>Rank Comparison: Best P-loss <small style="font-size:.6em;color:#888">(diagnostic — best ever visited)</small></h2>\n'
    html += '<p style="font-size:.85em;color:#7f8c8d">Minimum P-loss at any step. If best ranks match but returned ranks do not, the method visited good params but failed to return them.</p>\n'
    html += "<h3>IEEE paper</h3>\n" + rank_table(list(IEEE_PUBLISHED), best_ranks)
    html += "<h3>ISMIR paper (in-domain)</h3>\n" + rank_table(list(ISMIR_PUBLISHED), best_ranks)

    html += "<h2>Per-Synth Detail</h2>\n"
    for synth in SYNTHS:
        paper_tag = SYNTH_PAPER[synth]
        html += f'<h3>{SYNTH_LABELS[synth]} <small style="font-size:.65em;color:#888">({paper_tag}) <code>{synth}</code></small></h3>\n'
        html += f"""<table>
<tr>
  <th class="left">Loss</th>
  <th>n trials</th>
  <th>{returned_label}</th>
  <th>Best P-loss<br>median</th>
  <th>Published<br>rank</th>
  <th>Returned<br>rank</th>
  <th>Rank-1<br>match</th>
  <th>Best<br>rank</th>
</tr>\n"""
        for loss in LOSSES:
            d = data[synth][loss]
            pub = PUBLISHED[synth][loss]
            rr = ret_ranks[synth].get(loss, "—")
            br = best_ranks[synth].get(loss, "—")
            n = d["n"]
            returned, bests = d["returned"], d["bests"]
            ret_med = f"{np.median(returned):.4f}" if returned else "—"
            best_med = f"{np.median(bests):.4f}" if bests else "—"
            if pub == 1:
                r1_match = ("✓" if rr == 1 else ("—" if not isinstance(rr, int) else "✗"))
                r1_cls = "cell-match" if r1_match == "✓" else ("cell-na" if r1_match == "—" else "cell-mismatch")
            else:
                r1_match, r1_cls = "·", "cell-na"
            r1_style = ' style="outline:2px solid #f1c40f;outline-offset:-1px"' if pub == 1 else ""
            html += (
                f"<tr{r1_style}>"
                f'<td class="left">{loss}</td>'
                f"<td>{n}</td>"
                f"<td>{ret_med}</td>"
                f"<td>{best_med}</td>"
                f"<td>{rank_badge(pub)}</td>"
                f"<td>{rank_badge(rr, muted=not isinstance(rr,int))}</td>"
                f'<td class="{r1_cls}">{r1_match}</td>'
                f"<td>{rank_badge(br, muted=not isinstance(br,int))}</td>"
                f"</tr>\n"
            )
        html += "</table>\n"

    html += f"""
<div class="note" style="margin-top:40px">
  <strong>Interpretation:</strong><br>
  <strong>Rank-1 accuracy</strong> (yellow-bordered cells) is the headline metric —
  does the method correctly identify the best loss function for each synth?<br>
  <em>Returned P-loss</em>: {"last gradient step (step 200) for GD" if is_gd else "P-loss at the step with the lowest audio loss — the candidate the method returns"}.<br>
  <em>Best P-loss</em>: minimum P-loss ever visited (oracle). If best ranks match but returned ranks don't,
  the method found good params but {"GD diverged away from them" if is_gd else "returned a worse candidate"}.
</div>
<hr style="border:none;border-top:1px solid #ecf0f1;margin-top:40px">
<p style="font-size:.78em;color:#bdc3c7;text-align:center">
  Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
</p></body></html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to {out_path}")
    print(f"Rank-1 match:  {n_rank1_match}/{n_rank1_pub}")
    print(f"All ranks:     {n_match}/{n_total} match, {n_miss} missing, {n_fail} disagreements")
    print(f"Total trials:  {total_trials}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="GD", choices=METHODS)
    parser.add_argument("--out", default=None,
                        help="Output path (default: results/{method}_report.html)")
    args = parser.parse_args()
    out = Path(args.out) if args.out else RES / f"{args.method}_report.html"
    main(out, args.method)
