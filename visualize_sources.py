"""
visualize_sources.py
====================
Generate an interactive HTML trajectory profiler for the EEGK data sources.

Motivation
----------
"Have you looked at the specific trajectories?" -- a source's numbers (accuracy,
angle error, endpoint radius) can match while its trajectory *shape* is wrong, and
vice versa. This tool renders every source's trajectories in a compass layout
(8 target directions), sources overlaid by color with eegk_real as the anchor, so
shape and endpoint distribution can be eyeballed side by side. It doubles as the
sanity check for sim scaling: a raw-vs-scaled toggle shows the mismatch directly.

What it renders
---------------
    eegk_real                         (the anchor)
    eegk_sim  (raw AND radius-scaled, toggle in UI)
    eegk_surrogate                    (if data/surrogate/*.csv exists)
    blend: real+sim / real+surrogate / real+sim+surrogate   (concatenation views)

Sources with no data on disk are skipped with a note (e.g. before you've run
surrogate_constructor.py). The HTML bakes in a data snapshot at generation time;
re-run this script whenever the underlying data changes.

RUN
---
    python visualize_sources.py
    python visualize_sources.py --match step --max_per_cell 30 --out results/profiler.html
    python visualize_sources.py --subjects S02 S04     # restrict subjects

Everything is path-flexible; defaults match the repo's data/ layout. Output goes
to results/ (create it / gitignore it alongside runs/ and data/surrogate/).
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import copilot_dataset as cd
import sim_scaling as ss

# Default subjects: the 5 with real EEGK data (S03/S06 have no real counterpart).
DEFAULT_SUBJECTS = ["S01", "S02", "S04", "S05", "S07"]

# Source display colors -- real is the bright anchor; rest are colorblind-safe.
SRC_COLORS = {
    "eegk_real": "#e8eef4",
    "eegk_sim (raw)": "#f6866f",
    "eegk_sim (scaled)": "#4fd6c8",
    "eegk_sim (scaled+dwell)": "#7ee081",
    "eegk_surrogate": "#e879a6",
    "blend: real+sim": "#6ea8fe",
    "blend: real+surrogate": "#c9a227",
    "blend: real+sim+surrogate": "#b07de0",
}


def _subset(trajs: Sequence[cd.Trajectory], subjects) -> List[cd.Trajectory]:
    keep = set(subjects)
    return [t for t in trajs if t.subject_id in keep]


def _traj_pts(t: cd.Trajectory, ndigits: int) -> List[list]:
    return [[round(float(x), ndigits), round(float(y), ndigits)] for x, y in t.pos]


def _cells_for_source(
    trajs: Sequence[cd.Trajectory], subjects, max_per_cell: int, ndigits: int, rng
) -> Dict[str, list]:
    cells: Dict[str, list] = {}
    for subj in subjects:
        for lbl in range(8):
            group = [t for t in trajs if t.subject_id == subj and t.target_label == lbl]
            if not group:
                continue
            if len(group) > max_per_cell:
                idx = rng.choice(len(group), max_per_cell, replace=False)
                group = [group[i] for i in idx]
            cells[f"{subj}|{lbl}"] = [_traj_pts(t, ndigits) for t in group]
    return cells


def build_payload(cfg) -> dict:
    subjects = cfg["subjects"]
    rng = np.random.default_rng(cfg["seed"])

    # --- load the sources that exist ---
    real = _subset(cd.load_source("eegk_real", repo_root=cfg["repo_root"]), subjects)
    if not real:
        raise SystemExit(
            "No eegk_real trajectories found. Check --repo_root and that "
            "data/OnlineArmTrajectoryEEGK/ is present."
        )

    sim_raw = _subset(cd.load_source("eegk_sim", repo_root=cfg["repo_root"]), subjects)
    # scaled sim: calibrated to real, unreferenced subjects dropped (default)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim_scaled = ss.scale_sim_to_real(sim_raw, real, match=cfg["match"])
        # scaled + dwell-calibrated: prepend real-sampled hold (Option A)
        sim_dwell = ss.add_dwell_to_sim(sim_scaled, real, seed=cfg["seed"])

    sources: Dict[str, List[cd.Trajectory]] = {
        "eegk_real": real,
        "eegk_sim (raw)": sim_raw,
        "eegk_sim (scaled)": sim_scaled,
        "eegk_sim (scaled+dwell)": sim_dwell,
    }

    # optional surrogate
    surr: List[cd.Trajectory] = []
    surr_path = Path(cfg["repo_root"]) / cfg["surrogate_csv"]
    if surr_path.exists():
        surr = _subset(cd.load_csv_file(str(surr_path)), subjects)
        sources["eegk_surrogate"] = surr
    else:
        print(f"note: no surrogate at {surr_path} -- skipping (run surrogate_constructor.py)")

    # blends (concatenation views; ratio-controlled blends come from blend_constructor)
    # use the fully-calibrated sim (scaled+dwell) -- that's what training will use
    sources["blend: real+sim"] = real + sim_dwell
    if surr:
        sources["blend: real+surrogate"] = real + surr
        sources["blend: real+sim+surrogate"] = real + sim_dwell + surr

    # --- assemble compact payload ---
    payload = {
        "subjects": list(subjects),
        "label_to_name": {i: cd.DIR_NAMES[i] for i in range(8)},
        "match": cfg["match"],
        "sources": {},
        "meta": {},
        "colors": {k: SRC_COLORS.get(k, "#9aa7b4") for k in sources},
    }
    for sname, trajs in sources.items():
        payload["sources"][sname] = _cells_for_source(
            trajs, subjects, cfg["max_per_cell"], cfg["ndigits"], rng
        )
        b = cd.baseline_metrics(trajs)
        payload["meta"][sname] = {
            "acc": round(b["accuracy"] * 100, 2),
            "err": round(b["angle_error_deg"], 2),
            "n": len(trajs),
        }
    return payload


# --------------------------------------------------------------------------- #
# HTML template (compass layout, source overlay, paths/endpoints, raw/scaled)
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trajectory Profiler — EEGK sources</title>
<style>
  :root{--bg:#10161c;--panel:#182029;--panel-line:#263340;--ink:#e8eef4;
    --ink-dim:#8da0b3;--ink-faint:#5a6f82;--grid:#22303c;--accent:#4fd6c8;--target:#f2b134;}
  *{box-sizing:border-box;} html,body{margin:0;background:var(--bg);color:var(--ink);
    font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;}
  body{padding:20px 22px 60px;}
  .head{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 18px;margin-bottom:4px;}
  h1{font-size:15px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;margin:0;}
  .sub{font-size:12px;color:var(--ink-faint);letter-spacing:.03em;}
  .controls{display:flex;flex-wrap:wrap;gap:18px 26px;align-items:flex-start;margin:16px 0 18px;
    padding:14px 16px;background:var(--panel);border:1px solid var(--panel-line);border-radius:4px;}
  .ctl-group{display:flex;flex-direction:column;gap:7px;}
  .ctl-label{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-faint);}
  .chips{display:flex;flex-wrap:wrap;gap:6px;max-width:560px;}
  .chip{font:inherit;font-size:11px;padding:5px 10px;cursor:pointer;background:transparent;
    color:var(--ink-dim);border:1px solid var(--panel-line);border-radius:3px;transition:all .12s;
    user-select:none;display:flex;align-items:center;gap:7px;}
  .chip:hover{border-color:var(--ink-faint);color:var(--ink);}
  .chip.on{color:var(--ink);border-color:currentColor;}
  .chip .sw{width:9px;height:9px;border-radius:2px;flex:none;}
  .seg{display:flex;border:1px solid var(--panel-line);border-radius:3px;overflow:hidden;}
  .seg button{font:inherit;font-size:11px;padding:5px 11px;cursor:pointer;background:transparent;
    color:var(--ink-dim);border:none;border-right:1px solid var(--panel-line);}
  .seg button:last-child{border-right:none;}
  .seg button.on{background:var(--accent);color:#06231f;font-weight:600;}
  select,.rng{font:inherit;font-size:11px;background:var(--bg);color:var(--ink);
    border:1px solid var(--panel-line);border-radius:3px;padding:5px 8px;}
  .grid{display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(3,auto);
    gap:10px;max-width:1180px;}
  .cell{background:var(--panel);border:1px solid var(--panel-line);border-radius:4px;
    position:relative;aspect-ratio:1/1;}
  .cell.center{background:transparent;border:none;display:flex;align-items:center;
    justify-content:center;flex-direction:column;gap:8px;padding:8px;}
  .cell .dlabel{position:absolute;top:7px;left:9px;font-size:11px;letter-spacing:.1em;
    color:var(--ink-dim);font-weight:600;z-index:2;}
  .cell .dcount{position:absolute;top:7px;right:9px;font-size:9.5px;color:var(--ink-faint);z-index:2;}
  svg{width:100%;height:100%;display:block;}
  .legend-in{font-size:10px;color:var(--ink-faint);line-height:1.7;}
  .legend-in .row{display:flex;align-items:center;gap:6px;}
  .legend-in .sw{width:10px;height:2.5px;}
  .statbar{margin-top:4px;font-size:9.5px;color:var(--ink-faint);text-align:left;}
  .statbar b{color:var(--ink-dim);font-weight:600;}
  .foot{max-width:1180px;margin-top:16px;font-size:11px;color:var(--ink-faint);line-height:1.6;}
  .hint{color:var(--ink-faint);font-size:10.5px;}
  @media (max-width:720px){.grid{grid-template-columns:1fr 1fr;grid-template-rows:none;}
    .cell.center{grid-column:1/-1;aspect-ratio:auto;}}
</style></head><body>
  <div class="head"><h1>Trajectory Profiler</h1>
    <span class="sub">EEGK sources · normalized cursor space · match=__MATCH__</span></div>
  <div class="controls">
    <div class="ctl-group"><span class="ctl-label">Sources</span><div class="chips" id="srcChips"></div></div>
    <div class="ctl-group"><span class="ctl-label">Subject</span><select id="subjSel"></select></div>
    <div class="ctl-group"><span class="ctl-label">View</span>
      <div class="seg" id="viewSeg"><button data-v="paths" class="on">Paths</button>
      <button data-v="endpoints">Endpoints</button></div></div>
    <div class="ctl-group"><span class="ctl-label">Trajectories / cell</span>
      <input class="rng" id="capRange" type="range" min="1" max="__MAXCAP__" value="__CAP0__" style="width:120px">
      <span class="hint" id="capVal">__CAP0__</span></div>
  </div>
  <div class="grid" id="grid"></div>
  <div class="foot" id="foot"></div>
<script>
const DATA = __PAYLOAD__;
const LABEL_BY_NAME={}; for(const [lbl,name] of Object.entries(DATA.label_to_name)) LABEL_BY_NAME[name]=+lbl;
const COMPASS=['NW','N','NE','W',null,'E','SW','S','SE'];
const SRC_COLORS=DATA.colors;
const SOURCES=Object.keys(DATA.sources);
const DEF=SOURCES.filter(s=>s==='eegk_real'||s==='eegk_sim (scaled+dwell)');
const state={active:new Set(DEF.length?DEF:[SOURCES[0]]),subject:DATA.subjects[0],view:'paths',cap:__CAP0__};

const srcChips=document.getElementById('srcChips');
SOURCES.forEach(s=>{const c=document.createElement('div');
  c.className='chip'+(state.active.has(s)?' on':'');
  c.style.color=state.active.has(s)?SRC_COLORS[s]:'';
  c.innerHTML=`<span class="sw" style="background:${SRC_COLORS[s]}"></span>${s}`;
  c.onclick=()=>{if(state.active.has(s))state.active.delete(s);else state.active.add(s);
    c.classList.toggle('on');c.style.color=state.active.has(s)?SRC_COLORS[s]:'';render();};
  srcChips.appendChild(c);});
const subjSel=document.getElementById('subjSel');
DATA.subjects.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;subjSel.appendChild(o);});
subjSel.onchange=e=>{state.subject=e.target.value;render();};
document.querySelectorAll('#viewSeg button').forEach(b=>{b.onclick=()=>{
  document.querySelectorAll('#viewSeg button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');state.view=b.dataset.v;render();};});
const capRange=document.getElementById('capRange'),capVal=document.getElementById('capVal');
capRange.oninput=e=>{state.cap=+e.target.value;capVal.textContent=e.target.value;render();};

const VB=200,C=VB/2,S=C*0.92;
function toXY(x,y){return [C+x*S,C-y*S];}
function targetDot(lbl){const name=DATA.label_to_name[lbl];
  const ang={N:90,S:270,E:0,W:180,NE:45,NW:135,SE:315,SW:225}[name]*Math.PI/180;
  return toXY(0.85*Math.cos(ang),0.85*Math.sin(ang));}
function cellSVG(lbl){const svg=[`<svg viewBox="0 0 ${VB} ${VB}">`];
  svg.push(`<circle cx="${C}" cy="${C}" r="${S*0.5}" fill="none" stroke="var(--grid)" stroke-width="0.6"/>`);
  svg.push(`<circle cx="${C}" cy="${C}" r="${S*0.9}" fill="none" stroke="var(--grid)" stroke-width="0.6"/>`);
  svg.push(`<line x1="${C}" y1="6" x2="${C}" y2="${VB-6}" stroke="var(--grid)" stroke-width="0.5"/>`);
  svg.push(`<line x1="6" y1="${C}" x2="${VB-6}" y2="${C}" stroke="var(--grid)" stroke-width="0.5"/>`);
  const [tx,ty]=targetDot(lbl);
  svg.push(`<circle cx="${tx}" cy="${ty}" r="3.2" fill="none" stroke="var(--target)" stroke-width="1.3"/>`);
  svg.push(`<circle cx="${tx}" cy="${ty}" r="0.8" fill="var(--target)"/>`);
  svg.push(`<circle cx="${C}" cy="${C}" r="1.4" fill="var(--ink-faint)"/>`);
  for(const s of SOURCES.filter(s=>state.active.has(s))){const col=SRC_COLORS[s];
    const cell=DATA.sources[s][`${state.subject}|${lbl}`]; if(!cell)continue;
    const trajs=cell.slice(0,state.cap);
    if(state.view==='paths'){for(const t of trajs){let d='M';
      for(let i=0;i<t.length;i++){const [px,py]=toXY(t[i][0],t[i][1]);d+=`${i?'L':''}${px.toFixed(1)},${py.toFixed(1)} `;}
      svg.push(`<path d="${d}" fill="none" stroke="${col}" stroke-width="0.7" stroke-opacity="0.3" stroke-linejoin="round"/>`);}}
    else{for(const t of trajs){const [ex,ey]=toXY(t[t.length-1][0],t[t.length-1][1]);
      svg.push(`<circle cx="${ex.toFixed(1)}" cy="${ey.toFixed(1)}" r="1.7" fill="${col}" fill-opacity="0.55"/>`);}}}
  svg.push('</svg>');return svg.join('');}
function centerCell(){const act=SOURCES.filter(s=>state.active.has(s));
  const rows=act.map(s=>`<div class="row"><span class="sw" style="background:${SRC_COLORS[s]}"></span>${s}</div>`).join('');
  const meta=act.map(s=>{const m=DATA.meta[s];return `<div><b>${s}</b> · ${m.acc}% · ${m.err}° · n=${m.n}</div>`;}).join('');
  return `<div class="legend-in">${rows}</div><div class="statbar">${meta}</div>`;}
function render(){const grid=document.getElementById('grid');grid.innerHTML='';
  COMPASS.forEach(name=>{const div=document.createElement('div');
    if(name===null){div.className='cell center';div.innerHTML=centerCell();grid.appendChild(div);return;}
    const lbl=LABEL_BY_NAME[name];div.className='cell';
    let count=0;SOURCES.filter(s=>state.active.has(s)).forEach(s=>{const c=DATA.sources[s][`${state.subject}|${lbl}`];if(c)count+=Math.min(c.length,state.cap);});
    div.innerHTML=`<span class="dlabel">${name}</span><span class="dcount">${count} shown</span>${cellSVG(lbl)}`;
    grid.appendChild(div);});
  document.getElementById('foot').innerHTML=
    "Rings mark 0.5 and 0.9 normalized radius. Gold ring marks the target direction; "
    +"each faint line is one trial's cursor path from the origin. Compare eegk_sim (raw), "
    +"(scaled), and (scaled+dwell) to see each calibration step: scaling fixes reach, dwell adds "
    +"the origin-hold real has. Switch to Endpoints to check where trials land. "
    +"Scaling and dwell are direction-preserving, so endpoints move radially onto the target, never around it.";}
render();
</script></body></html>"""


def render_html(payload: dict, cfg) -> str:
    cap0 = min(cfg["max_per_cell"], 15)
    html = HTML_TEMPLATE
    html = html.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    html = html.replace("__MATCH__", payload["match"])
    html = html.replace("__MAXCAP__", str(cfg["max_per_cell"]))
    html = html.replace("__CAP0__", str(cap0))
    return html


def main():
    ap = argparse.ArgumentParser(description="Generate the interactive trajectory profiler.")
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--match", default="radius", choices=["radius", "step"],
                    help="sim-to-real scaling moment (passed to sim_scaling)")
    ap.add_argument("--max_per_cell", type=int, default=25,
                    help="cap trajectories drawn per (source,subject,direction)")
    ap.add_argument("--ndigits", type=int, default=2, help="position rounding (smaller = lighter file)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--surrogate_csv", default="data/surrogate/surrogate_trajectories.csv")
    ap.add_argument("--out", default="results/trajectory_profiler.html")
    cfg = vars(ap.parse_args())

    payload = build_payload(cfg)
    html = render_html(payload, cfg)

    out = Path(cfg["repo_root"]) / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    kb = os.path.getsize(out) / 1024
    print(f"\nwrote {out}  ({kb:.0f} KB)")
    print(f"sources: {list(payload['sources'].keys())}")
    print("baselines:")
    for s, m in payload["meta"].items():
        print(f"  {s:32s} acc={m['acc']:5.2f}%  err={m['err']:5.2f}°  n={m['n']}")


if __name__ == "__main__":
    main()
