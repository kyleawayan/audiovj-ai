"""Build a self-contained HTML review tool for a recorded session.

Renders the waveform, the Ableton Link beatgrid, the model's p(drop) trace and
your note-61 drop_start / drop_end labels on one timeline, so a label placed on
the wrong downbeat can be found and fixed by eye and ear.

Peaks are precomputed here rather than in the browser: a 30-minute float32 WAV
is ~320 MB and decoding it in a page is slow and memory-hungry. The page draws
from a small peak array and streams the WAV separately for playback, so it
opens instantly regardless of set length.

Usage:
    uv run python experiments/_labeltool.py ~/audiovj-sessions/<session>
    uv run python experiments/_serve.py ~/audiovj-sessions/<session>
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

BUCKET = 1024   # samples per peak column -> ~23 ms at 44.1 kHz


def read_float32_wav(path: Path):
    """Minimal reader for the float32 WAV the recorder writes."""
    raw = path.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {path}")
    pos, sr, data = 12, None, None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt, ch, sr = struct.unpack("<HHI", body[:8])
            if fmt != 3:
                raise ValueError(f"expected IEEE float (3), got format {fmt}")
        elif cid == b"data":
            # An interrupted session leaves a zero data size; fall back to the
            # rest of the file rather than returning nothing.
            data = body if size else raw[pos + 8:]
        pos += 8 + size + (size & 1)
    if sr is None or data is None:
        raise ValueError("missing fmt or data chunk")
    n = len(data) // 4
    return np.frombuffer(data[:n * 4], dtype="<f4"), sr


def main() -> None:
    sess = Path(sys.argv[1]).expanduser()
    man = json.loads((sess / "manifest.json").read_text())
    recs = [json.loads(l) for l in (sess / "events.jsonl").read_text().splitlines() if l.strip()]
    sr = int(man.get("sample_rate", 44100))
    start = int(man.get("audio_start_pos", 0))

    wav_path = sess / "audio.wav"
    if wav_path.exists():
        samples, sr = read_float32_wav(wav_path)
    else:
        samples = np.zeros(0, dtype=np.float32)
        print("WARNING: no audio.wav — drawing events only")

    n_buckets = max(1, len(samples) // BUCKET)
    usable = samples[:n_buckets * BUCKET].reshape(n_buckets, BUCKET)
    if n_buckets:
        lo = np.clip(usable.min(axis=1) * 127, -127, 127).astype(np.int8)
        hi = np.clip(usable.max(axis=1) * 127, -127, 127).astype(np.int8)
    else:
        lo = hi = np.zeros(0, dtype=np.int8)

    def to_sec(rec):
        return (rec.get("audio_pos", start) - start) / sr

    downbeats = [{"t": to_sec(r), "p": (r.get("probs") or [0] * 10)[3],
                  "ph": r.get("current", ""), "irr": bool(r.get("irregular")),
                  "i": r.get("downbeat_index")}
                 for r in recs if r["kind"] == "downbeat"]
    # downbeat_index is only stamped on labels in some builds; derive if absent
    for i, d in enumerate(downbeats):
        if d["i"] is None:
            d["i"] = i + 1

    labels = [{"t": to_sec(r), "kind": r.get("label", "drop_start"),
               "db": r.get("downbeat_index"),
               "suspect": bool(r.get("press_suspect")),
               "early": r.get("press_beats_early"),
               "p": r.get("p_drop_at_label"),
               "phrase": r.get("phrase_at_label", "")}
              for r in recs if r["kind"] == "drop_label"]

    cues = [{"t": to_sec(r), "kind": e["kind"]}
            for r in recs if r["kind"] == "downbeat"
            for e in (r.get("events") or [])
            if e["kind"] in ("drop_start", "drop_end")]

    data = {
        "sr": sr, "bucket": BUCKET, "duration": len(samples) / sr if sr else 0,
        "lo": lo.tolist(), "hi": hi.tolist(),
        "downbeats": downbeats, "labels": labels, "modelCues": cues,
        "session": sess.name,
        "bpm": man.get("bpm_at_start"),
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    out = sess / "review.html"
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(downbeats)} downbeats, {len(labels)} labels, "
          f"{data['duration']:.0f}s audio")
    print(f"\nnext:\n  cd '{sess}' && python3 -m http.server 8000")
    print("  open http://localhost:8000/review.html")


TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>AudioVJ label review</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0d0f12; color:#e6e6e6;
         font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }
  header { padding:10px 14px; border-bottom:1px solid #23272e; display:flex;
           gap:18px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:14px; margin:0; font-weight:600; letter-spacing:.02em; }
  button { background:#1b1f26; color:#e6e6e6; border:1px solid #333a44;
           border-radius:6px; padding:5px 11px; cursor:pointer; font:inherit; }
  button:hover { background:#252b34; }
  button.warn { border-color:#7a3030; color:#ffb4b4; }
  #wrap { padding:10px 14px; }
  canvas { display:block; width:100%; border:1px solid #23272e; border-radius:6px;
           background:#101318; cursor:crosshair; }
  #over { height:90px; margin-bottom:10px; }
  #main { height:340px; }
  .k { color:#7d8590; }
  .pill { padding:2px 8px; border-radius:999px; border:1px solid #333a44; }
  #list { margin-top:12px; max-height:230px; overflow:auto;
          border:1px solid #23272e; border-radius:6px; }
  table { border-collapse:collapse; width:100%; }
  th,td { text-align:left; padding:5px 10px; border-bottom:1px solid #1b1f26;
          white-space:nowrap; }
  th { color:#7d8590; font-weight:500; position:sticky; top:0; background:#141820; }
  tr.sel td { background:#1d2733; }
  tr.suspect td { color:#ffcf8f; }
  tr:hover td { background:#191e26; cursor:pointer; }
  .start { color:#5fd48a; } .end { color:#ff8a8a; }
  #help { margin-top:10px; color:#7d8590; }
  kbd { background:#1b1f26; border:1px solid #333a44; border-radius:4px;
        padding:1px 5px; }
</style>
<header>
  <h1>AudioVJ label review</h1>
  <span class="pill" id="sess"></span>
  <span class="pill" id="counts"></span>
  <button id="play">Play / Pause</button>
  <button id="addS">Add drop_start (S)</button>
  <button id="addE">Add drop_end (E)</button>
  <button id="del" class="warn">Delete selected (Del)</button>
  <button id="exp">Export corrected JSON</button>
  <span class="k" id="clock">0:00</span>
</header>
<div id="wrap">
  <canvas id="over"></canvas>
  <canvas id="main"></canvas>
  <div id="list"></div>
  <div id="help">
    <kbd>Space</kbd> play/pause &nbsp; <kbd>&larr;</kbd><kbd>&rarr;</kbd> nudge
    selected label one downbeat &nbsp; <kbd>S</kbd>/<kbd>E</kbd> add label at
    playhead (snaps to nearest downbeat) &nbsp; <kbd>Del</kbd> remove &nbsp;
    <kbd>+</kbd>/<kbd>-</kbd> zoom &nbsp; click a row to jump
  </div>
</div>
<audio id="au" src="audio.wav" preload="metadata"></audio>
<script>
const D = __DATA__;
const au = document.getElementById('au');
const over = document.getElementById('over'), main = document.getElementById('main');
let labels = D.labels.map((l,i) => ({...l, id:i}));
let sel = null, view = 30, center = 0, nextId = labels.length;
// The playhead is OURS, not the <audio> element's. If audio fails to load or a
// browser refuses the seek, reading currentTime pins everything at 0 and the
// whole tool goes dead. Editing must work with no audio at all.
let head = 0;
function seek(t){
  head = Math.max(0, Math.min(D.duration || t, t));
  center = head;
  try { au.currentTime = head; } catch (e) {}
}

document.getElementById('sess').textContent =
  D.session + '  ' + D.duration.toFixed(0) + 's' + (D.bpm ? '  ' + D.bpm.toFixed(1) + ' BPM' : '');

function fit(c){ const r=c.getBoundingClientRect(), d=devicePixelRatio||1;
  c.width=r.width*d; c.height=r.height*d;
  const x=c.getContext('2d'); x.setTransform(d,0,0,d,0,0); return x; }

// nearest downbeat time -- labels must sit ON the grid, never between it
function snap(t){
  if(!D.downbeats.length) return t;
  let b=D.downbeats[0];
  for(const d of D.downbeats) if(Math.abs(d.t-t)<Math.abs(b.t-t)) b=d;
  return b.t;
}
function dbIndexAt(t){
  let b=null;
  for(const d of D.downbeats) if(b===null||Math.abs(d.t-t)<Math.abs(b.t-t)) b=d;
  return b?b.i:null;
}
function peakAt(i){ return [D.lo[i]/127, D.hi[i]/127]; }

function drawWave(ctx,w,h,t0,t1,dense){
  const spb = D.bucket/D.sr, mid=h/2;
  const i0=Math.max(0,Math.floor(t0/spb)), i1=Math.min(D.lo.length,Math.ceil(t1/spb));
  ctx.fillStyle='#2f6f9f';
  for(let px=0;px<w;px++){
    const a=i0+(i1-i0)*px/w, b=i0+(i1-i0)*(px+1)/w;
    let lo=0,hi=0;
    for(let i=Math.floor(a);i<Math.max(Math.floor(a)+1,Math.ceil(b));i++){
      if(i<0||i>=D.lo.length) continue;
      const [l,h2]=peakAt(i); if(l<lo)lo=l; if(h2>hi)hi=h2;
    }
    ctx.fillRect(px, mid-hi*mid*0.92, 1, Math.max(1,(hi-lo)*mid*0.92));
  }
}

function drawGrid(ctx,w,h,t0,t1){
  const span=t1-t0;
  // Beatgrid: every downbeat is a bar line. Only drawn when they are far
  // enough apart to read, otherwise the view is a solid block of lines.
  const pxPerDb = D.downbeats.length>1 ? w*( (D.downbeats[1].t-D.downbeats[0].t)/span ) : 999;
  if(pxPerDb>6){
    for(const d of D.downbeats){
      if(d.t<t0||d.t>t1) continue;
      const x=(d.t-t0)/span*w;
      ctx.fillStyle = d.irr ? '#c0693a' : '#252c36';
      ctx.fillRect(x,0,1,h);
      if(pxPerDb>46){ ctx.fillStyle='#4a5563'; ctx.font='10px monospace';
        ctx.fillText(d.i, x+3, h-4); }
    }
  }
  // p(drop) trace
  ctx.strokeStyle='#e8c46a'; ctx.lineWidth=1.5; ctx.beginPath();
  let started=false;
  for(const d of D.downbeats){
    if(d.t<t0-1||d.t>t1+1) continue;
    const x=(d.t-t0)/span*w, y=h-6-d.p*(h-26);
    started?ctx.lineTo(x,y):(ctx.moveTo(x,y),started=true);
  }
  ctx.stroke();
  ctx.fillStyle='#6b6250'; ctx.font='10px monospace';
  ctx.fillText('p(drop)', 6, 12);
  // 0.30 reference line -- the live onset threshold
  const y30=h-6-0.30*(h-26);
  ctx.strokeStyle='#3d3a2e'; ctx.setLineDash([4,4]); ctx.beginPath();
  ctx.moveTo(0,y30); ctx.lineTo(w,y30); ctx.stroke(); ctx.setLineDash([]);
}

function drawLabels(ctx,w,h,t0,t1,small){
  const span=t1-t0;
  for(const c of D.modelCues){           // what the model fired, for comparison
    if(c.t<t0||c.t>t1) continue;
    const x=(c.t-t0)/span*w;
    ctx.fillStyle = c.kind==='drop_start' ? '#2f6b45' : '#6b2f2f';
    ctx.fillRect(x-1,h-14,2,14);
  }
  for(const l of labels){
    if(l.t<t0||l.t>t1) continue;
    const x=(l.t-t0)/span*w;
    const col = l.kind==='drop_start' ? '#5fd48a' : '#ff8a8a';
    ctx.fillStyle = (sel===l.id)?'#ffffff':col;
    ctx.fillRect(x-1,0,2,h);
    if(!small){
      ctx.fillStyle=(sel===l.id)?'#ffffff':col;
      ctx.font='11px monospace';
      ctx.fillText((l.kind==='drop_start'?'START':'END')+(l.suspect?' ?':''), x+4, 14);
    }
  }
}

let lastTick = performance.now(), audioTracks = true;
function render(){
  const now = performance.now(), dt = (now - lastTick) / 1000; lastTick = now;
  if(!au.paused){
    // Adopt the element's clock only while it is actually following us. If the
    // server cannot serve byte ranges the element is stuck near 0, and copying
    // that into head would yank the view back to the start on every frame.
    if(au.readyState > 2 && Math.abs(au.currentTime - head) < 1.5){
      head = au.currentTime; audioTracks = true;
    } else {
      head = Math.min(D.duration, head + dt); audioTracks = false;
    }
  }
  const t = head;
  if(t<center-view*0.4||t>center+view*0.4) center=t;
  let t0=Math.max(0,center-view/2), t1=Math.min(D.duration,t0+view); t0=Math.max(0,t1-view);

  let c=fit(over), w=over.getBoundingClientRect().width, h=over.getBoundingClientRect().height;
  c.clearRect(0,0,w,h); drawWave(c,w,h,0,D.duration); drawLabels(c,w,h,0,D.duration,true);
  c.fillStyle='#ffffff'; c.fillRect(t/D.duration*w,0,1,h);
  c.strokeStyle='#48525f';
  c.strokeRect(t0/D.duration*w,0,(t1-t0)/D.duration*w,h);

  c=fit(main); w=main.getBoundingClientRect().width; h=main.getBoundingClientRect().height;
  c.clearRect(0,0,w,h);
  drawGrid(c,w,h,t0,t1); drawWave(c,w,h,t0,t1,true); drawLabels(c,w,h,t0,t1,false);
  c.fillStyle='#ffffff'; c.fillRect((t-t0)/(t1-t0)*w,0,1,h);

  document.getElementById('clock').textContent =
    Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0')+
    '  bar '+(dbIndexAt(t)??'-')+'  zoom '+view.toFixed(0)+'s'+
    (audioTracks ? '' : '  [audio not seeking — serve with _serve.py]');
  main._t0=t0; main._t1=t1;
  requestAnimationFrame(render);
}

function counts(){
  const s=labels.filter(l=>l.kind==='drop_start').length;
  const e=labels.filter(l=>l.kind==='drop_end').length;
  document.getElementById('counts').textContent =
    s+' start / '+e+' end'+(s!==e?'  UNPAIRED':'');
}
function table(){
  const rows=labels.slice().sort((a,b)=>a.t-b.t).map(l=>
    `<tr data-id="${l.id}" class="${sel===l.id?'sel':''} ${l.suspect?'suspect':''}">
      <td class="${l.kind==='drop_start'?'start':'end'}">${l.kind==='drop_start'?'START':'END'}</td>
      <td>${Math.floor(l.t/60)}:${String(Math.floor(l.t%60)).padStart(2,'0')}</td>
      <td class="k">bar ${l.db??dbIndexAt(l.t)??'-'}</td>
      <td class="k">p(drop) ${l.p==null?'-':l.p.toFixed(2)}</td>
      <td class="k">model said ${l.phrase||'-'}</td>
      <td class="k">${l.early==null?'':'pressed '+l.early.toFixed(1)+'b early'}</td>
      <td class="k">${l.suspect?'SUSPECT — pressed early in the bar':''}</td>
    </tr>`).join('');
  document.getElementById('list').innerHTML =
    `<table><tr><th>kind</th><th>time</th><th>bar</th><th>p(drop) then</th>
     <th>model</th><th>press</th><th>flag</th></tr>${rows}</table>`;
  document.querySelectorAll('#list tr[data-id]').forEach(tr=>tr.onclick=()=>{
    sel=+tr.dataset.id; const l=labels.find(x=>x.id===sel);
    seek(Math.max(0,l.t-4)); center=l.t; table();
  });
  counts();
}

main.onmousedown = ev => {
  const r=main.getBoundingClientRect(), x=ev.clientX-r.left;
  const t=main._t0+(x/r.width)*(main._t1-main._t0);
  let best=null,bd=1e9;
  for(const l of labels){ const d=Math.abs(l.t-t); if(d<bd){bd=d;best=l;} }
  const pxPerSec=r.width/(main._t1-main._t0);
  if(best && bd*pxPerSec<8){ sel=best.id; table(); }
  else seek(t);
};
over.onmousedown = ev => {
  const r=over.getBoundingClientRect();
  seek(((ev.clientX-r.left)/r.width)*D.duration);
};

function addLabel(kind){
  const t=snap(head);
  labels.push({id:nextId++, t, kind, db:dbIndexAt(t), suspect:false,
               early:null, p:null, phrase:'(added by hand)'});
  sel=nextId-1; table();
}
function nudge(dir){
  const l=labels.find(x=>x.id===sel); if(!l) return;
  const idx=D.downbeats.findIndex(d=>Math.abs(d.t-l.t)<1e-6);
  const j=(idx<0?0:idx)+dir;
  if(j<0||j>=D.downbeats.length) return;
  l.t=D.downbeats[j].t; l.db=D.downbeats[j].i; l.p=D.downbeats[j].p;
  l.phrase=D.downbeats[j].ph; l.suspect=false; center=l.t; table();
}
document.onkeydown = e => {
  if(e.key===' '){ e.preventDefault(); au.paused?au.play():au.pause(); }
  else if(e.key==='ArrowRight'){ e.preventDefault(); sel!==null?nudge(1):seek(head+2); }
  else if(e.key==='ArrowLeft'){ e.preventDefault(); sel!==null?nudge(-1):seek(head-2); }
  else if(e.key==='Delete'||e.key==='Backspace'){ labels=labels.filter(l=>l.id!==sel); sel=null; table(); }
  else if(e.key==='s'||e.key==='S') addLabel('drop_start');
  else if(e.key==='e'||e.key==='E') addLabel('drop_end');
  else if(e.key==='+'||e.key==='=') view=Math.max(4,view/1.5);
  else if(e.key==='-'||e.key==='_') view=Math.min(D.duration||600,view*1.5);
};
document.getElementById('play').onclick=()=>au.paused?au.play():au.pause();
document.getElementById('addS').onclick=()=>addLabel('drop_start');
document.getElementById('addE').onclick=()=>addLabel('drop_end');
document.getElementById('del').onclick=()=>{labels=labels.filter(l=>l.id!==sel);sel=null;table();};
document.getElementById('exp').onclick=()=>{
  const out={session:D.session, sample_rate:D.sr, corrected_at:new Date().toISOString(),
    labels:labels.slice().sort((a,b)=>a.t-b.t).map(l=>({
      kind:l.kind, t_sec:l.t, audio_sample:Math.round(l.t*D.sr),
      downbeat_index:l.db, hand_edited:l.phrase==='(added by hand)'}))};
  const b=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(b); a.download='labels_corrected.json'; a.click();
};
au.addEventListener('error', () => {
  document.getElementById('sess').textContent +=
    '  — audio.wav failed to load (serve the folder over http, not file://). '
  + 'Editing still works.';
});
table(); render();
</script>
"""

if __name__ == "__main__":
    main()
