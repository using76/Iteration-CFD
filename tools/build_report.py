#!/usr/bin/env python3
"""
Build a self-contained HTML report from the rendered plume frames.

    python tools/build_report.py <framesDir> <runJson> <out.html>

`framesDir` is what render_plume.py wrote (PNGs + frames.json). `runJson` holds
the solver timings and case conditions - see `example_run_json()` for the shape.
Every image is base64-inlined so the report is one file with no dependencies.
"""

import base64
import json
import os
import sys


def b64(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def sparkline(frames, key, w=560, h=120, pad=28):
    """A small inline SVG series. No chart library, and none needed."""
    xs = [f["t"] for f in frames]
    ys = [f[key] for f in frames]
    if not xs:
        return ""
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 1, y1 + 1

    def px(t):
        return pad + (t - x0) / max(x1 - x0, 1e-9) * (w - 2 * pad)

    def py(v):
        return h - pad - (v - y0) / (y1 - y0) * (h - 2 * pad)

    pts = " ".join(f"{px(t):.1f},{py(v):.1f}" for t, v in zip(xs, ys))
    dots = "".join(
        f'<circle cx="{px(t):.1f}" cy="{py(v):.1f}" r="2.6" class="pt"/>'
        for t, v in zip(xs, ys)
    )
    return f"""<svg viewBox="0 0 {w} {h}" class="spark" role="img"
     aria-label="{esc(key)} versus time">
  <line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" class="ax"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" class="ax"/>
  <polyline points="{pts}" class="ln"/>{dots}
  <text x="{pad-6}" y="{pad+4}" class="tick" text-anchor="end">{y1:.0f}</text>
  <text x="{pad-6}" y="{h-pad+4}" class="tick" text-anchor="end">{y0:.0f}</text>
  <text x="{pad}" y="{h-8}" class="tick">{x0:.0f} s</text>
  <text x="{w-pad}" y="{h-8}" class="tick" text-anchor="end">{x1:.0f} s</text>
</svg>"""


def player(pid, frames, images, caption):
    """A slider + play button over one image series, like the FDS report's."""
    srcs = ",".join(f'"{images[f["stem"]]}"' for f in frames)
    labels = ",".join(f'"{f["t"]:.0f}"' for f in frames)
    n = len(frames)
    return f"""
<figure class="player" data-player="{pid}">
  <img id="img-{pid}" src="{images[frames[0]['stem']]}" alt="{esc(caption)}">
  <div class="ctl">
    <button type="button" id="play-{pid}" aria-label="재생">&#9654;</button>
    <input type="range" id="rng-{pid}" min="0" max="{n-1}" value="0" step="1"
           aria-label="시간 프레임">
    <span class="tval" id="lab-{pid}">t = 2 s</span>
  </div>
  <figcaption>{caption}</figcaption>
</figure>
<script>
(function() {{
  var src = [{srcs}], lab = [{labels}];
  var img = document.getElementById("img-{pid}");
  var rng = document.getElementById("rng-{pid}");
  var out = document.getElementById("lab-{pid}");
  var btn = document.getElementById("play-{pid}");
  var timer = null;
  function show(i) {{
    img.src = src[i]; out.textContent = "t = " + lab[i] + " s"; rng.value = i;
  }}
  rng.addEventListener("input", function() {{ show(+rng.value); }});
  btn.addEventListener("click", function() {{
    if (timer) {{ clearInterval(timer); timer = null; btn.innerHTML = "&#9654;"; return; }}
    btn.innerHTML = "&#9724;";
    timer = setInterval(function() {{
      var i = (+rng.value + 1) % src.length;
      show(i);
      if (i === src.length - 1) {{ clearInterval(timer); timer = null; btn.innerHTML = "&#9654;"; }}
    }}, 700);
  }});
  show(0);
}})();
</script>"""


CSS = """
:root{--paper:#F2F5F7;--card:#FBFCFD;--ink:#131A21;--soft:#4A5964;--faint:#7E8D98;
--rule:#D2DBE2;--open:#1F7A8C;--wall:#55636D;--hot:#C0451A;
--warnbg:rgba(196,156,30,.12);--warn:#8A6A12;
--shadow:0 1px 2px rgba(19,26,33,.05),0 8px 24px rgba(19,26,33,.06)}
:root:not([data-theme="light"]){@media (prefers-color-scheme:dark){
--paper:#0D1216;--card:#141B21;--ink:#E4EBF0;--soft:#9DAAB4;--faint:#6B7A85;
--rule:#263139;--open:#5FB8C9;--wall:#8697A3;--hot:#F0764A;
--warnbg:rgba(217,179,74,.10);--warn:#D9B34A;
--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35)}}
:root[data-theme="dark"]{--paper:#0D1216;--card:#141B21;--ink:#E4EBF0;--soft:#9DAAB4;
--faint:#6B7A85;--rule:#263139;--open:#5FB8C9;--wall:#8697A3;--hot:#F0764A;
--warnbg:rgba(217,179,74,.10);--warn:#D9B34A;
--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35)}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI","Malgun Gothic",sans-serif;
font-size:15px;line-height:1.65;-webkit-font-smoothing:antialiased}
.sheet{max-width:1080px;margin:0 auto;padding:48px 28px 90px;display:flex;flex-direction:column;gap:44px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11.5px;font-weight:500;
letter-spacing:.14em;text-transform:uppercase;color:var(--faint)}
h1{margin:.35em 0 0;font-size:clamp(27px,4vw,38px);font-weight:600;letter-spacing:-.02em;
line-height:1.15;text-wrap:balance}
.sub{margin:10px 0 0;color:var(--soft);max-width:70ch}
.meta{margin-top:14px;font-family:"IBM Plex Mono",monospace;font-size:12.5px;
color:var(--faint);font-variant-numeric:tabular-nums}
h2{margin:0;font-size:13px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
color:var(--soft);padding-bottom:8px;border-bottom:1px solid var(--rule);
display:flex;align-items:baseline;gap:10px}
h2 .num{font-family:"IBM Plex Mono",monospace;color:var(--faint);font-weight:500}
section{display:flex;flex-direction:column;gap:16px}
p{margin:0;max-width:72ch;color:var(--soft)}
p.body{color:var(--ink)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:18px 20px;
box-shadow:var(--shadow);display:flex;flex-direction:column;gap:4px}
.card b{font-family:"IBM Plex Mono",monospace;font-size:30px;font-weight:600;
letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1.1}
.card span{font-size:12.5px;color:var(--faint)}
.card.hot b{color:var(--hot)}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;background:var(--card)}
table{border-collapse:collapse;width:100%;min-width:520px;font-size:14px}
th,td{text-align:left;padding:10px 16px;border-bottom:1px solid var(--rule);vertical-align:top}
thead th{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.1em;
text-transform:uppercase;color:var(--faint);font-weight:500;background:var(--paper)}
tbody tr:last-child td{border-bottom:0}
td.mono,th.mono{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
td.k{width:34%;color:var(--ink);font-weight:500}
.player{margin:0;background:var(--card);border:1px solid var(--rule);border-radius:3px;
padding:16px;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:12px}
.player img{display:block;width:100%;height:auto;border-radius:2px;background:#0d1014}
.ctl{display:flex;align-items:center;gap:14px}
.ctl button{width:36px;height:32px;border:1px solid var(--rule);border-radius:3px;
background:var(--paper);color:var(--ink);cursor:pointer;font-size:12px;line-height:1}
.ctl button:hover{border-color:var(--hot);color:var(--hot)}
.ctl button:focus-visible{outline:2px solid var(--hot);outline-offset:2px}
.ctl input[type=range]{flex:1;accent-color:var(--hot)}
.tval{font-family:"IBM Plex Mono",monospace;font-size:13px;font-variant-numeric:tabular-nums;
min-width:74px;text-align:right;color:var(--soft)}
figcaption{font-size:13px;color:var(--soft);border-top:1px solid var(--rule);padding-top:10px}
.bar{display:flex;height:12px;border-radius:2px;overflow:hidden;border:1px solid var(--rule)}
.bar i{flex:1}
.scale{display:flex;justify-content:space-between;font-family:"IBM Plex Mono",monospace;
font-size:11.5px;color:var(--faint);font-variant-numeric:tabular-nums}
.spark{width:100%;max-width:620px;height:auto}
.spark .ax{stroke:var(--rule);stroke-width:1}
.spark .ln{fill:none;stroke:var(--hot);stroke-width:2;stroke-linejoin:round}
.spark .pt{fill:var(--hot)}
.spark .tick{font-family:"IBM Plex Mono",monospace;font-size:10.5px;fill:var(--faint)}
.warn{background:var(--warnbg);border-left:3px solid var(--warn);border-radius:0 3px 3px 0;
padding:18px 22px;display:flex;flex-direction:column;gap:10px}
.warn h3{margin:0;font-size:14px;font-weight:600;color:var(--ink)}
code{font-family:"IBM Plex Mono",monospace;font-size:.92em;
background:rgba(85,99,109,.16);padding:1px 5px;border-radius:2px}
footer{border-top:1px solid var(--rule);padding-top:18px;font-size:13px;color:var(--faint)}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def build(frames_dir, run, out_path):
    with open(os.path.join(frames_dir, "frames.json"), encoding="utf-8") as fh:
        man = json.load(fh)

    frames = man["frames"]
    lo, hi = man["range"]

    images = {}
    for f in frames:
        for suffix in ("_3d", "_section", "_zceiling", "_zbreath"):
            p = os.path.join(frames_dir, f["stem"] + suffix + ".png")
            if os.path.exists(p):
                images[f["stem"] + suffix] = b64(p)

    def sub(series, suffix):
        return [{**f, "stem": f["stem"] + suffix} for f in series]

    stops = [(10,16,34),(38,26,96),(99,30,128),(168,44,105),
             (222,78,58),(245,141,30),(252,202,62),(255,248,214)]
    bar = "".join(f'<i style="background:rgb{c}"></i>' for c in stops)

    cond_rows = "".join(
        f'<tr><td class="k">{esc(k)}</td><td class="mono">{esc(v)}</td></tr>'
        for k, v in run["conditions"]
    )

    perf_rows = "".join(
        f'<tr><td class="k">{esc(k)}</td><td class="mono">{esc(v)}</td></tr>'
        for k, v in run["performance"]
    )

    cards = "".join(
        f'<div class="card{" hot" if c.get("hot") else ""}">'
        f'<b>{esc(c["value"])}</b><span>{esc(c["label"])}</span></div>'
        for c in run["cards"]
    )

    frame_rows = "".join(
        f'<tr><td class="mono">{f["t"]:.0f}</td>'
        f'<td class="mono">{f["min"]:.1f}</td>'
        f'<td class="mono">{f["max"]:.1f}</td>'
        f'<td class="mono">{f["mean"]:.1f}</td>'
        f'<td class="mono">{f["ceiling_max"]:.1f}</td>'
        f'<td class="mono">{f["breath_max"]:.1f}</td></tr>'
        for f in frames
    )

    html = f"""<title>{esc(run["title"])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>{CSS}</style>

<div class="sheet">
<header>
  <div class="eyebrow">ofgpu · transient result</div>
  <h1>{esc(run["heading"])}</h1>
  <p class="sub">{run["lede"]}</p>
  <div class="meta">{esc(run["metaline"])}</div>
</header>

<section>
  <h2><span class="num">01</span> 요약</h2>
  <div class="cards">{cards}</div>
</section>

<section>
  <h2><span class="num">02</span> 해석 조건</h2>
  <div class="tablewrap"><table><tbody>{cond_rows}</tbody></table></div>
</section>

<section>
  <h2><span class="num">03</span> 3-D 온도장 &nbsp;— &nbsp;체적 렌더링</h2>
  {player("v3d", sub(frames, "_3d"), images,
          "해석영역 전체의 온도장을 등각 투상으로 적분 렌더링. 주위 공기는 투명하게 두어 플룸만 보이도록 했습니다.")}
  <div class="bar">{bar}</div>
  <div class="scale"><span>{lo:.0f} K ({lo-273.15:.0f} °C)</span>
    <span>{(lo+hi)/2:.0f} K</span><span>{hi:.0f} K ({hi-273.15:.0f} °C)</span></div>
</section>

<section>
  <h2><span class="num">04</span> 종단면 &nbsp;— &nbsp;유입구 중심 (y = 0)</h2>
  {player("vsec", sub(frames, "_section"), images,
          "x–z 단면. 왼쪽이 xMin 벽, 오른쪽이 outlet(개방). 바닥 중앙에서 올라간 고온 기류가 천장을 따라 흐르다 우측으로 빠져나가는 경로가 보입니다.")}
</section>

<section>
  <h2><span class="num">05</span> 수평면 온도 분포</h2>
  {player("vceil", sub(frames, "_zceiling"), images,
          "천장 하부 z = 2.8 m 평면. FDS 비교 보고서와 같은 높이입니다.")}
  {player("vbre", sub(frames, "_zbreath"), images,
          "호흡 높이 z = 1.5 m 평면.")}
</section>

<section>
  <h2><span class="num">06</span> 시간 이력</h2>
  <div class="cards" style="grid-template-columns:repeat(auto-fit,minmax(280px,1fr))">
    <div class="card"><span>최고 온도 [K]</span>{sparkline(frames,"max")}</div>
    <div class="card"><span>영역 평균 온도 [K]</span>{sparkline(frames,"mean")}</div>
    <div class="card"><span>천장면(z=2.8 m) 최고 온도 [K]</span>{sparkline(frames,"ceiling_max")}</div>
  </div>
  <div class="tablewrap"><table>
    <thead><tr><th class="mono">t [s]</th><th class="mono">min</th><th class="mono">max</th>
    <th class="mono">mean</th><th class="mono">z=2.8 max</th><th class="mono">z=1.5 max</th></tr></thead>
    <tbody>{frame_rows}</tbody></table></div>
  <p>모든 값은 K 단위입니다.</p>
</section>

<section>
  <h2><span class="num">07</span> 연산 성능</h2>
  <div class="tablewrap"><table><tbody>{perf_rows}</tbody></table></div>
</section>

<section>
  <h2><span class="num">08</span> 이 결과의 범위</h2>
  <div class="warn">
    <h3>{esc(run["caveat_title"])}</h3>
    {"".join(f"<p>{p}</p>" for p in run["caveat"])}
  </div>
</section>

<footer>{run["footer"]}</footer>
</div>
"""

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)

    size = os.path.getsize(out_path) / 1e6
    print(f"wrote {out_path}  ({size:.1f} MB, {len(frames)} frames)")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    with open(sys.argv[2], encoding="utf-8") as fh:
        run = json.load(fh)
    build(sys.argv[1], run, sys.argv[3])
    return 0


if __name__ == "__main__":
    sys.exit(main())
