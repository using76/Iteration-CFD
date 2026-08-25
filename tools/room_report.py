#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
"""One-file HTML report for the serpentine room case.

Two contour panels (T, |U|) share a single time slider; every image is
base64-inlined so the page has no external assets.

    python room_report.py <slicesDir> <out.html>
"""

import base64
import json
import os
import sys


def b64(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def spark(points, w=640, h=90, pad=10, lo=None, hi=None, unit=""):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lo = min(ys) if lo is None else lo
    hi = max(ys) if hi is None else hi
    rng = max(hi - lo, 1e-30)
    X = lambda t: pad + (t - xs[0]) / max(xs[-1] - xs[0], 1e-30) * (w - 2 * pad)
    Y = lambda v: h - pad - (v - lo) / rng * (h - 2 * pad)
    d = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in points)
    last_t, last_v = points[-1]
    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
        f'<polyline points="{d}" fill="none" stroke="var(--accent)" stroke-width="2"/>'
        f'<circle cx="{X(last_t):.1f}" cy="{Y(last_v):.1f}" r="3.5" fill="var(--accent)"/>'
        f'<text x="{w-pad}" y="{Y(last_v)-8:.1f}" text-anchor="end" class="sv">{last_v:.1f}{unit}</text>'
        f"</svg>"
    )


def main():
    sdir, out = sys.argv[1], sys.argv[2]
    meta = json.load(open(os.path.join(sdir, "frames.json")))
    frames = [f for f in meta["frames"] if f["t"] >= 2.0 and abs(f["t"] % 2.0) < 1e-9]
    times = [f["t"] for f in frames]
    t_imgs = json.dumps([b64(f["T"]) for f in frames])
    u_imgs = json.dumps([b64(f["U"]) for f in frames])

    tmax_pts = [(f["t"], f["Tmax"] - 273.15) for f in frames]
    tmean_pts = [(f["t"], f["Tmean"] - 273.15) for f in frames]
    umax_pts = [(f["t"], f["Umax"]) for f in frames]

    html = f"""<title>사행 유로 열기류</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>
  :root {{
    --bg: #101014; --surface: #17171d; --line: #2a2a33;
    --text: #e8e8ee; --muted: #9a9aa6; --accent: #ff8c3c;
    --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
  }}
  body {{
    background: var(--bg); color: var(--text);
    font: 15px/1.65 "IBM Plex Sans KR", "Pretendard", "Malgun Gothic", sans-serif;
    margin: 0; padding: 0 16px 64px;
  }}
  main {{ max-width: 1160px; margin: 0 auto; }}
  header {{ padding: 40px 0 8px; border-bottom: 1px solid var(--line); }}
  h1 {{ font-size: 26px; font-weight: 600; margin: 0 0 6px; text-wrap: balance; }}
  .sub {{ color: var(--muted); margin: 0 0 18px; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 20px; }}
  .chip {{
    background: var(--surface); border: 1px solid var(--line); border-radius: 6px;
    padding: 4px 12px; font-family: var(--mono); font-size: 12.5px;
    color: var(--text); font-variant-numeric: tabular-nums;
  }}
  .chip b {{ color: var(--accent); font-weight: 600; }}
  h2 {{ font-size: 17px; font-weight: 600; margin: 40px 0 12px;
        letter-spacing: .01em; }}
  h2 .k {{ color: var(--muted); font-weight: 400; font-size: 13px; margin-left: 8px; }}
  .panes {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  @media (max-width: 900px) {{ .panes {{ grid-template-columns: 1fr; }} }}
  .pane {{ background: var(--surface); border: 1px solid var(--line);
           border-radius: 8px; padding: 10px; }}
  .pane img {{ width: 100%; height: auto; display: block; border-radius: 4px;
               image-rendering: pixelated; }}
  .bar {{ display: flex; align-items: center; gap: 14px; margin: 14px 0 6px;
          background: var(--surface); border: 1px solid var(--line);
          border-radius: 8px; padding: 10px 14px; }}
  .bar button {{
    background: var(--accent); color: #1a1006; border: 0; border-radius: 6px;
    font: 600 14px/1 "IBM Plex Sans KR", sans-serif; padding: 9px 16px;
    cursor: pointer; min-width: 74px;
  }}
  .bar button:focus-visible {{ outline: 2px solid var(--text); outline-offset: 2px; }}
  .bar input[type=range] {{ flex: 1; accent-color: var(--accent); }}
  .t {{ font-family: var(--mono); font-size: 15px; min-width: 88px;
        text-align: right; font-variant-numeric: tabular-nums; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  .tblwrap {{ overflow-x: auto; }}
  th, td {{ text-align: left; padding: 7px 14px 7px 0; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--muted); font-weight: 500; font-size: 12.5px;
        text-transform: uppercase; letter-spacing: .06em; }}
  td.num, .mono {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
  .sparks {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
             gap: 14px; }}
  .spark {{ background: var(--surface); border: 1px solid var(--line);
            border-radius: 8px; padding: 12px 14px 6px; }}
  .spark .lab {{ color: var(--muted); font-size: 12.5px; margin-bottom: 4px; }}
  .spark svg {{ width: 100%; height: 90px; }}
  .sv {{ fill: var(--text); font: 12px var(--mono); }}
  .note {{ background: var(--surface); border-left: 3px solid var(--accent);
           border-radius: 0 8px 8px 0; padding: 12px 16px; color: var(--muted);
           font-size: 14px; }}
  .note b {{ color: var(--text); }}
  footer {{ margin-top: 48px; color: var(--muted); font-size: 12.5px;
            border-top: 1px solid var(--line); padding-top: 14px; }}
</style>
<main>
<header>
  <h1>사행 유로 열기류</h1>
  <p class="sub">지그재그 배플 실내 부력유동 · meteor-cfd (ofgpu-buoyant) · z = {meta['z']:.3f} m 수평 단면</p>
  <div class="chips">
    <span class="chip">공간 <b>10 × 10 × 3 m</b></span>
    <span class="chip">격자 <b>80 × 80 × 24</b> · 144,960 유체셀</span>
    <span class="chip">유입 <b>2 m/s · 300 °C</b> (−x 전면)</span>
    <span class="chip">유출 <b>2 × 2 m 문</b> (+x, 개방)</span>
    <span class="chip">해석 <b>30 s</b> · Δt 5 ms</span>
    <span class="chip">벽시계 <b>125 s</b> · RTX 5070 Ti</span>
  </div>
</header>

<h2>온도 · 속도 컨투어 <span class="k">2초 간격 · 슬라이더 공유 · ←/→ 키 지원</span></h2>
<div class="bar">
  <button id="play" aria-label="재생/정지">▶ 재생</button>
  <input id="sl" type="range" min="0" max="{len(frames)-1}" step="1" value="0"
         aria-label="시간 선택">
  <span class="t" id="tv">t = {times[0]:g} s</span>
</div>
<div class="panes">
  <div class="pane"><img id="imT" alt="온도 컨투어"></div>
  <div class="pane"><img id="imU" alt="속도 크기 컨투어"></div>
</div>

<h2>시계열 <span class="k">단면 전체에 대한 통계</span></h2>
<div class="sparks">
  <div class="spark"><div class="lab">단면 최고 온도 [°C]</div>{spark(tmax_pts, unit="°C")}</div>
  <div class="spark"><div class="lab">단면 평균 온도 [°C]</div>{spark(tmean_pts, unit="°C")}</div>
  <div class="spark"><div class="lab">단면 최대 |U| [m/s]</div>{spark(umax_pts, unit=" m/s")}</div>
</div>

<h2>해석 조건</h2>
<div class="tblwrap"><table>
  <tr><th>항목</th><th>설정</th></tr>
  <tr><td>지오메트리</td><td class="mono">10 × 10 × 3 m 실내, 사행 배플 3장 (두께 0.25 m, 전고) — STL로 조각(castellation)</td></tr>
  <tr><td>배플 배치</td><td class="mono">x = 2.5 (y 0–7.5) · x = 5.0 (y 2.5–10) · x = 7.5 (y 0–7.5) — 갭 교차로 사행 유로</td></tr>
  <tr><td>유입</td><td class="mono">−x 전면 · fixedValue U = (2, 0, 0) m/s · T = 573.15 K</td></tr>
  <tr><td>유출</td><td class="mono">+x 문 (y 4–6, z 0–2 m) · p = 0 · T/U inletOutlet</td></tr>
  <tr><td>부력</td><td class="mono">밀도비 b = g(T<sub>ref</sub>/T − 1) · g = (0, 0, −9.81) · T<sub>ref</sub> = 293.15 K</td></tr>
  <tr><td>난류</td><td class="mono">표준 k-ε · 벽함수 (kqR / epsilon / nutk)</td></tr>
  <tr><td>수치</td><td class="mono">PISO 계열 과도해석 · Euler 시간전진 · 대류 upwind(운동량 linearUpwind) · Δt = 5 ms (문 Co ≈ 0.1)</td></tr>
  <tr><td>선형해법</td><td class="mono">압력 backend 자동선택 · PBiCGStab / 다색 DILU</td></tr>
  <tr><td>성능</td><td class="mono">6,000 스텝 / 125 s = 20.8 ms/스텝 · 벽시계 4.2 s per 해석-s</td></tr>
</table></div>

<h2>읽는 법</h2>
<p class="note">
  단면은 <b>z = 2.81 m(천장 근처)</b>의 열층입니다. 열기류가 첫 배플의
  <b>상부(y 7.5–10) 갭</b>으로 넘어가 두 번째 챔버를 채우고, 하부 갭 → 상부 갭
  순서로 사행하며 전진합니다. 문(z &lt; 2 m)은 이 단면보다 낮아 컨투어에는
  표식으로만 나타납니다. 회색 띠가 배플, 좌측 가장자리의 고온 띠가 유입면입니다.
  30초 시점에도 마지막 챔버는 승온 초기 단계로, 사행 유로가 열전선 도달을
  지연시키는 효과가 그대로 보입니다.
</p>

<footer>
  meteor-cfd · 주식회사 메테오시뮬레이션 · GPU 상주 유한체적 해석 ·
  케이스 <span class="mono">room</span> + <span class="mono">baffles.stl</span> ·
  결과는 검증 목적의 데모이며 설계 판단에는 독립 검증이 필요합니다.
</footer>
</main>
<script>
  const T = {t_imgs};
  const U = {u_imgs};
  const times = {json.dumps(times)};
  const sl = document.getElementById("sl"), tv = document.getElementById("tv");
  const imT = document.getElementById("imT"), imU = document.getElementById("imU");
  const play = document.getElementById("play");
  let timer = null;
  function show(i) {{
    imT.src = T[i]; imU.src = U[i];
    tv.textContent = "t = " + times[i] + " s";
    sl.value = i;
  }}
  function stop() {{ if (timer) clearInterval(timer); timer = null; play.textContent = "▶ 재생"; }}
  play.onclick = () => {{
    if (timer) return stop();
    play.textContent = "❚❚ 정지";
    timer = setInterval(() => {{
      let i = (+sl.value + 1) % T.length;
      show(i);
      if (i === T.length - 1) stop();
    }}, 600);
  }};
  sl.oninput = () => {{ stop(); show(+sl.value); }};
  document.addEventListener("keydown", e => {{
    if (e.key === "ArrowRight") {{ stop(); show(Math.min(+sl.value + 1, T.length - 1)); }}
    if (e.key === "ArrowLeft")  {{ stop(); show(Math.max(+sl.value - 1, 0)); }}
  }});
  show(0);
</script>
"""
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"{out}: {os.path.getsize(out)/1e6:.1f} MB, {len(frames)} frames")


if __name__ == "__main__":
    main()
