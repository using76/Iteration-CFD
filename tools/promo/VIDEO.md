<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
     No GPL-licensed source was consulted. -->

# The F1 promo video (PROMO-VIDEO, 2026-10-05)

Record of the 82-second F1 promo film. The composition source lives in
`tools/promo/video/` (this repository, text only); every render, capture clip,
still and the finished MP4 is made from the CC BY 4.0 car model and stays
outside the repository, under the HyperFrames build project.

## What the film says

"An in-house GPU CFD suite takes a real F1 car from CAD to a resolved flow
field and studio visualisation, driven in plain Korean by an AI assistant."
1920x1080, 82 s, silent, Korean captions first and English second, for
engineers and engineering managers evaluating a CFD toolchain.

## Sources

`tools/promo/video_assets.py stage` copies the composition HTML into the
build project and encodes the six film videos (ffmpeg 9, libx264, CRF 14,
yuv420p, 30 fps, `-frames:v` pinned to the expected count, every output
re-probed):

| asset | source | segment | crop | scale | speed | hold | frames |
|---|---|---|---|---|---|---|---|
| `assets/cad_turntable.mp4` | `<render>/turntable_clean/frame_%04d.png` (240 files) | all | - | - | - | 0 | 240 |
| `assets/cp_turntable.mp4` | `<render>/turntable_cp/frame_%04d.png` (240 files) | all | - | - | - | 0 | 240 |
| `assets/reveal.mp4` | `<render>/reveal/frame_%04d.png` (150 files) | all | - | - | - | 1.5 | 195 |
| `assets/chat.mp4` | `<capture>/clips/s1_geometry.mp4` | ss 0, t 130.5 s | - | - | 9.0x | 1.5 | 480 |
| `assets/mesh.mp4` | `<capture-v1>/clips/s2_mesh.mp4` | ss 348.5, t 9.0 s | `1040:585:460:180` | `1920:1080` | 1.0 | 0 | 270 |
| `assets/residuals.mp4` | `<capture>/clips/s5_residuals.mp4` | ss 0, t 8.3 s | `1170:640:330:84` | `1756:960` | 1.0 | 0.7 | 270 |

Besides the six videos it copies the five stills (`hero`, `side`, `top_rear`,
`streamlines`, `slice`) byte-identical, the Noto Sans KR variable font
(SIL OFL 1.1) as `assets/fonts/NotoSansKR-VF.ttf`, and the CC BY 4.0 model's
`CREDITS.txt`/`LICENSE.txt`, then writes `assets/assets.json` (sha256,
frames, size and ffmpeg arguments per video). Measured 2026-10-05 on the real
inputs: `STAGE OK 6 videos 5 stills 12 sources 18.5 s`, `CHECK PASS 26/26`,
selftest `SELFTEST PASS 6/6` in 1.1 s.

## Timeline

| id | start | dur | content |
|---|---|---|---|
| title | 0 | 5 | kicker, Korean/English headline over the faint streamline still, waterfall in |
| cad | 5 | 8 | the clean red-car turntable, 1.00 -> 1.03 drift, CC BY attribution bottom-right |
| chat | 13 | 16 | the studio driven in Korean at 9x, prompt card 0.6-6.0 s, push into the answer (target 1387,650, scale 1.8) at 9.0-11.5 s |
| mesh | 29 | 9 | the castellated mesh with edges, count-up to 3,235,813 cells at 0.6-3.6 s, 1.00 -> 1.04 drift |
| solve | 38 | 9 | residual chart wiped in (clip-path 0.5-4.5 s) beside four tiles: RTX 5070 Ti, 250 km/h, 1,000 steps, about 2 h 17 min |
| cpturn | 47 | 7 | the Cp turntable, colour bar kept clear, no overlay |
| cpstills | 54 | 7.5 | hero / side / top_rear stills, 2.5 s each, 0.5 s crossfades at 2.5 and 5.0, per-still drift |
| reveal | 61.5 | 6.5 | the 150-frame streamline reveal (+1.5 s hold) |
| slice | 68 | 5 | the y = 0 speed slice still, 1.00 -> 1.04 drift |
| close | 73 | 9 | Iterations-CFD / meteor-cfd card, company line, the four credit lines, the demonstration disclaimer, fade out over 8.2-9.0 s |
| captions | 0 | 82 | the bilingual caption track below and the demonstration chip |

Every scene sits in one wrap that fades in over 0.5 s and out over the last
0.4 s (title has no fade-in, close fades out over the last 0.8 s). GSAP
timelines are paused and registered per composition; the root timeline is
near-empty. Design tokens: bg `#0e0f12`, surface `#171a1f`, fg `#f1f3f5`,
muted `#a7afb8`, accent `#ff5a36`; Korean and English text in Noto Sans KR,
numbers and technical labels in IBM Plex Mono with tabular numerals.

## Captions

| key | in | out | KO | EN |
|---|---|---|---|---|
| cad | 5.6 | 12.5 | `실제 F1 2026 컨셉 CAD 모델` | `A real F1 2026 concept CAD model` |
| chat1 | 13.6 | 21.0 | `한국어로 요청하면, AI 어시스턴트가 스튜디오를 직접 조작합니다` | `Ask in plain Korean - the AI assistant drives the studio itself` |
| chat2 | 21.3 | 28.5 | `형상을 열고, 해석 결과를 불러와 화면에 띄웁니다` | `It opens the geometry, loads the solve and puts it on screen` |
| mesh | 29.6 | 37.5 | `자체 격자 생성기 · 계단형 육면체 격자` | `In-house mesher - castellated hex mesh` |
| solve | 38.6 | 46.5 | `GPU 한 장으로 1,000 스텝 의사-과도 해석` | `1,000 pseudo-transient steps on a single GPU` |
| cp | 47.6 | 61.0 | `차체 표면 압력 분포 (Cp)` | `Surface pressure on the car (Cp)` |
| reveal | 62.0 | 67.5 | `유선 - 앞날개, 바퀴, 뒷날개 주위의 흐름` | `Streamlines around the front wing, wheels and rear wing` |
| slice | 68.6 | 72.5 | `차량 중앙 단면의 유속 · 자유류 250 km/h` | `Flow speed on the car's centre plane - free stream 250 km/h` |

The demonstration chip (`시연용 계산 · 검증된 공력 결과 아님 | Demonstration
run - not a validated aerodynamic result`) sits top-left from 47.2 s to
72.8 s.

## Honesty rules

- No aerodynamic coefficient appears anywhere in the film: the strings Cd,
  Cl, drag, downforce (Korean included) are verified absent by a grep of every composition
  file (V6). The numbers shown are cells, wall time, GPU, steps and
  speed only.
- The residual chart is cropped to the chart panel so the chat panel that
  shows the coefficients is cut away; capture v2 s4 (the Korean answer that
  prints them) is not used at all.
- The mesh shown is the castellated, unsnapped staircase hex mesh the full run
  actually wrote (snap abandoned all 30 iterations), not an idealised one.
- The solve is classed unsteady by the house stopping rule; the caption says
  pseudo-transient, and the closing card says the run is a demonstration, not
  a validated aerodynamic result.
- The red failed-tool cards that flash by in the chat time-lapse are left in.
- Colour-bar tick values baked into the renders are field legends, not claims.

## Decisions (by recommendation, recorded in the project BRIEF)

- No voice-over: captions carry the story bilingually.
- No music: no licence-clean bgm track could be resolved, so the film is
  silent.
- Product names spelled as the repository READMEs spell them:
  "Iterations-CFD" and "meteor-cfd".
- The credit lines are verbatim from `promo/render/CREDITS.txt`.

## Credits

```
Car model: "F1 2026 concept" by Qvist_Designs, CC BY 4.0, via Sketchfab.
Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
Changes: scaled to metres, closed into a watertight simulation surface, meshed and solved with meteor-cfd; the colours on the car, the streamlines and the slice are CFD results and are not part of the original model.
Rendered with Blender 5.1.2 (Cycles).
```

## How to reproduce

```
python tools/promo/video_assets.py stage \
  --render <scratch>/promo/render --capture <scratch>/promo/capture \
  --capture-v1 <scratch>/promo/capture-v1 \
  --font C:/Windows/Fonts/NotoSansKR-VF.ttf --project <build>/video
python tools/promo/video_assets.py check --project <build>/video
cd <build>/video && npx --yes hyperframes@0.8.122 check
cd <build>/video && npx --yes hyperframes@0.8.122 render --quality delivery --fps 30 --output renders/iteration-cfd-f1-promo.mp4
```

## Render (2026-10-05)

The supervisor rendered the film in a visible console with
`npx --yes hyperframes@0.8.122 render --quality delivery --fps 30`: H.264
yuv420p 1920x1080 at 30 fps, 2460 frames, 82.000 s, no audio stream, 73.6 MB
(sha256 prefix bea90a42), rendered in 3 m 15.2 s (capture 3 m 6.7 s,
hardware GPU, one worker). Before the render `hyperframes check` passed with 0
lint/runtime/layout/motion findings and 38/38 WCAG AA contrast checks. A copy
is at `C:/Users/sdd32/Videos/iteration-cfd-f1-promo.mp4` with the model's
`CREDITS.txt` beside it.

No output file of this run is in the repository - only this record and the
composition source.
