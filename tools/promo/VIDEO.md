<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
     No GPL-licensed source was consulted. -->

# The F1 promo video (PROMO-VIDEO-V2, 2026-10-05)

Record of the 90-second F1 promo film (v2). The composition source lives in
`tools/promo/video/` (this repository, text only); every render, capture clip,
still, the soundtrack and the finished MP4 are made from the CC BY 4.0 car
model and stay outside the repository, under the HyperFrames build project.

## What the film says

"Iterations' AI-driven CFD solver: an F1 race car goes from CAD to a resolved
flow field, set up by AI and solved on one GPU, fast and automatic."
1920x1080, 90 s, English narration (12 lines, local Kokoro-82M) with Korean
captions only, over a licensed music bed and timed SFX, for engineers and
engineering managers evaluating a CFD toolchain. The film opens on the
Iterations mark E "damped residual" lockup ((주)이터레이션즈 ·
Iterations Co., Ltd., subtitle "F1 레이싱카 공력해석 테스트") and closes on
"AI-driven CFD Solver".

## Sources

`tools/promo/video_assets.py stage` copies `tools/promo/video/index.html` and
every `tools/promo/video/compositions/*.html` into the build project and
encodes the six film videos (ffmpeg 9, libx264, CRF 14, yuv420p, 30 fps,
`-frames:v` pinned to the expected count, every output re-probed):

| asset | source | segment | crop | scale | speed | hold | frames |
|---|---|---|---|---|---|---|---|
| `assets/cad_turntable.mp4` | `<render>/turntable_clean/frame_%04d.png` (240 files) | all | - | - | - | 0 | 240 |
| `assets/cp_turntable.mp4` | `<render>/turntable_cp/frame_%04d.png` (240 files) | all | - | - | - | 0 | 240 |
| `assets/reveal.mp4` | `<render>/reveal/frame_%04d.png` (150 files) | all | - | - | - | 1.5 | 195 |
| `assets/chat.mp4` | `<capture>/clips/s1_geometry.mp4` | ss 0, t 130.5 s | - | - | 9.0x | 1.5 | 480 |
| `assets/mesh.mp4` | `<capture-v1>/clips/s2_mesh.mp4` | ss 348.5, t 9.0 s | `1040:585:460:180` | `1920:1080` | 1.0 | 0 | 270 |
| `assets/residuals.mp4` | `<capture>/clips/s5_residuals.mp4` | ss 0, t 8.3 s | `1170:640:330:84` | `1756:960` | 1.0 | 1.7 | 300 |

Besides the six videos it copies the five stills (`hero`, `side`, `top_rear`,
`streamlines`, `slice`) byte-identical, the Noto Sans KR variable font
(SIL OFL 1.1) as `assets/fonts/NotoSansKR-VF.ttf`, and the CC BY 4.0 model's
`CREDITS.txt`/`LICENSE.txt`, then writes `assets/assets.json` (sha256,
frames, size and ffmpeg arguments per video). Measured 2026-10-05 on the real
inputs: `STAGE OK 6 videos 5 stills 12 sources 23.3 s`, `CHECK PASS 26/26`,
selftest `SELFTEST PASS 6/6`.

## Soundtrack

`tools/promo/video/audio/soundtrack.json` is the single source: 10 scenes
summing to 90.0 s, 12 VO lines, 33 SFX cues (audio v3), the music descriptor,
the mix specification and the master gates.

- **Music** — one continuous licensed excerpt, no joins: HeyGen audio library
  id `a0dc53f249ae416f8a7dcc15912581a1` (`assets/music/src/a0dc53f2.mp3`,
  sha256 `d00372d5...46a5ca6`), 120 BPM, steady grid, src 0-90 s -> film
  0-90 s at `bed_gain_db` -11.0, its native final hit at film 79.85 s. Chosen
  by the supervisor's library scan (see `assets/music/src/SOURCES.json` in
  the build project).
- **SFX** — 33 cues from the licensed HeyGen library set copied (not edited)
  into `assets/sfx/src` (index: `assets/sfx/SFX_INDEX.json` in the build
  project); the v2 build placed 36, three of them the synthesised `sub`
  thump, which audio v3 removed (kind `sub` stays synthesised in
  `video_audio.py`: an exponential 78 -> 40 Hz sine sweep over 0.9 s,
  envelope exp(-t/0.35), plus a 60 Hz, 0.12 s raised-cosine thump, peak 0.9
  before gain - no v3 cue uses it).
- **Voice** — local Kokoro-82M, voice `am_michael`, through
  `npx --yes hyperframes@0.8.122 tts` with the scratch kokoro venv as
  `HYPERFRAMES_PYTHON`; `V10` at speed 1.05, everything else at 1.0. Measured
  clip seconds land in `assets/voice/voice.json`.
- **Chain** (`video_audio.py build`) — music bed (decode -> src slice ->
  -11 dB) + VO track (placed at round(start*SR), mono to both) with a presence
  envelope p(t) (20 ms RMS windows, gate -45 dBFS, hold 0.20 s, slew limits
  1/0.12 s up and 1/0.45 s down); the bed is carved -2 dB at 180-420 Hz and
  -4.5 dB at 900-4200 Hz (Butterworth order 4, zero-phase sosfiltfilt) and
  ducked -6 dB by 10^(-6*p/20); the SFX bank is placed from the cue list;
  master to -14 LUFS / <= -2.0 dBTP with its own BS.1770-4 integrated meter
  and a 4x-oversampled true-peak limiter (per-sample required gain, forward
  maximum hold over the 0.005 s lookahead, one-pole release 0.08 s,
  gain+limit repeated to convergence), then verified with ffmpeg
  `ebur128=peak=true`.
- **Measured this run** (run 2, the voice clips are run 1's, re-verified by
  `check`): `BUILD OK master I -14.00 LUFS TP -2.00 dBTP 17.6 s` (first
  attempt, ceiling -2.0), master 48 kHz 24-bit stereo, exactly 4,320,000
  samples, LRA 2.8 LU, master sha256 prefix 043085fc,
  `python tools/promo/video_audio.py check` -> `CHECK PASS 61/61`. ebur128 on
  the stems over the VO windows now gives vo -20.4..-20.9 LUFS against music
  -30.1..-31.3 LUFS (a 9.2-10.9 dB gap); at run 1's bed -6.0 the same windows
  measured vo -20.5..-20.9 against music -25.1..-26.3, only ~5 dB.
- **ASR** (`faster-whisper` small, CPU, in its own venv): `ASR PASS stem
  12/12 master 12/12`. The judge cuts each line's [start - 0.3, end + 0.5]
  excerpt out of stems/vo.wav and master.wav into a temp 16 kHz mono WAV and
  judges the line on all the words of its own excerpt, with no time filter
  (the two whole-file transcripts are still made in the same `_whisper` call
  and kept in asr.json under `full`). The reason: faster-whisper's word
  timestamps drift at segment boundaries after a pause, so run 1's whole-file
  window filter stamped V07's "pressure" at 47.24/47.26 s - 0.56 s before the
  clip's film start of 48.1 s, attaching it to V06's tail across the pure
  silence at 46.945-48.1 s. The words are spoken in the right place by
  construction; the whole-file transcript's clock was the wrong measurement,
  and on its own excerpt V07 reads "Pressure mapped across every surface of
  the car." in both the stem and the master.

## Timeline

| slot | start | dur | content |
|---|---|---|---|
| title | 0 | 6.0 | company opening: the mark-E lockup (damped wave draws on 0.3-1.1 s, dot lands 1.05-1.45 s) with (주)이터레이션즈 · Iterations Co., Ltd. under it, hairline, "F1 레이싱카 공력해석 테스트" / "F1 race-car aerodynamics test" at 2.4-2.55 s, fade out 5.6-6.0 s |
| cad | 6.0 | 7.5 | the clean red-car turntable, 1.00 -> 1.03 drift, CC BY attribution bottom-right |
| chat | 13.5 | 15.0 | the studio driven at 9x, prompt card 0.6-6.0 s, badge out at 12.7 s, push into the answer (target 1387,650, scale 1.8) at 9.0-11.5 s |
| mesh | 28.5 | 9.0 | the castellated mesh with edges, count-up to 3,235,813 cells at 0.6-3.6 s, 1.00 -> 1.04 drift |
| solve | 37.5 | 10.0 | residual chart wiped in (clip-path 0.5-4.5 s) beside four tiles: RTX 5070 Ti, 250 km/h, 1,000 steps, about 2 h 17 min |
| cpturn | 47.5 | 6.0 | the Cp turntable, colour bar kept clear, no overlay |
| cpstills | 53.5 | 6.0 | hero / side / top_rear stills, 2.0 s each, 0.5 s crossfades at 1.75 and 3.75, per-still drift |
| reveal | 59.5 | 6.5 | the 150-frame streamline reveal (+1.5 s hold) |
| slice | 66.0 | 6.0 | the y = 0 speed slice still, 1.00 -> 1.04 drift |
| close | 72.0 | 18.0 | meteor-cfd kicker, "AI-driven CFD Solver" headline at 2.0 s, "Set up by AI. Solved on the GPU. Fast and automatic." at 4.75/6.05/7.9 s, glow pulse on the music's final hit, company block at 10.0-10.24 s, the four credit lines and the demonstration disclaimer, fade out 17.2-18.0 s |
| captions | 0 | 90 | the Korean-only VO-timed caption track and the demonstration chip (47.7-71.8 s) |

Every scene sits in one wrap that fades in over 0.5 s and out over its last
0.4 s (title 0.4 s, close over its last 0.8 s); the audio track is the single
`<audio id="el-master">` element playing `assets/audio/master.wav` over the
whole film. GSAP timelines are paused and registered per composition; the
root timeline is near-empty. Design tokens: bg `#0e0f12`, surface `#171a1f`,
fg `#f1f3f5`, muted `#a7afb8`, accent `#ff5a36`; Korean and English text in
Noto Sans KR, numbers and technical labels in IBM Plex Mono with tabular
numerals.

## Narration and captions

From `assets/audio/audio_meta.json` (end = start + measured clip seconds;
cap times follow the C5 rule). V01 has no caption - the title card already
shows the Korean subtitle.

| id | start | end | English line | Korean caption | cap_in | cap_out |
|---|---|---|---|---|---|---|
| V01 | 1.2 | 5.104 | Iterations. An F1 race-car aerodynamics test. | - | - | - |
| V02 | 6.6 | 12.701 | It starts from a detailed F1 2026 concept car, straight from CAD. | 정교한 F1 2026 컨셉카, CAD 모델에서 바로 시작합니다. | 6.55 | 13.0 |
| V03 | 14.1 | 20.287 | Ask in plain language. The AI assistant can set up the run and drive the studio. | 평범한 말로 요청하면, AI 어시스턴트가 해석을 설정하고 스튜디오를 조작할 수 있습니다. | 14.05 | 20.59 |
| V04 | 21.2 | 26.96 | It opens the car, brings in the mesh and the solution, and puts the results on screen. | 형상을 열고, 격자와 해석 결과를 불러와 화면에 띄웁니다. | 21.15 | 27.26 |
| V05 | 29.0 | 34.739 | Our in-house mesh generator builds 3.2 million cells in about 32 minutes. | 자체 격자 생성기가 약 32분 만에 320만 셀을 만듭니다. | 28.95 | 35.04 |
| V06 | 37.9 | 46.945 | Next, a single GPU, an RTX 5070 Ti, solves the flow at 250 km/h. | 이어서 GPU 한 장, RTX 5070 Ti가 시속 250 km의 유동을 풉니다. | 37.85 | 47.24 |
| V07 | 48.1 | 51.023 | Pressure, mapped across every surface of the car. | 차체 모든 표면의 압력 분포. | 48.05 | 51.32 |
| V08 | 60.0 | 64.8 | Streamlines trace the air over the front wing, the wheels and the rear wing. | 유선이 앞날개, 바퀴, 뒷날개 위로 흐르는 공기를 따라갑니다. | 59.95 | 65.1 |
| V09 | 66.4 | 71.285 | The centre plane shows the wake. From CAD to flow field, end to end. | 중앙 단면이 후류를 보여줍니다. CAD에서 유동장까지, 처음부터 끝까지. | 66.35 | 71.59 |
| V10 | 72.5 | 76.596 | meteor-cfd. The AI-driven CFD solver. | meteor-cfd. AI 기반 CFD 솔버. | 72.45 | 76.6 |
| V11 | 76.7 | 79.751 | Set up by AI. Solved on the GPU. | AI가 설정하고, GPU가 풉니다. | 76.65 | 79.8 |
| V12 | 79.9 | 81.479 | Fast, and automatic. | 빠르고, 자동으로. | 79.85 | 81.78 |

## Honesty rules

- No aerodynamic coefficient appears anywhere in the film: the strings Cd, Cl,
  drag, downforce (Korean included) are verified absent by a grep of every
  composition file and of soundtrack.json (the `check` item `honesty`).
- The VO's only numbers are 3.2 million cells, about 32 minutes (the CPU mesh
  build), 250 km/h and one GPU / RTX 5070 Ti.
- The assistant "can" set up the run - the capture shows it driving the studio
  on a pre-built solve.
- The residual chart is cropped to the chart panel so the chat panel that
  shows the coefficients is cut away; capture v2 s4 (the Korean answer that
  prints them) is not used at all.
- The mesh shown is the castellated, unsnapped staircase hex mesh the full run
  actually wrote (snap abandoned all 30 iterations), not an idealised one.
- The solve is classed unsteady by the house stopping rule; it is the solve
  tile `의사-과도 · pseudo-transient` that says so (the captions are the VO's
  Korean lines), and the closing card says the run is a demonstration, not
  a validated aerodynamic result.
- The red failed-tool cards that flash by in the chat time-lapse are left in.
- Colour-bar tick values baked into the renders are field legends, not claims.

## Decisions (by recommendation, recorded in the project BRIEF)

- English narration, Korean captions only (the v1 bilingual captions are
  gone); the music and SFX are built the way the user's own Shinhwa showreel
  builds its soundtrack (read-only reference, not GPL).
- The master is the only audio element in the composition; stems stay in the
  build project for the ASR and re-mixing.
- The bed sits 5 dB lower than run 1 (`bed_gain_db` -6.0 -> -11.0) so the VO
  rides ~10 dB over the music where it speaks, the reference film's balance;
  the VO, SFX, text, keys and gates are unchanged.
- The ASR judge cuts each line's own excerpt and judges it on its own
  transcript, because faster-whisper's word timestamps drift across a pause at
  segment boundaries - run 1's whole-file window filter measured the
  transcript's clock, not the audio, and failed a line that is spoken in the
  right place (ASR now PASS 12/12 in both files).
- Product names spelled as the repository READMEs spell them:
  "Iterations-CFD" (browser title), "meteor-cfd", "(주)Iterations".
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
python tools/promo/video_audio.py --selftest
python tools/promo/video_assets.py --selftest
python tools/promo/video_audio.py voice --project <build>/video \
  --tts-python <kokoro-venv>/Scripts/python.exe
python tools/promo/video_audio.py build --project <build>/video
python tools/promo/video_assets.py stage --render <scratch>/promo/render \
  --capture <scratch>/promo/capture --capture-v1 <scratch>/promo/capture-v1 \
  --font C:/Windows/Fonts/NotoSansKR-VF.ttf --project <build>/video
python tools/promo/video_assets.py check --project <build>/video
python tools/promo/video_audio.py asr --project <build>/video \
  --asr-python <asr-venv>/Scripts/python.exe
python tools/promo/video_audio.py check --project <build>/video
cd <build>/video && npx --yes hyperframes@0.8.122 check
cd <build>/video && npx --yes hyperframes@0.8.122 render --quality delivery --fps 30 --output renders/iteration-cfd-f1-promo-v2-raw.mp4
ffmpeg -v error -y -i renders/iteration-cfd-f1-promo-v2-raw.mp4 -i assets/audio/master.wav \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 320k -ar 48000 -shortest -movflags +faststart \
  renders/iteration-cfd-f1-promo-v2.mp4
```

## v1

The 82-second silent v1 (bilingual captions, no music, sha256 prefix
bea90a42, rendered 2026-10-05) is kept at
`C:/Users/sdd32/Videos/iteration-cfd-f1-promo.mp4` with the model's
`CREDITS.txt` beside it; its record and the v1 sources table were the base of
this file.

## Render (v2)

The supervisor rendered the film in a visible console (through the shared
machine lock) with `npx --yes hyperframes@0.8.122 render --quality delivery
--fps 30 --output renders/iteration-cfd-f1-promo-v2-raw.mp4`: 2700 frames,
1 m 30.0 s, rendered in 3 m 39.7 s (screenshot capture, hardware GPU, capture
3 m 29.3 s); the GPU was idle before and after (nvidia-smi 0 %, 907 / 946 MiB).
The raw file's own audio measured -14.1 LUFS / -1.5 dBTP, so the verified master
was remuxed onto it (`-c:v copy`, AAC-LC 320k, 48 kHz):
`renders/iteration-cfd-f1-promo-v2.mp4` - H.264 High yuv420p 1920x1080 at
30 fps, 2700 video packets, 90.000 s, full `-xerror` decode OK, -14.0 LUFS
integrated, -1.7 dBTP, LRA 2.8, 78.7 MB, sha256 prefix 4891b9ea. Before the
render: `hyperframes check` passed (0 lint / layout / motion findings, 38/38
WCAG AA contrast) and an independent faster-whisper pass over the master read
every line. A copy is at `C:/Users/sdd32/Videos/iteration-cfd-f1-promo-v2.mp4`
beside v1 and the model's credits file.

## Audio v3 (light SFX)

The user asked for the text-reveal hits to be lighter and less obtrusive
(PROMO-SFX-LIGHT, 2026-10-05). In `soundtrack.json`'s cues: every synthesised
`sub` thump is gone (0.25, 72.0, 79.9 s), the five `impact`
(soft-impact-on-lockup) cues became `airy` at 0.25 and 72.0 s (-14 / -13 dB),
`ping` at 74.0 and 79.9 s (-17 / -16 dB) and `tick` at 82.0 s (-15 dB), and
the three `pop` cues dropped from -10 to -13 dB (2.4, 76.75, 78.05 s). VO,
music, mix, gates and the picture are unchanged; the soundtrack now holds 33
cues (the `sub` kind stays synthesised in `video_audio.py`, no cue uses it;
the selftest's cue count and the docstring moved 36 -> 33).

Rebuilt with the same chain: `BUILD OK master I -14.00 LUFS TP -2.00 dBTP`
(first attempt, ceiling -2.0), `ASR PASS stem 12/12 master 12/12`,
`CHECK PASS 61/61` with `honesty` clean. The new master was remuxed onto the
same v2 raw render (`-c:v copy`, AAC-LC 320k, 48 kHz) without re-rendering a
frame: `C:/Users/sdd32/Videos/iteration-cfd-f1-promo-v3.mp4` - H.264
1920x1080 30 fps, 2700 video frames, 90.000 s, AAC 48 kHz stereo, -14.0 LUFS
integrated, -1.9 dBTP, LRA 2.6, 78.7 MB,
sha256 `ea57a9f8edaf00bc99c3a6e7eb3817fa9e8a173c0af3f82ab51f8a42dbda67f3`.
v1 and v2 are kept beside it.

## v4 (mark)

The user found the (주)Iterations title type mismatched and asked for a company
mark (PROMO-LOGO-V4, 2026-10-05). They chose mark E "damped residual" with the
capital-I wordmark. `tools/promo/brand/` holds the mark SVG, its README
(geometry, colours, the lockup rule) and the wordmark font: Bricolage
Grotesque VF (SIL OFL 1.1, from the google/fonts repo,
sha256 `413e7357809ddd12fd80a96a8a396de0e401638d4acd3cb3e37532f0472ac682`,
`OFL.txt` beside it).

- **Title card** — the text-only company block became the horizontal lockup:
  the mark at 204.8 x 153.6 px, gap 51.2 px (0.25 x mark width), then
  `Iterations` in Bricolage Grotesque 800 at 145 px (96 px cap height). Under
  it `(주)이터레이션즈 · Iterations Co., Ltd.` at 36 px (cap ~26 px = 0.28 x
  the wordmark cap), letter-spaced, muted, left-aligned to the wordmark (mark
  width + gap = 256 px); the subtitle stays below at 55/30 px (cap 40/22 px).
  All sizes derive from one cap-height scale using the measured cap ratios
  (Bricolage 0.66, Noto Sans KR 0.733 - read from the OS/2 tables). The mark's
  viewBox is cropped to `0 30 240 180`, the drawn band, so the lockup rule
  "mark height = 1.6 x cap height" measures the drawn mark, not the empty
  240x240 canvas. Animation: the lockup rises at 0.3 s, the wave draws on left
  to right 0.3-1.1 s (stroke-dashoffset on getTotalLength), the teal dot lands
  1.05-1.45 s (back.out); the airy cue window 0.25-0.45 s and the subtitle at
  2.4 s are kept, as are the fades.
- **Close card** — the company block became the same lockup, smaller: wordmark
  82 px (54 px cap), mark 115.2 x 86.4 px, gap 28.8 px, company line 20 px,
  left-aligned to the wordmark (144 px). The column is now laid out for each
  state: `#close-inner` sits pushed down 200 px until the company block
  appears, so the headline `AI-driven CFD Solver` is centred while it leads
  the card (film 74-82 s) and the block moves up over 0.7 s as the company
  block lands at 10.0 s (film 82.0 s). Headline, value line, collaboration
  line, credits and disclaimer are verbatim with their timings (76.75, 78.05,
  79.9 and 82.0 s). 200 px keeps the invisible tail inside the canvas before
  it exists visually.
- **Assets** — `video_assets.py` bumped to `promo-video/2`: stage now also
  copies the brand font to `assets/fonts/BricolageGrotesque-VF.ttf` beside
  Noto and `check` verifies it. `STAGE OK 6 videos 5 stills 12 sources
  25.3 s`, `CHECK PASS 27/27`, selftest `SELFTEST PASS 6/6`. The stager keeps
  its pin probe honest: `upgrade --check` offered 0.8.122 -> 0.8.130 and the
  pin was kept on purpose (v2/v3 were rendered on 0.8.122).
- **Check** — `hyperframes check` (0.8.122): 0 lint / layout / motion
  findings, 38/38 WCAG AA. The `snapshot --at` stills for the title lagged
  ~1 s behind (the snapshot run rides the 25 MB master WAV's load; the close
  frames, later in the film, were exact) - the frame-seeked render is the
  ground truth, and a draft render confirmed the subtitle lands at 2.4 s
  exactly (frames 45/75/90 extracted and inspected).
- **Render (v4)** — the same delivery render in a visible console through the
  shared machine lock: 2700 frames in 3 m 42.8 s (screenshot capture, hardware
  GPU). The v3 audio master was remuxed unchanged (`-c:v copy`, AAC-LC 320k,
  48 kHz): `C:/Users/sdd32/Videos/iteration-cfd-f1-promo-v4.mp4` - H.264
  1920x1080 30 fps, 2700 video frames, 90.000 s, full `-xerror` decode OK,
  -14.0 LUFS integrated, -1.9 dBTP, LRA 2.6, 75.7 MB,
  sha256 `acbefae957dcdd61520ce915fb2cbaf4eac0c208d9328ee3fcc5d324e1ab6b45`.
  `python tools/promo/video_audio.py check` -> `CHECK PASS 61/61` with
  `honesty` clean. v1, v2 and v3 are kept beside it.

## v5 (the mark's own opening, no meteor-cfd in the close)

The user asked for two changes (PROMO-V5, 2026-10-05): the closing card drops
"meteor-cfd" and goes straight to the product claim, and the title scene tells
the mark's own story - the screen splits into the mark's grid of vertical
lines, a full-frame damped residual travels through it and converges into the
mark, and only then does "Iterations" appear.

- **Close card** — the `#close-kicker` "meteor-cfd" element is gone (CSS,
  element and its GSAP cue); the card opens straight onto the headline
  "AI-driven CFD Solver" at 2.0 s. The per-state column push grows from v4's
  200 px to 225.6 px (200 + the 51.2 px of column height the kicker
  contributed, minus the 25.6 px the centred column gains without it), so the
  headline holds the exact v4 position while it leads the card. The small
  credit line now reads "meshed and solved with Iterations" (the user's
  wording; it deliberately differs from the `promo/render/CREDITS.txt` source
  line, which still says meteor-cfd).
- **VO V10** — "meteor-cfd. The AI-driven CFD solver." became "The AI-driven
  CFD solver." (tts "The A.I.-driven C.F.D. solver.", asr keys
  ["driven", "solver"]), re-voiced with the same Kokoro am_michael command at
  speed 1.05: clip 2.368 s (was 4.096 s), start kept at 72.5 s (ends 74.868,
  1.8 s clear of V11). The Korean caption is "AI 기반 CFD 솔버." and the
  caption window moved to 72.45-75.17 (was 72.45-76.6).
- **Title scene** — the lockup no longer rises; it is present from frame 0
  with everything but the opening hidden. The opening lives INSIDE the mark's
  own svg (`#title-open`, `overflow: visible`), oversized in the mark's user
  space, so its convergence is registered on the teal dot by construction
  (`svgOrigin "212 120"`) with no screen-space measurement:
  - 0.0-0.9 s the screen splits: the mark's line progression (x = 40, 80,
    112, 138, 158, 174, 186, 196 with opacity 0.12-0.33 in a 240 viewBox)
    tiled every 600 user units from tile -2 to 3 (48 lines), swept in left to
    right by a 0.014 s per-line stagger, opacity still rising to the right;
  - 0.9-2.3 s the damped residual: a full-frame polyline of the mark's law
    y = A*exp(-4.2 t)*cos(2*pi*3.2 t) (A = 340 user units, t stretched so the
    amplitude dies ~12 units before the dot), drawn on over 1.4 s while the
    grid compresses to 0.84 in x around the dot and fades to 0.5;
  - 2.3-2.7 s convergence: the overlay collapses (scale 0.106, opacity 0,
    power2.in) onto the dot while the real mark draws on beneath it (wave
    0.32 s, lines 0.25 s) and the dot lands 2.45-2.7 s (back.out(1.7));
  - 2.7-3.2 s the wordmark (clean reveal, no bounce), then the company line
    at 3.02 s; the subtitle lands at 3.4/3.55 s and holds 2.2 s until the
    5.6 s fade (at least 1.5 s required).
  - **Dash nub** — v4 left a ~1 px dot of the wave colour at the mark wave's
    start point (20, 58 user) before its draw began: a [A, A] dasharray with
    offset A always lands a dash boundary exactly on s = 0 and the round line
    cap renders it as a nub. v5 uses [L, L + 24] with the draw starting at
    L + 12; pixel-checked gone (brightest pixel in the nub window of the
    1.4 s frame is 53,56,56 = background; v4 measured 230,242,240 there).
- **SFX** — the three title cues became four (33 -> 34): airy -15 dB at
  0.05 s (the split), scan -16 dB at 0.9 s (the wave), ping -16 dB at 2.6 s
  (the dot lands), tick -14 dB at 3.4 s (the subtitle). `video_audio.py`'s
  docstring and the selftest counts moved 33/41 -> 34/40 and T9's V10
  substring fixture lost "meteor".
- **Check** — `video_audio.py --selftest` 9/9; `STAGE OK 6 videos 5 stills
  12 sources`; `video_assets.py check` 27/27; `video_audio.py check` 61/61
  with `honesty` clean; `hyperframes check` (0.8.122): 0 lint / layout /
  motion findings, 37/37 WCAG AA (one fewer than v4 - the removed orange
  kicker was a contrast sample). The pin probe offered 0.8.122 -> 0.8.132 and
  the pin was kept on purpose, as in v4 (v2-v5 all render on 0.8.122).
- **Audio** — `BUILD OK master I -14.00 LUFS TP -2.00 dBTP` (first attempt),
  master sha256 `58318fecd2045451dbc1bed4589319c5bcf5d22bc3b327e7127e17ca4260635c`,
  `ASR PASS stem 12/12 master 12/12`.
- **Render (v5)** — the same delivery render in a visible console through the
  shared machine lock (video-v5-render.cmd): 2700 frames in 3 m 26.1 s
  (screenshot capture, hardware GPU). The verified master was remuxed onto
  the raw render (`-c:v copy`, AAC-LC 320k, 48 kHz):
  `C:/Users/sdd32/Videos/iteration-cfd-f1-promo-v5.mp4` - H.264 1920x1080
  30 fps, 2700 video frames, 90.000 s, full `-xerror` decode OK, -14.0 LUFS
  integrated, -1.8 dBTP, LRA 2.5, 79.7 MB,
  sha256 `9ef72c460961dfa50c5176ef868ddb91286930e1d23c8ab2e29e07e69bdcd9ca`.
  v1-v4 are kept beside it.
- **Snapshots** — `snapshots-v5/` in the build project (0.3, 0.8, 1.4, 2.0,
  2.5, 3.0, 4.0, 73, 76, 80 and 84 s plus contact sheets); the two
  verification rounds are `snapshots-v5b/` and `snapshots-v5c/`. The title
  snapshots ran close to their nominal times this run (v4's ~1 s lag did not
  reappear); the close frames were exact as before.

## v6 (the stagnation-flow opening)

The user asked for the opening to be made with real care
(PROMO-OPENING-V6, 2026-10-05/06): "Iterations 오프닝 을 좀더 잘만들어보자
정성스럽게 모든 역량을 들여서, 배경에 지금 세로줄이 나는데 이거 꼭 마크에
있는 대로만 하지말고 우측으로 갈수록 촘촘해지게 하고 우측이 벽이고
왼쪽에서 우측으로 유동이 있는데 벽을 맞고 stagnation flow가 형성되는
벡터장의 화살표와 속도에 따라 벽근처는 파랑색으로 느리고 좌측은 빠르고 그런
느낌으로 빠르게" - instead of the mark's own line pattern, a graded mesh
densifying toward a right-hand wall, a stagnation flow of RK4-integrated
arrow particles hitting it, coloured by speed (fast and warm on the left,
slow and blue near the wall), and the residual line still converging into
the mark. The background is the solver's own answer: `title.html`'s
`#title-field` canvas draws the actual solution of the actual case.

- **Case** — a new `tools/promo/opening_field.py` writes a 56 x 48 channel
  (`ofgpu-generate-mesh.exe channel <dir> 56 48 1 -extent 0 3.2 -1.35 1.35
  0 0.1 -grading x=0.08 -grading y=1`, 2688 cells; dx 0.155511 first,
  0.012441 last, so the vertical lines tighten geometrically toward the
  wall), retypes the preset's patches (outlet -> no-slip `wall`,
  bottomWall/topWall -> open `outletBottom`/`outletTop`, back/front stay
  `empty`), sets `simulationType laminar` and `nu 0.01 m2/s`, and the case
  is solved on the GPU by the pinned
  `rust/target/gpu-pin/ofgpu-lowmach.exe`
  (sha256 `50471caaa54125e0c2eee4fb34eebdfc2da44727d2b818a2b96bb4234c02103c`),
  `-iters 800 -check 50`: final residuals at iter 799 are |U| 4.006e-10 and
  |p| 5.919e-09, `wall_seconds 4.26`; nvidia-smi before/after: RTX 5070 Ti
  1219 MiB 0 % -> 1219 MiB 31 %.
- **Export** — `opening_field.py export` reads the cell velocities from
  `800/U`, maps each cell to its (j, i) by centre (never by order), and
  bilinearly samples a 64 x 54 grid (x = 0.025 + 0.05*i, y = -1.325 + 0.05*j)
  padded with the inlet/no-slip/zero-gradient boundary values into
  `tools/promo/video/opening/stagnation_field.json` (119,295 bytes) with a
  provenance line naming the solver, sha256 50471caa, the case, the
  residuals and the writing tool.
  `opening_field.py --selftest` -> `SELFTEST PASS 6/6`;
  `check` -> `CHECK PASS 8/8`: G1 cells 2688, ratio 0.080000; G3 residuals
  above; G4 mass balance out 2.6813 (rel 0.00692, gate 0.02); G5 symmetry
  max|u-u'| 1.00e-08, max|v+v'| 0; G6 centreline strictly decreasing, last
  0.000161601, max|U| 1.07817; G7 5/5 oracle fixtures within 2e-3; G8 the
  inlined object equals the JSON file.
- **Opening timeline** — one pure `drawField(t)` on a proxy tween
  (deterministic, seek-safe): the graded mesh lines sweep left to right
  0.02-0.66 s (each grows from the vertical centre, brighter toward the
  wall); the hatched wall slab slides in 0.42-0.72 s; 986 arrow seeds enter
  from 0.40 s (each at `0.40 + 0.95*(x_seed+0.45)/3.6`, the near-wall ones
  last, ~1.35 s) and advect along their RK4 pathlines to rest at 2.30 s;
  the damped residual line draws 1.15-2.10 s with a soft glow at its head
  (its width stays the mark's 8.533 px); at 2.30-2.80 s the whole field,
  mesh and wall compress into the mark under a feathered destination-in
  mask (no shrinking rectangle, nothing outside the mapped window); the
  mark forms on top of it (lines 2.62 s, dot 2.74 s back.out(1.7), wave
  2.78 s - the canvas wave already drew it), wordmark 2.92 s, company line
  3.22 s, subtitle 3.60/3.75 s holding to the 5.6 s fade.
- **Speed colour** - sRGB stops at s = 0 `#2f6bff` (blue, the slow
  near-wall flow), 0.30 `#14b8a6` (teal), 0.65 `#ffc45c` (amber) and 1.00
  `#ff5a36` (orange, the fast free stream); arrow length scales with
  speed^0.85.
- **SFX** - the supervisor retimed the scan cue onto the residual draw: the
  four title cues are airy -15 dB at 0.05 s (the mesh sweep), scan -16 dB
  at 1.15 s (the residual line), ping -16 dB at 2.76 s (the dot lands) and
  tick -14 dB at 3.6 s (the subtitle); `sfx.cues` stays 34.
  `video_audio.py --selftest` -> `SELFTEST PASS 9/9`.
- **Review rounds** - (1) run 1's snapshots (`snapshots-v6/`) showed a
  near-black first third (the mesh too faint at t = 0.15 s) and, during the
  compression, a hard shrinking rectangle with arrows leaking outside it;
  run 2 brightened the mesh lines (opacity 0.16 + 0.30, sweep 0.02-0.66 s,
  growth 0.22 s), moved the residual draw to 1.15-2.10 s with a head glow,
  and melted the compression into a feathered two-pass destination-in mask
  (`snapshots-v6b/` - clean in one round). (2) run 3 moved the mask's
  transparent stops onto the window edges (`featherStops` helper) so the
  mask holds exactly zero outside the field window; a pixel probe on the
  2.5 s frame found 0 stray pixels outside the residual line's pass
  (the old mask leaked 1,428 tinted pixels, max channel 80) in
  `snapshots-v6c/`, plus the supervisor's own `snapshots-v6-sup1/` and
  `snapshots-v6-zoom/` review sets. (3) the supervisor's final review
  accepted the opening and retimed the scan cue to 1.15 s (above).
- **Check** - `STAGE OK 6 videos 5 stills 12 sources 52.5 s`;
  `video_assets.py check` 27/27; `hyperframes check` (0.8.122): Runtime 0,
  Layout 0 issues across 9 samples, Motion 0, Contrast 37/37 WCAG AA,
  Check passed (lint: 0 errors, the same 2 pre-existing warnings as runs
  1-2 - the onUpdate DOM-measurement pattern and the inline JSON size; the
  pin probe offered 0.8.130+ and the pin stays on 0.8.122, as v2-v5).
- **Audio** - `BUILD OK master I -14.00 LUFS TP -2.00 dBTP` (first
  attempt), `ASR PASS stem 12/12 master 12/12`, `video_audio.py check`
  `CHECK PASS 61/61` with `honesty` clean.
- **Render (v6)** - the same delivery render in a visible console through
  the shared machine lock (video-v6-render.cmd, lock
  `iterations_promo_v6_render`): 2700 frames, 1 m 30.0 s, rendered in
  9 m 40.9 s (screenshot capture, hardware GPU, capture 9 m 18.5 s; the
  machine was busier than v5's 3 m 26.1 s); nvidia-smi before/after:
  RTX 5070 Ti 1067 MiB 0 % -> 1308 MiB 0 %. The verified master was remuxed
  onto the raw render (`-c:v copy`, AAC-LC 320k, 48 kHz,
  `-shortest -movflags +faststart`):
  `C:/Users/sdd32/Videos/iteration-cfd-f1-promo-v6.mp4` - H.264 1920x1080
  30 fps, 2700 video frames, 90.000 s, full `-xerror` decode OK, -14.0 LUFS
  integrated, -1.8 dBTP, LRA 2.5, 77.5 MB,
  sha256 `0ca7b2413a3d914e0cdfa00c23a6c3d800130d1bf6c1bb41041dd4970c4e76c7`.
  v5 is kept beside it; v1-v4 were already absent from that folder before
  this run.
- **Frames** - `renders/v6-frames/` in the build project holds frames
  9/24/45/66/84/102/126 (0.3/0.8/1.5/2.2/2.8/3.4/4.2 s) extracted from the
  raw render, so the real render can be checked against the review
  snapshots.
