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
company ((주)Iterations / Iterations Co., Ltd., subtitle "F1 레이싱카
공력해석 테스트") and closes on "AI-driven CFD Solver".

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
summing to 90.0 s, 12 VO lines, 36 SFX cues, the music descriptor, the mix
specification and the master gates.

- **Music** — one continuous licensed excerpt, no joins: HeyGen audio library
  id `a0dc53f249ae416f8a7dcc15912581a1` (`assets/music/src/a0dc53f2.mp3`,
  sha256 `d00372d5...46a5ca6`), 120 BPM, steady grid, src 0-90 s -> film
  0-90 s at `bed_gain_db` -11.0, its native final hit at film 79.85 s. Chosen
  by the supervisor's library scan (see `assets/music/src/SOURCES.json` in
  the build project).
- **SFX** — 36 cues from the licensed HeyGen library set copied (not edited)
  into `assets/sfx/src` (index: `assets/sfx/SFX_INDEX.json` in the build
  project); kind `sub` is synthesised in `video_audio.py` (an exponential
  78 -> 40 Hz sine sweep over 0.9 s, envelope exp(-t/0.35), plus a 60 Hz,
  0.12 s raised-cosine thump, peak 0.9 before gain).
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
| title | 0 | 6.0 | company opening: (주)Iterations / Iterations Co., Ltd., hairline, "F1 레이싱카 공력해석 테스트" / "F1 race-car aerodynamics test" at 2.4-2.55 s, fade out 5.6-6.0 s |
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
