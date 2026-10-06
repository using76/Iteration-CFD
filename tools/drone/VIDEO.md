<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
     No GPL-licensed source was consulted. -->

# The Iterations defence drone showreel (90 s)

## What the film says

A 90-second film, provided by Meteo Simulation (제공 메테오시뮬레이션), about what the
tool does on a real drone: the PX4 x500 in forward flight, set up, meshed, solved on
one GPU, automatically reviewed and re-meshed, and exported to Isaac Sim; it then shows
Meteo Simulation's BUL:C low-visibility training worlds (the isaacsim_cuFFT smoke run, in
which a trained policy flies its own stand-in drone, not the x500). English narration (17 VO lines, Kokoro
am_michael), Korean captions, one continuous licensed music bed, light SFX only
(no sub, no impact; every cue gain <= -11 dB).

Branding rules (binding, user 2026-10-05): the program name is "Iterations"; the main
credit is 제공 메테오시뮬레이션 / Meteo Simulation with the Meteo logo on a light plate
in the close (and the substituted title legal line); the only (주)Iterations mention on
screen is one small "Collaboration by (주)Iterations" line at the bottom right of the
close; no "Iterations Co., Ltd." and no "이터레이션즈" appears on screen (they stay in
the repository header comments).

## Sources

Everything is staged by `tools/drone/video_assets.py stage` from `SP/drone/render`,
`SP/drone/isaac`, `SP/drone/solve-full`, the isaacsim_cuFFT smoke run, the Meteo brand
assets and the Noto Sans KR font; the renders and everything made from them stay
OUTSIDE the repository (PX4 x500, BSD-3-Clause, no endorsement implied).

Videos (encoded with the promo stager's `encode_job`, 30 fps):

| out                        | source                              | frames | seconds |
|----------------------------|-------------------------------------|-------:|--------:|
| assets/turntable_clean.mp4 | {render}/turntable_clean/frame_%04d |    240 |     8.0 |
| assets/turntable_cp.mp4    | {render}/turntable_cp/frame_%04d    |    240 |     8.0 |
| assets/reveal.mp4          | {render}/reveal/frame_%04d          |    195 |     6.5 |

Copies (59, byte-identical): the six render stills (hero_clean, hero_cp, downwash,
slice, before, after), grid_clean + the 36 multiverse variants, three Isaac Sim frames,
seven isaacsim_cuFFT smoke frames, the two Meteo brand images, the two fonts,
LICENSE.txt / LICENSE, and (staged, not copied) compositions/title.html.

The staged title: `P/compositions/title.html` is the promo v6 opening
(`tools/promo/video/compositions/title.html`, whose 76 kB generated canvas line is
never kept in this tree) with EXACTLY four text substitutions, each required to occur
once in the source, recorded with the source sha256 in assets.json: the background
still `streamlines.png` -> `downwash.png`; the legal line `(주)이터레이션즈 · Iterations
Co., Ltd.` -> `제공 메테오시뮬레이션 · Meteo Simulation`; the Korean subtitle ->
`국방·드론을 위한 AI 기반 CFD`; the English subtitle -> `AI-driven CFD for defence and
drones`. Its timeline, wrap fade (5.6 + 0.4) and canvas field are untouched.

The residual chart: `assets/residuals.svg` is drawn by `residual_svg` from the solve's
own `residuals.csv` (1000 rows, steps 0..999): 560x300 viewBox, three gridlines at
log10 -1/-5/-9, one polyline per series (U_res teal #14B8A6, p_res amber #f5a524,
cont_err grey #a7afb8), rows with step % 5 == 0 plus the last row, x linear in step
across the plot box (56,20)-(540,260), y linear in log10 value clamped into
[-9, -1], points at one decimal, no text in the SVG (the composition writes the
labels).

## Soundtrack

Music: HeyGen audio library `295267cb9a224310b0d170e9194db151` "Astral Generated
Music: 295267cb" (driving synth-heavy tension, cinematic military build), 90 s, bpm
117.3, final hit 80.78 s, silent from 85.85 s; sha256
`dca6366a...586b`; used as ONE continuous excerpt src 0-90 -> film 0-90, no joins;
bed gain -11 dB. Full provenance: `P/assets/music/src/SOURCES.json` (the supervisor's
HeyGen library search of 2026-10-06, credentials read from the user profile, never
printed). SFX: 35 licensed HeyGen cues from `P/assets/sfx/SFX_INDEX.json`, light only
(kinds airy/ping/riser/scan/tick/ticks/typing/whoosh; no "sub", no "impact"; every
gain <= -11 dB).

VO: 17 English lines, Kokoro-82M (`am_michael`), local, speed 1.0, synthesised through
`npx --yes hyperframes@0.8.122 tts` with the Kokoro venv as HYPERFRAMES_PYTHON. The
mix/master chain is the promo's `video_audio.py` run through the drone module's
import wiring (the same bed gain, VO presence carve/duck, SFX bank, BS.1770-4 own
meter and 4x-oversampled true-peak limiter, ffmpeg ebur128 gate loop); only `check`
and the selftest are the drone's own. Measured THIS run:

- `VOICE OK 17 lines` (clip_s below, Narration table)
- `BUILD OK master I -14.00 LUFS TP -2.00 dBTP` (gates [-14.3, -13.7] / -1.9, attempt 1)
- `ASR PASS stem 17/17 master 17/17`. Run 1 heard V17 as "VASTER FLIGHT SCIENCE" with
  0.3 s of lead silence; its `tts` is now `"Faster, flight science."` (the comma breaks
  the run together; `text`, `ko` and `asr_keys` are unchanged in the contract) and
  whisper passes both the stem excerpt and the master.
- `CHECK PASS 82/82` (video_audio), `CHECK PASS 74/74` (video_assets)
- `npx --yes hyperframes@0.8.122 check`: 0 lint errors, 3 warnings (the solve.html
  `data-start="4.8"` nested-media note, and two on the staged promo title.html - a
  gsap DOM-measurement callback and its 334-line length), Layout 0 issues,
  Motion 0 errors, Contrast 58/58 WCAG AA, "Check passed".

## Timeline

| scene | start | dur | on screen |
|---|---:|---:|---|
| title | 0.0 | 6.0 | the promo v6 opening: the solved stagnation field resolving into the Iterations mark, then 제공 메테오시뮬레이션 · Meteo Simulation and the drone subtitles |
| problem | 6.0 | 8.0 | the clean hero still, darkened, slow push-in; "Air you cannot see", ISR / LOGISTICS / SWARM chips; PX4 x500 · forward flight 15 m/s |
| setup | 14.0 | 11.5 | the request card (vehicle, flight, rotors), the automatic setup lines, the clean turntable video, then the production wall mesh and the 1,286,601 cells count-up |
| solve | 25.5 | 10.5 | the streamline reveal video, then the Cp turntable; the GPU solve card: residuals.svg wipe, legend, axis, four tiles (1,000 steps, 13.4 min, RTX 5070 Ti, 1.29 M cells) |
| review | 36.0 | 14.0 | the eight-check before card (REDUCED pair), the amber highlight on arm capture, the REM-WALL-LEVEL remedy, then the after walls and 0.20 -> 0.59 |
| multiverse | 50.0 | 10.0 | 36 visual variants flying outward on golden-angle orbits into the grid plate, the baseline cell outlined "solved: baseline"; concept visualisation, not solved |
| isaac | 60.0 | 10.0 | the Isaac Sim headless renders, the three exported file chips (usda, usdc, vdb), the Blender-render flow note, "autonomy training environments" |
| bulc | 70.0 | 10.5 | seven path-traced smoke frames from the isaacsim_cuFFT fire run, BUL:C brand, SMOKE/HAZE/FOG chips, the visibility stat card |
| close | 80.5 | 9.5 | the Iterations lockup over the blurred hero, the light 제공 plate with the Meteo logo (제공 appears once, on the plate; the line under it reads 메테오시뮬레이션 · Meteo Simulation), licence + honesty credits, "Collaboration by (주)Iterations" |

## Narration and captions

English text and cap_in/cap_out from this run's `P/assets/audio/audio_meta.json`
(caption rule: promo `caption_times` - in at start-0.05/scene+0.10, out at end+0.30,
clamped to scene and neighbours, rounded to 0.01); Korean captions are the soundtrack
`ko` strings with the line break as `<br />`.

| id | start | end | English | Korean | cap_in | cap_out |
|---|---:|---:|---|---|---:|---:|
| V01 | 0.70 | 5.74 | Iterations. AI-driven CFD, by Meteo Simulation. | - | - | - |
| V02 | 6.50 | 9.66 | Every drone mission depends on air you cannot see. | 보이지 않는 공기가 모든 드론 임무를 좌우합니다. | 6.45 | 9.96 |
| V03 | 10.20 | 13.38 | Rotor downwash and the wake decide how it flies. | 로터 하강류와 후류가 비행을 결정합니다. | 10.15 | 13.68 |
| V04 | 14.40 | 19.39 | Describe the drone and the flight. Iterations sets up the case on its own. | 드론과 비행 조건을 알려주면, Iterations가 해석 케이스를 스스로 설정합니다. | 14.35 | 19.60 |
| V05 | 19.70 | 25.05 | 1.3 million cells in seven minutes, with the rotors as actuator disks. | 7분 만에 130만 셀, 로터는 작동원판(actuator disk)으로 모델링합니다. | 19.65 | 25.36 |
| V06 | 25.90 | 31.13 | Then one GPU solves the flow around the airframe and its rotors. | 이어서 GPU 한 장이 기체와 로터 주변의 유동을 풉니다. | 25.85 | 31.40 |
| V07 | 31.50 | 34.40 | A thousand time steps in under fourteen minutes. | 1,000 타임스텝을 14분 안에. | 31.45 | 34.70 |
| V08 | 36.50 | 41.47 | Then it reviews its own result. Eight checks. The arms are under-resolved. | 그리고 스스로 결과를 검토합니다. 8개 항목 점검, 암(arm) 표면 해상도 부족. | 36.45 | 41.70 |
| V09 | 41.80 | 45.92 | It picks a fix, refines the mesh and runs again, automatically. | 처방을 골라 격자를 세분화하고 다시 돌립니다. 자동으로. | 41.75 | 46.10 |
| V10 | 46.20 | 49.57 | Arm surface capture: 20 to 59 percent. | 암 표면 포착률 20 % → 59 %. | 46.15 | 49.87 |
| V11 | 50.60 | 56.42 | Every design variant can go through the same loop. Set up, solved, reviewed, improved. | 모든 설계 변형이 같은 루프를 거칠 수 있습니다. 설정, 해석, 검토, 개선. | 50.55 | 56.72 |
| V12 | 60.40 | 65.78 | Results export as USD and OpenVDB, which Isaac Sim imports. | 결과는 Isaac Sim이 불러오는 USD와 OpenVDB로 내보냅니다. | 60.35 | 65.90 |
| V13 | 66.00 | 69.71 | So drone autonomy training environments are set up fast. | 드론 자율비행 훈련 환경을 빠르게 구성합니다. | 65.95 | 69.90 |
| V14 | 70.50 | 75.53 | And with BUL:C, Meteo Simulation's fire and evacuation simulator, | 그리고 메테오시뮬레이션의 화재·피난 시뮬레이터 BUL:C로, | 70.45 | 75.83 |
| V15 | 76.00 | 79.61 | smoke, haze and fog become physics-based training worlds. | 연기, 연무, 안개가 물리 기반 훈련 환경이 됩니다. | 75.95 | 79.91 |
| V16 | 80.90 | 85.53 | Iterations. AI-driven CFD for defence and drones. | Iterations. 국방과 드론을 위한 AI 기반 CFD. | 80.85 | 85.83 |
| V17 | 86.00 | 87.69 | Faster flight science. (spoken "Faster, flight science.") | 더 빠른 비행 과학. | 85.95 | 87.98 |

The honesty chip (`#captions-chip`, film 25.7 - 49.4 s) reads
`시연용 계산 · 검증된 공력 계수 아님 | Demonstration run - no validated aerodynamic
coefficients`.

## Honesty rules

- No aerodynamic coefficient and no drag/lift number anywhere; the `check` honesty
  scan (the promo pattern plus `lift` and `nvdb`, with word boundaries) runs over the
  raw text of index.html, every staged composition and soundtrack.json.
- The solve numbers on screen are the run's own: 1,286,601 cells, 1,000 time steps,
  13.4 min wall time, RTX 5070 Ti, 1.29 M cells; the residual chart plots the solve's
  own residuals.csv.
- The review pair is the REDUCED test pair (max level 5, 250 steps) and the composition
  says so ("Before card · REDUCED test pair"); the after image is the production mesh.
- The multiverse is "Concept visualisation · variants not solved" (개념 시각화 · 변형
  형상은 해석하지 않음); no variant was solved.
- The isaac scene shows the drone from the Isaac Sim headless render; the flow there is
  the Blender render of the same solve and is labelled "Flow shown: Blender render of
  the same solve".
- Never ".nvdb" - the chip says OpenVDB (which the honesty pattern does not match).
- The smoke frames are labelled "Isaac Sim 6.0 · path-traced smoke from a GPU
  fire-solver run (isaacsim_cuFFT)".
- The demonstration chip fades in at 25.7 s and out at 49.4 s over the solve and
  review scenes, where the run's numbers are on screen.

## Credits

The staged `P/assets/CREDITS.txt` (written by `video_assets.py credits_text`, UTF-8,
LF): the film title line with 제공 메테오시뮬레이션; the PX4 x500 model credit
((c) 2022 Rudis Laboratories / PX4 Autopilot for Drones, BSD-3-Clause, no endorsement
implied, LICENSE.txt/LICENSE beside the file); the music line (HeyGen audio library
295267cb9a224310b0d170e9194db151 with its sha256); the SFX line (HeyGen audio library,
assets/sfx/SFX_INDEX.json); the narration line (Kokoro-82M, Apache-2.0, am_michael,
local); the smoke-frames line (isaacsim_cuFFT, 2026-09-21, gui_0921_114721); the
Isaac-Sim-frames line; the fonts line (Noto Sans KR, Bricolage Grotesque, SIL OFL
1.1); and the honesty line (unsteady unvalidated run, no validated coefficients;
multiverse variants are concept visualisation, not solved).

## How to reproduce

    export PYTHONIOENCODING=utf-8
    SP=<scratchpad>; P=$SP/drone/video
    V=<ASR/TTS venv scratchpad>
    python tools/drone/video_assets.py --selftest
    python tools/drone/video_audio.py --selftest
    python tools/drone/video_assets.py stage --render $SP/drone/render --isaac $SP/drone/isaac \
      --solve $SP/drone/solve-full --smoke <isaacsim_cuFFT smoke run dir> \
      --brand <Meteo brand dir> --font C:/Windows/Fonts/NotoSansKR-VF.ttf --project $P
    python tools/drone/video_assets.py check --project $P
    python tools/drone/video_audio.py voice --project $P --tts-python $V/kokovenv/Scripts/python.exe
    python tools/drone/video_audio.py build --project $P
    python tools/drone/video_audio.py asr --project $P --asr-python $V/asrvenv/Scripts/python.exe
    python tools/drone/video_audio.py check --project $P
    (cd $P && npx --yes hyperframes@0.8.122 check)
    # render the film (the supervisor runs this), then remux the VERIFIED master's
    # audio with -c:v copy - never re-encode the verified audio:

## Render

Rendered 2026-10-06 by the supervisor (DRONE-VIDEO), visible console
`SP/drone/video-render.cmd` through the shared heavy.sh lock: `npx --yes
hyperframes@0.8.122 render --quality delivery --fps 30`, 2700 frames, rendered in
2 m 55.4 s (screenshot capture, hardware GPU; RTX 5070 Ti idle before and after,
779 / 798 MiB, 0 %). The raw render's own audio was replaced by the verified master:
`ffmpeg -i renders/iterations-defence-drone-reel-raw.mp4 -i assets/audio/master.wav
-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 320k -ar 48000 -ac 2 -movflags +faststart`.

- `renders/iterations-defence-drone-reel.mp4` (copy:
  `C:/Users/sdd32/Videos/iterations-defence-drone-reel.mp4`, with
  `iterations-defence-drone-reel-CREDITS.txt` and the PX4 `LICENSE.txt` / `LICENSE`
  beside it as `iterations-defence-drone-reel-PX4-LICENSE.txt` / `-PX4-LICENSE`):
  H.264 1920x1080, 30 fps, 2700 frames = 90.000 s; AAC 48 kHz stereo; 31,964,660 bytes;
  sha256 `f1713c7e78122141b9ad8d2efc1c0c971406de6e043e00efd3c0764d3624fb79`.
  Full `-xerror` decode OK; ebur128 on the file: -14.0 LUFS integrated, -1.8 dBTP,
  LRA 2.1 LU.
- `renders/iterations-defence-drone-reel_contact.jpg`: one tile every 3 s, inspected
  by the supervisor (every scene, the captions band, the close's branding).
