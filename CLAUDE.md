# Ishani Dave na Garba — Navratri concert promo

60-second, 16:9 (1920x1080) hype edit for the artist **Ishani Dave** ("ISHANI DAVE NA GARBA").
Concert-energy look: beat-synced cuts, motion graphics, effects, transitions, kinetic typography.

## Inputs
| What | Where |
|---|---|
| Music (only audio to use) | `assets/audio/track.m4a` |
| Artist logo (black on transparent PNG, recolour as needed) | `assets/logo/ishani_dave_na_garba.png` |
| Camera shots (Sony `CLIP` folder) | `footage/camera/` |
| Drone shots | `footage/drone/` |
| FPV shots | `footage/fpv/` |
| 4 reference videos | `footage/references/` — **`01 Rerfrence.mp4` is the main one** |

`footage/` is git-ignored: never commit raw footage or renders.

## Rules from the client
- Never use audio from the reference videos or clips; the only soundtrack is `track.m4a`.
- Take transitions/effects/typography ideas from **all four** references (01 is primary), then go further creatively.
- Only the artist logo — no sponsor/event logos.
- End on the song's natural ending.

## Music map (track.m4a, 58.77 s, ~143 BPM, beat ≈ 0.42 s)
- 0–3.3 s: soft intro → drone establishing shots, logo tease
- 3.37 s: first big hit — cutting starts
- 3.4–51 s: full energy; strongest accents ≈ 12.4, 18, 24.3, 35, 41, 47 s (biggest transitions here)
- 47.3–50.2 s: rapid hits → FPV flurry / finale
- 51–54 s fade, 54–58.8 s silence → full logo reveal end card
- Beat grid: `python3 -c` with librosa `beat_track` on a WAV of the track (first beat 3.37 s)

## Creative plan
- Drone builds the intro; camera close-ups of artist & crowd in verses; FPV dives on drops.
- Effects: flash/strobe cuts, zoom punches, whip & glitch transitions, RGB split, speed ramps, beat shake,
  light leaks in festive red/orange/gold/magenta, garba-circle / dandiya-ring wipes,
  kaleidoscope (rangoli) crowd moments, logo particle/mandala reveal at the first drop and the end.
- Warm, saturated concert grade.

## Workflow
1. Analyse the 4 references (frame grabs, shot-length stats, effect notes) → `notes/references.md`.
2. Catalogue footage (contact sheets, best moments per clip) → `notes/shotlist.md`.
3. Build the edit as a reproducible script (ffmpeg/Python) under `edit/`, render to `renders/`.
4. Final: `renders/ishani_dave_navratri_16x9.mp4`, H.264 1080p, AAC audio from track.m4a.
