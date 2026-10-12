#!/usr/bin/env bash
# Run on the Mac from the repo root:  bash tools/make_proxies.sh
# Makes lightweight 1080p copies (no audio) of the footage and references
# under footage/, so they fit in GitHub (each file kept well under 100 MB).
# Needs ffmpeg:  brew install ffmpeg
set -euo pipefail

SRC="/Users/kirtan/Documents/ID"

convert_dir() {  # <source dir> <dest dir> <height>
  local src="$1" dst="$2" h="$3"
  mkdir -p "$dst"
  # Searches subfolders too (e.g. Camera Shots/CLIP, DCIM/100MEDIA); skips macOS ._ files.
  find "$src" -type f ! -name '._*' \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mxf' -o -iname '*.m4v' -o -iname '*.avi' -o -iname '*.mts' \) -print0 |
  while IFS= read -r -d '' f; do
    rel="${f#"$src"/}"
    name="$(echo "${rel%.*}" | tr ' /' '__')"
    out="$dst/$name.mp4"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "-> $out"
    ffmpeg -nostdin -v error -i "$f" -an \
      -vf "scale=-2:'min($h,ih)',fps=30" \
      -c:v libx264 -preset medium -crf 23 -maxrate 6M -bufsize 12M \
      -pix_fmt yuv420p -movflags +faststart "$out"
  done
}

convert_dir "$SRC/Data/Camera Shots"   footage/camera 1080
convert_dir "$SRC/Data/Drone"          footage/drone  1080
convert_dir "$SRC/Data/FVP shots"      footage/fpv    1080
convert_dir "$SRC/Refrence Videos"     footage/references 720

echo
echo "Files over 95 MB (need trimming or Git LFS before pushing):"
find footage -type f -size +95M -print
du -sh footage/*
