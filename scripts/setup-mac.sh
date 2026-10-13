#!/usr/bin/env bash
# One-time setup on macOS: installs Homebrew (if missing), ffmpeg, and Claude Code.
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"
fi

command -v ffmpeg >/dev/null 2>&1 || brew install ffmpeg
command -v claude >/dev/null 2>&1 || curl -fsSL https://claude.ai/install.sh | bash

echo
ffmpeg -version | head -1
echo "Setup done. Put footage in ./input, then run: claude"
