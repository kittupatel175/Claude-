# Video editing workspace

Drop raw footage in `input/`, ask Claude to edit it, and the results land in `output/`.
Editing is done with [ffmpeg](https://ffmpeg.org/). Video files are git-ignored, so they stay on your Mac.

```
input/    raw clips you paste in
output/   edited videos Claude renders
scripts/  setup helpers
```

## Setup (once, on your Mac)

```bash
git clone https://github.com/kittupatel175/Claude-.git ~/video-edit
cd ~/video-edit
./scripts/setup-mac.sh
```

## Editing

```bash
cd ~/video-edit
claude
```

Then ask in plain words, giving the file path. For example:

- `Trim input/clip.mp4 from 0:12 to 0:45`
- `Join input/a.mp4 and input/b.mp4 into one video`
- `Make input/clip.mp4 vertical 9:16 for Reels`
- `Add background music input/song.mp3 at 30% volume`
- `Burn subtitles input/subs.srt onto input/talk.mp4`
- `Compress input/big.mov to under 50 MB`
- `Speed up input/clip.mp4 2x and remove the audio`

Files outside the folder work too — just give the full path, e.g. `~/Desktop/footage/clip.mov`.
