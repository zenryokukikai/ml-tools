"""
face_crop_parallel._sources - VideoSource 実装

DefaultClipSource: クリップ .mp4 群から全フレームを処理 (従来互換)
VADSource: 元動画 + vad.json でセグメント切り出し + fps 変換
"""

import json
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

from face_crop_parallel._core import (
    ClipTask,
    VideoSource,
    COORDS_FILE,
    is_task_complete,
)


class DefaultClipSource(VideoSource):
    """クリップ .mp4 群から全フレームを処理する (従来動作と完全互換)。

    入力ディレクトリ構造:
        clips_dir/{video_name}/{clip_name}.mp4
    """

    def __init__(self, clips_dir: str):
        self.clips_dir = str(clips_dir)

    def discover(self, out_dir: Path) -> Tuple[List[ClipTask], int, int]:
        clips_root = Path(self.clips_dir)
        all_clips  = sorted(clips_root.rglob("*.mp4"))

        pending = []
        skipped = 0
        for clip in all_clips:
            video_name = clip.parent.name
            clip_name  = clip.stem
            if is_task_complete(Path(out_dir), video_name, clip_name):
                skipped += 1
                continue
            pending.append(ClipTask(
                video_path=str(clip),
                clip_id=clip_name,
                video_name=video_name,
            ))
        return pending, skipped, len(all_clips)

    def iter_frames(self, task: ClipTask) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(task.video_path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    def count_frames(self, tasks: List[ClipTask]) -> int:
        total = 0
        for task in tasks:
            cap = cv2.VideoCapture(task.video_path)
            n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            total += max(n, 0)
        return total


# ---------------------------------------------------------------------------
# VADSource
# ---------------------------------------------------------------------------

@dataclass
class VADClipTask(ClipTask):
    """VAD セグメント情報を追加した ClipTask。"""
    segment_start: float = 0.0
    segment_end:   float = 0.0
    native_fps:    float = 0.0
    target_fps:    float = 25.0


_DEFAULT_VIDEO_EXTS = frozenset({
    ".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV",
})


class VADSource(VideoSource):
    """元動画 + vad.json からセグメントを切り出し、fps 変換してフレームを供給する。

    入力ディレクトリ構造:
        videos_dir/{stem}.mp4
        vad_dir/{stem}/vad.json

    vad.json フォーマット:
        {"segments": [{"start": 0.5, "end": 3.2}, ...], "total_duration": 120.0}
    """

    def __init__(self, videos_dir: str, vad_dir: str,
                 target_fps: float = 25.0, video_exts=None):
        self.videos_dir = str(videos_dir)
        self.vad_dir    = str(vad_dir)
        self.target_fps = target_fps
        self.video_exts = frozenset(video_exts) if video_exts else _DEFAULT_VIDEO_EXTS
        self._cached_path: str = ""
        self._cached_cap = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cached_path"] = ""
        state["_cached_cap"] = None
        return state

    def _get_cap(self, video_path: str):
        """同一動画の連続セグメントで VideoCapture を使い回す。"""
        if self._cached_path != video_path:
            if self._cached_cap is not None:
                self._cached_cap.release()
            self._cached_cap = cv2.VideoCapture(video_path)
            self._cached_path = video_path
        return self._cached_cap

    def discover(self, out_dir: Path) -> Tuple[List[VADClipTask], int, int]:
        videos_root = Path(self.videos_dir)
        vad_root    = Path(self.vad_dir)
        out_root    = Path(out_dir)

        all_tasks: List[VADClipTask] = []
        for video_file in sorted(videos_root.iterdir()):
            if not video_file.is_file() or video_file.suffix not in self.video_exts:
                continue
            stem     = video_file.stem
            vad_path = vad_root / stem / "vad.json"
            if not vad_path.exists():
                continue

            vad_data = json.loads(vad_path.read_text())
            segments = vad_data.get("segments", [])

            cap = cv2.VideoCapture(str(video_file))
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            for seg_idx, seg in enumerate(segments):
                all_tasks.append(VADClipTask(
                    video_path=str(video_file),
                    clip_id=f"{seg_idx:05d}",
                    video_name=stem,
                    segment_start=seg["start"],
                    segment_end=seg["end"],
                    native_fps=native_fps,
                    target_fps=self.target_fps,
                ))

        pending: List[VADClipTask] = []
        skipped = 0
        for task in all_tasks:
            if is_task_complete(out_root, task.video_name, task.clip_id):
                skipped += 1
                continue
            pending.append(task)

        return pending, skipped, len(all_tasks)

    def iter_frames(self, task: ClipTask) -> Iterator[np.ndarray]:
        if not isinstance(task, VADClipTask):
            raise TypeError(
                f"VADSource requires VADClipTask, got {type(task).__name__}"
            )

        cap = self._get_cap(task.video_path)
        native_fps  = task.native_fps
        start_frame = int(task.segment_start * native_fps)
        end_frame   = int(task.segment_end * native_fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if native_fps <= task.target_fps:
            for _ in range(end_frame - start_frame):
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        else:
            ratio   = native_fps / task.target_fps
            n_target = max(
                1,
                int((task.segment_end - task.segment_start) * task.target_fps),
            )
            current_pos = start_frame

            for t in range(n_target):
                target_src = start_frame + int(t * ratio)
                if target_src >= end_frame:
                    break
                while current_pos < target_src:
                    cap.grab()
                    current_pos += 1
                ok, frame = cap.read()
                if not ok:
                    break
                current_pos += 1
                yield frame

    def count_frames(self, tasks: List[ClipTask]) -> int:
        total = 0
        for task in tasks:
            if isinstance(task, VADClipTask):
                duration = task.segment_end - task.segment_start
                effective_fps = min(task.native_fps, task.target_fps)
                total += max(1, int(duration * effective_fps))
        return total
