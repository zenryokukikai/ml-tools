"""
face_crop_parallel._core - YuNet による並列顔クロップの中核モジュール

VideoSource プラグインで入力形式を切り替え可能:
- DefaultClipSource: クリップ .mp4 群から全フレームを処理 (従来互換)
- VADSource: 元動画 + vad.json からセグメントを切り出して fps 変換

特徴:
- YuNet (OpenCV 組み込み, CPU のみ) で高速顔検出
- マルチプロセス (spawn) で並列化 ─ GIL を回避
- OpenCV / OpenMP / MKL スレッドを各ワーカー内で 1 に制限し CPU 競合を防止
- ストリーミングデコード ─ 1 フレームずつ処理してメモリ使用量を抑制
- 再開可能 ─ coords.npy + JPEG 枚数が一致するタスクは自動スキップ
- tqdm でフレーム単位のスループット / ETA をリアルタイム表示

Usage:
  # Clip mode (default)
  face-crop-parallel --clips_dir clips/ --out_dir out/ --yunet model.onnx

  # VAD mode
  face-crop-parallel --videos_dir videos/ --vad_dir vad/ --out_dir out/ --yunet model.onnx

YuNet モデルのダウンロード:
  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
"""

import argparse
import os
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple
import multiprocessing as mp
import threading
import time

from tqdm import tqdm

# ---------------------------------------------------------------------------
# デフォルト定数 (CLI 引数で上書き可能)
# ---------------------------------------------------------------------------
DEFAULT_DET_SCALE    = 0.5
DEFAULT_BBOX_MARGIN  = 0.25
DEFAULT_FACE_SIZE    = 192
DEFAULT_JPEG_QUALITY = 95
DEFAULT_WORKERS      = 8
DEFAULT_DET_MAX_LONG   = 640
DEFAULT_DET_CENTER_CROP = 1.0
COORDS_FILE          = "coords.npy"
COUNTER_FLUSH_EVERY  = 50


# ---------------------------------------------------------------------------
# ClipTask / VideoSource プロトコル
# ---------------------------------------------------------------------------

@dataclass
class ClipTask:
    """1 クリップ (= 出力サブディレクトリ) に対応する処理単位。

    Attributes:
        video_path: 入力動画のパス
        clip_id:    出力サブディレクトリ名 (例: "00042")
        video_name: 出力親ディレクトリ名 (例: "video_001")
    """
    video_path: str
    clip_id: str
    video_name: str


class VideoSource:
    """動画入力の抽象化 — _worker にフレームを供給するプラグインインターフェース。

    実装上の制約:
    - pickle 可能であること (spawn コンテキストでワーカーに渡すため)
    - iter_frames() 内で cv2.VideoCapture を開くこと (ファイルハンドルを保持しない)
    """

    def discover(self, out_dir: Path) -> Tuple[List[ClipTask], int, int]:
        """入力を探索し、処理対象のタスクリストを返す。

        Returns:
            (pending_tasks, skipped_count, total_count)
        """
        raise NotImplementedError

    def iter_frames(self, task: ClipTask) -> Iterator[np.ndarray]:
        """1 タスク分の BGR フレーム (np.ndarray) を yield する。"""
        raise NotImplementedError

    def count_frames(self, tasks: List[ClipTask]) -> int:
        """総フレーム数を推定する (プログレスバー用)。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# スキップ判定ユーティリティ
# ---------------------------------------------------------------------------

def is_task_complete(out_dir: Path, video_name: str, clip_id: str) -> bool:
    """指定タスクが処理済みか判定 (coords.npy 存在 + JPEG 枚数一致)。"""
    clip_dir    = out_dir / video_name / clip_id
    coords_path = clip_dir / COORDS_FILE
    if not coords_path.exists():
        return False
    try:
        coords_arr   = np.load(str(coords_path))
        valid_count  = int((coords_arr[:, 0] >= 0).sum())
        existing_jpg = len(list(clip_dir.glob("*.jpg")))
        return existing_jpg == valid_count
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 顔検出ユーティリティ
# ---------------------------------------------------------------------------

def select_best_face(faces, frame_w: int, frame_h: int):
    """YuNet の検出結果から最も中央に近い大きな顔を選ぶ。

    Returns:
        [x, y, w, h] (det 座標系) または None
    """
    if faces is None or len(faces) == 0:
        return None
    cx_f = frame_w / 2.0
    cy_f = frame_h / 2.0
    best_score = -1.0
    best_box   = None
    for face in faces:
        x, y, fw, fh = face[:4]
        area  = fw * fh
        cx    = x + fw / 2.0
        cy    = y + fh / 2.0
        dist  = ((cx - cx_f) ** 2 + (cy - cy_f) ** 2) ** 0.5
        score = area / (1.0 + dist * 0.01)
        if score > best_score:
            best_score = score
            best_box   = [x, y, fw, fh]
    return best_box


def det_box_to_orig(
    box_xywh,
    det_scale: float,
    frame_w: int,
    frame_h: int,
    margin: float,
) -> list:
    """YuNet 検出ボックス (det 座標系) → 元フレーム座標系 + マージン付加。

    Returns:
        [x1, y1, x2, y2] (int, 元フレーム座標系, クリップ済み)
    """
    x, y, fw, fh = box_xywh
    x_orig = x  / det_scale
    y_orig = y  / det_scale
    w_orig = fw / det_scale
    h_orig = fh / det_scale
    mx = w_orig * margin
    my = h_orig * margin
    x1 = max(0,       int(x_orig - mx))
    y1 = max(0,       int(y_orig - my))
    x2 = min(frame_w, int(x_orig + w_orig + mx))
    y2 = min(frame_h, int(y_orig + h_orig + my))
    return [x1, y1, x2, y2]


def _det_box_to_orig_xy(
    box_xywh,
    scale_x: float,
    scale_y: float,
    frame_w: int,
    frame_h: int,
    margin: float,
) -> list:
    """det_box_to_orig の x/y 個別スケール版 (32 倍丸め対応)。"""
    x, y, fw, fh = box_xywh
    x_orig = x  / scale_x
    y_orig = y  / scale_y
    w_orig = fw / scale_x
    h_orig = fh / scale_y
    mx = w_orig * margin
    my = h_orig * margin
    x1 = max(0,       int(x_orig - mx))
    y1 = max(0,       int(y_orig - my))
    x2 = min(frame_w, int(x_orig + w_orig + mx))
    y2 = min(frame_h, int(y_orig + h_orig + my))
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# ワーカープロセス本体
# ---------------------------------------------------------------------------

_PROFILE_INTERVAL = 30.0  # 秒ごとにプロファイルログを出力


def _worker(
    worker_id:      int,
    tasks:          list,
    out_dir:        str,
    yunet_model:    str,
    source:         "VideoSource",
    det_scale:      float,
    bbox_margin:    float,
    face_size:      int,
    jpeg_quality:   int,
    det_max_long:   int,
    det_center_crop: float,
    frames_counter,
    clips_counter,
) -> None:
    """並列ワーカー。担当タスクを順番に処理する。

    source.iter_frames(task) でフレームを受け取り、
    YuNet で顔検出 → クロップ → JPEG 保存を行う。
    """
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    if not os.path.exists(yunet_model):
        print(f"[worker {worker_id}] ERROR: YuNet model not found: {yunet_model}",
              flush=True)
        return

    detector = cv2.FaceDetectorYN.create(
        yunet_model, "",
        (320, 320),
        score_threshold=0.5,
        nms_threshold=0.3,
        top_k=100,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU,
    )

    out_root       = Path(out_dir)
    local_frames   = 0
    flushed_frames = 0

    t_decode_total    = 0.0
    t_resize_total    = 0.0
    t_detect_total    = 0.0
    t_crop_save_total = 0.0
    prof_frames     = 0
    prof_clips      = 0
    prof_last       = time.time()
    prof_start      = prof_last

    for task in tasks:
        clip_out    = out_root / task.video_name / task.clip_id
        coords_path = clip_out / COORDS_FILE

        clip_out.mkdir(parents=True, exist_ok=True)
        coords_list = []
        frame_idx   = 0
        frame_w = frame_h = det_w = det_h = 0

        scale_x = scale_y = det_scale
        roi_x0 = 0

        t_before_next = time.time()
        frame_iter = source.iter_frames(task)

        for frame in frame_iter:
            t_after_decode = time.time()
            t_decode_total += t_after_decode - t_before_next

            if frame_idx == 0:
                frame_h, frame_w = frame.shape[:2]
                roi_x0 = 0
                roi_w = frame_w
                if det_center_crop < 1.0 and frame_w > frame_h:
                    roi_w = int(frame_w * det_center_crop)
                    roi_x0 = (frame_w - roi_w) // 2
                det_w = max(32, int(roi_w * det_scale) // 32 * 32)
                det_h = max(32, int(frame_h * det_scale) // 32 * 32)
                if det_max_long > 0 and max(det_w, det_h) > det_max_long:
                    shrink = det_max_long / max(det_w, det_h)
                    det_w = max(32, int(det_w * shrink) // 32 * 32)
                    det_h = max(32, int(det_h * shrink) // 32 * 32)
                scale_x = det_w / roi_w
                scale_y = det_h / frame_h
                detector.setInputSize((det_w, det_h))

            t2 = time.time()
            roi = frame[:, roi_x0:roi_x0 + roi_w] if roi_x0 > 0 else frame
            small = cv2.resize(roi, (det_w, det_h))
            t3 = time.time()
            _, faces = detector.detect(small)
            t4 = time.time()

            box = select_best_face(faces, det_w, det_h)

            if box is not None:
                coord = _det_box_to_orig_xy(
                    box, scale_x, scale_y, roi_w, frame_h, bbox_margin,
                )
                if roi_x0 > 0:
                    coord[0] = min(coord[0] + roi_x0, frame_w)
                    coord[2] = min(coord[2] + roi_x0, frame_w)
                coords_list.append(coord)
                x1, y1, x2, y2 = coord
                face_raw = frame[y1:y2, x1:x2]
                if face_raw.size > 0:
                    face_sq = cv2.resize(
                        face_raw, (face_size, face_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imwrite(
                        str(clip_out / f"{frame_idx}.jpg"),
                        face_sq,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                    )
            else:
                coords_list.append([-1, -1, -1, -1])

            t5 = time.time()

            t_resize_total    += t3 - t2
            t_detect_total    += t4 - t3
            t_crop_save_total += t5 - t4

            t_before_next = time.time()

            frame_idx    += 1
            local_frames += 1
            prof_frames  += 1

            if (local_frames - flushed_frames) >= COUNTER_FLUSH_EVERY:
                with frames_counter.get_lock():
                    frames_counter.value += COUNTER_FLUSH_EVERY
                flushed_frames += COUNTER_FLUSH_EVERY

        unflushed = local_frames - flushed_frames
        if unflushed > 0:
            with frames_counter.get_lock():
                frames_counter.value += unflushed
            flushed_frames = local_frames

        if coords_list:
            np.save(str(coords_path), np.array(coords_list, dtype=np.int32))
        else:
            np.save(str(coords_path), np.zeros((0, 4), dtype=np.int32))

        with clips_counter.get_lock():
            clips_counter.value += 1

        prof_clips += 1
        now = time.time()
        if now - prof_last >= _PROFILE_INTERVAL:
            wall = now - prof_start
            fps  = prof_frames / wall if wall > 0 else 0
            print(
                f"[worker {worker_id}] PROFILE "
                f"wall={wall:.1f}s frames={prof_frames} clips={prof_clips} "
                f"fps={fps:.1f} | "
                f"decode={t_decode_total:.2f}s "
                f"resize={t_resize_total:.2f}s "
                f"detect={t_detect_total:.2f}s "
                f"crop_save={t_crop_save_total:.2f}s | "
                f"det_size={det_w}x{det_h} "
                f"roi_x0={roi_x0} "
                f"src={frame_w}x{frame_h} "
                f"video={task.video_name}",
                flush=True,
            )
            prof_last = now


# ---------------------------------------------------------------------------
# 監視スレッド
# ---------------------------------------------------------------------------

def _monitor(
    frames_counter,
    clips_counter,
    total_frames: int,
    total_clips:  int,
    done_event,
) -> None:
    """メインプロセスで動く監視スレッド。tqdm でリアルタイム進捗を表示する。"""
    bar = tqdm(
        total=total_frames,
        desc="face_crop",
        unit="frames",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    prev_frames = 0

    while not done_event.is_set():
        time.sleep(1.0)
        cur_frames = frames_counter.value
        cur_clips  = clips_counter.value
        delta      = cur_frames - prev_frames
        bar.update(delta)
        bar.set_postfix(clips=f"{cur_clips}/{total_clips}", fps=f"{delta:.0f}")
        prev_frames = cur_frames

    bar.update(frames_counter.value - prev_frames)
    bar.close()


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------

def run(
    source:       "VideoSource",
    out_dir:      str,
    yunet:        str,
    workers:      int   = DEFAULT_WORKERS,
    det_scale:    float = DEFAULT_DET_SCALE,
    bbox_margin:  float = DEFAULT_BBOX_MARGIN,
    face_size:    int   = DEFAULT_FACE_SIZE,
    jpeg_quality: int   = DEFAULT_JPEG_QUALITY,
    det_max_long:   int   = DEFAULT_DET_MAX_LONG,
    det_center_crop: float = DEFAULT_DET_CENTER_CROP,
) -> None:
    """VideoSource からフレームを受け取り、顔を検出・クロップして JPEG 保存する。

    Args:
        source: フレーム供給元 (DefaultClipSource / VADSource / カスタム)
        out_dir: 出力ディレクトリ
        yunet: YuNet ONNX モデルのパス
        workers: ワーカープロセス数
    """
    out_root = Path(out_dir)

    pending, skipped, total = source.discover(out_root)
    print(f"総タスク数: {total}, workers: {workers}", flush=True)
    print(f"スキップ: {skipped} / 処理対象: {len(pending)}", flush=True)

    if not pending:
        print("全タスク処理済みです。")
        return

    print("総フレーム数を集計中...", end=" ", flush=True)
    total_frames = source.count_frames(pending)
    print(f"{total_frames:,} frames", flush=True)

    ctx = mp.get_context("spawn")

    video_groups: dict = {}
    for task in pending:
        video_groups.setdefault(task.video_name, []).append(task)

    chunks = [[] for _ in range(workers)]
    for i, (_, group) in enumerate(sorted(video_groups.items())):
        chunks[i % workers].extend(group)

    frames_counter = ctx.Value('i', 0)
    clips_counter  = ctx.Value('i', skipped)

    done_event     = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor,
        args=(frames_counter, clips_counter, total_frames, total, done_event),
        daemon=True,
    )
    monitor_thread.start()

    start = time.time()
    procs = []
    for wid, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = ctx.Process(
            target=_worker,
            args=(
                wid, chunk, out_dir, yunet,
                source,
                det_scale, bbox_margin, face_size, jpeg_quality,
                det_max_long, det_center_crop,
                frames_counter, clips_counter,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    done_event.set()
    monitor_thread.join()

    elapsed   = time.time() - start
    total_jpg = sum(1 for _ in out_root.rglob("*.jpg"))
    total_npy = sum(1 for _ in out_root.rglob(COORDS_FILE))
    print(
        f"\n完了: {elapsed:.1f}s ({elapsed / 3600:.2f}h) | "
        f"JPG: {total_jpg:,} | coords.npy: {total_npy}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel face crop using YuNet (CPU-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Clip mode
    parser.add_argument(
        "--clips_dir", default=None,
        help="Clip mode: root dir of video clips ({video}/{clip}.mp4)",
    )
    # VAD mode
    parser.add_argument(
        "--videos_dir", default=None,
        help="VAD mode: directory containing source videos",
    )
    parser.add_argument(
        "--vad_dir", default=None,
        help="VAD mode: directory containing {stem}/vad.json",
    )
    parser.add_argument(
        "--target_fps", type=float, default=25.0,
        help="VAD mode: target fps for frame subsampling",
    )
    # Common
    parser.add_argument("--out_dir",      required=True,
                        help="Output directory for cropped JPEG + coords.npy")
    parser.add_argument("--yunet",        required=True,
                        help="Path to YuNet ONNX model")
    parser.add_argument("--workers",      type=int,   default=DEFAULT_WORKERS)
    parser.add_argument("--det_scale",    type=float, default=DEFAULT_DET_SCALE,
                        help="Detection frame downscale ratio")
    parser.add_argument("--bbox_margin",  type=float, default=DEFAULT_BBOX_MARGIN,
                        help="Bounding box margin ratio per side")
    parser.add_argument("--face_size",    type=int,   default=DEFAULT_FACE_SIZE,
                        help="Output JPEG size (square, px)")
    parser.add_argument("--jpeg_quality", type=int,   default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--det_max_long", type=int,   default=DEFAULT_DET_MAX_LONG,
                        help="Cap detection longest side (0=unlimited)")
    parser.add_argument("--det_center_crop", type=float, default=DEFAULT_DET_CENTER_CROP,
                        help="Center crop ratio for landscape detection (1.0=no crop)")
    args = parser.parse_args()

    try:
        from face_crop_parallel._sources import DefaultClipSource, VADSource
    except ImportError:
        from _sources import DefaultClipSource, VADSource

    if args.videos_dir and args.vad_dir:
        source = VADSource(args.videos_dir, args.vad_dir, args.target_fps)
    elif args.clips_dir:
        source = DefaultClipSource(args.clips_dir)
    else:
        parser.error("--clips_dir or (--videos_dir + --vad_dir) is required")

    run(
        source=source,
        out_dir=args.out_dir,
        yunet=args.yunet,
        workers=args.workers,
        det_scale=args.det_scale,
        bbox_margin=args.bbox_margin,
        face_size=args.face_size,
        jpeg_quality=args.jpeg_quality,
        det_max_long=args.det_max_long,
        det_center_crop=args.det_center_crop,
    )


if __name__ == "__main__":
    main()
