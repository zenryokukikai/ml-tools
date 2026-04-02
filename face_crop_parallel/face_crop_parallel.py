"""
face_crop_parallel.py - YuNet による並列顔クロップ前処理ツール

動画クリップ群から顔領域を検出・クロップして正方形 JPEG として保存する。
検出座標は coords.npy にキャッシュされ、次回実行時はスキップされる。

特徴:
- YuNet (OpenCV 組み込み, CPU のみ) で高速顔検出
- マルチプロセス (spawn) で並列化 ─ GIL を回避
- OpenCV / OpenMP / MKL スレッドを各ワーカー内で 1 に制限し CPU 競合を防止
- ストリーミングデコード ─ 1 フレームずつ処理してメモリ使用量を抑制
- 再開可能 ─ coords.npy + JPEG 枚数が一致するクリップは自動スキップ
- tqdm でフレーム単位のスループット / ETA をリアルタイム表示

入出力ディレクトリ構造:
  clips_dir/
    {video_name}/
      {clip_name}.mp4

  out_dir/
    {video_name}/
      {clip_name}/
        0.jpg, 1.jpg, ...   (顔検出フレームのみ、0-indexed)
        coords.npy          shape=(N_frames, 4), dtype=int32
                            [x1,y1,x2,y2] or [-1,-1,-1,-1] (未検出)

Usage:
  python face_crop_parallel.py \\
    --clips_dir /path/to/clips \\
    --out_dir   /path/to/output \\
    --yunet     /path/to/face_detection_yunet_2023mar.onnx \\
    --workers   16

YuNet モデルのダウンロード:
  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import multiprocessing as mp
import threading
import time

from tqdm import tqdm

# ---------------------------------------------------------------------------
# デフォルト定数 (CLI 引数で上書き可能)
# ---------------------------------------------------------------------------
DEFAULT_DET_SCALE    = 0.5    # 検出用縮小率 (大きな解像度の動画を高速処理するため)
DEFAULT_BBOX_MARGIN  = 0.25   # 検出ボックスを各辺 25% 拡張
DEFAULT_FACE_SIZE    = 192    # 出力 JPEG のサイズ (正方形, px)
DEFAULT_JPEG_QUALITY = 95
DEFAULT_WORKERS      = 8
COORDS_FILE          = "coords.npy"
COUNTER_FLUSH_EVERY  = 50     # N フレームごとに共有カウンタを更新 (ロック競合低減)


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def select_best_face(faces, frame_w: int, frame_h: int):
    """YuNet の検出結果から最も中央に近い大きな顔を選ぶ。

    Args:
        faces: cv2.FaceDetectorYN.detect() の出力 (N×15+) または None
        frame_w: 検出フレームの幅 (det 座標系)
        frame_h: 検出フレームの高さ (det 座標系)

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

    Args:
        box_xywh: [x, y, w, h] (det 座標系)
        det_scale: 検出時の縮小率 (例: 0.5)
        frame_w: 元フレームの幅
        frame_h: 元フレームの高さ
        margin: ボックス拡張率 (例: 0.25 → 各辺を 25% 拡張)

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


# ---------------------------------------------------------------------------
# ワーカープロセス本体
# ---------------------------------------------------------------------------

def _worker(
    worker_id:      int,
    clip_paths:     list,
    out_dir:        str,
    yunet_model:    str,
    det_scale:      float,
    bbox_margin:    float,
    face_size:      int,
    jpeg_quality:   int,
    frames_counter,       # ctx.Value('i') - spawn コンテキストで作成すること
    clips_counter,        # ctx.Value('i') - 同上
) -> None:
    """並列ワーカー。担当クリップを順番に処理する。

    設計上の注意:
    - cv2.setNumThreads(1) + 環境変数でスレッド数を制限し、CPU 競合を防ぐ。
    - ストリーミングデコードで 1 フレームずつ処理し、メモリ使用量を最小化。
    - frames_counter / clips_counter は spawn コンテキストの ctx.Value で
      作成しなければ lock がクラッシュする (fork context の mp.Value は不可)。
    """
    # OpenCV と数値演算ライブラリのスレッドを 1 に制限
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    if not os.path.exists(yunet_model):
        print(f"[worker {worker_id}] ERROR: YuNet model not found: {yunet_model}", flush=True)
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

    out_root     = Path(out_dir)
    local_frames = 0

    for clip_path in clip_paths:
        clip_path   = Path(clip_path)
        video_name  = clip_path.parent.name
        clip_name   = clip_path.stem
        clip_out    = out_root / video_name / clip_name
        coords_path = clip_out / COORDS_FILE

        # --- スキップ判定: coords.npy + JPEG 枚数が一致していれば処理済み ---
        if coords_path.exists():
            try:
                coords_arr  = np.load(str(coords_path))
                valid_count = int((coords_arr[:, 0] >= 0).sum())
                existing_jpg = len(list(clip_out.glob("*.jpg")))
                if existing_jpg == valid_count:
                    with clips_counter.get_lock():
                        clips_counter.value += 1
                    continue
            except Exception:
                pass  # coords.npy が壊れている場合は再処理

        clip_out.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(clip_path))
        ok, first_frame = cap.read()
        if not ok:
            cap.release()
            with clips_counter.get_lock():
                clips_counter.value += 1
            continue

        frame_h, frame_w = first_frame.shape[:2]
        det_w = int(frame_w * det_scale)
        det_h = int(frame_h * det_scale)
        detector.setInputSize((det_w, det_h))

        coords_list = []

        def _process_one(frame, frame_idx: int) -> None:
            small     = cv2.resize(frame, (det_w, det_h))
            _, faces  = detector.detect(small)
            box       = select_best_face(faces, det_w, det_h)
            if box is not None:
                coord = det_box_to_orig(box, det_scale, frame_w, frame_h, bbox_margin)
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

        _process_one(first_frame, 0)
        local_frames += 1

        frame_idx = 1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _process_one(frame, frame_idx)
            local_frames += 1
            frame_idx   += 1

            if local_frames % COUNTER_FLUSH_EVERY == 0:
                with frames_counter.get_lock():
                    frames_counter.value += COUNTER_FLUSH_EVERY

        cap.release()

        remainder = local_frames % COUNTER_FLUSH_EVERY
        if remainder:
            with frames_counter.get_lock():
                frames_counter.value += remainder

        np.save(str(coords_path), np.array(coords_list, dtype=np.int32))

        with clips_counter.get_lock():
            clips_counter.value += 1


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
# メイン
# ---------------------------------------------------------------------------

def _count_frames_fast(clip_paths: list) -> int:
    """動画メタデータからフレーム数を合算する (デコードなし, 高速)。"""
    total = 0
    for p in clip_paths:
        cap = cv2.VideoCapture(str(p))
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        total += max(n, 0)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel face crop preprocessing using YuNet (CPU-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clips_dir", required=True,
        help="入力: 動画クリップのルートディレクトリ ({video}/{clip}.mp4 の構造)",
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="出力: クロップ済み JPEG と coords.npy の保存先",
    )
    parser.add_argument(
        "--yunet", required=True,
        help="YuNet ONNX モデルのパス (face_detection_yunet_2023mar.onnx)",
    )
    parser.add_argument("--workers",      type=int,   default=DEFAULT_WORKERS)
    parser.add_argument("--det_scale",    type=float, default=DEFAULT_DET_SCALE,
                        help="顔検出時のフレーム縮小率")
    parser.add_argument("--bbox_margin",  type=float, default=DEFAULT_BBOX_MARGIN,
                        help="検出ボックスの拡張率 (各辺)")
    parser.add_argument("--face_size",    type=int,   default=DEFAULT_FACE_SIZE,
                        help="出力 JPEG のサイズ (正方形, px)")
    parser.add_argument("--jpeg_quality", type=int,   default=DEFAULT_JPEG_QUALITY)
    args = parser.parse_args()

    clips_root = Path(args.clips_dir)
    all_clips  = sorted(clips_root.rglob("*.mp4"))
    print(f"総クリップ数: {len(all_clips)}, workers: {args.workers}", flush=True)

    if not all_clips:
        print("ERROR: クリップが見つかりません。--clips_dir を確認してください。")
        return

    # スキップ済みクリップを除外して処理対象を特定
    out_root      = Path(args.out_dir)
    pending_clips = []
    skipped       = 0
    for clip in all_clips:
        coords_path = out_root / clip.parent.name / clip.stem / COORDS_FILE
        if coords_path.exists():
            try:
                coords_arr   = np.load(str(coords_path))
                valid_count  = int((coords_arr[:, 0] >= 0).sum())
                existing_jpg = len(list((out_root / clip.parent.name / clip.stem).glob("*.jpg")))
                if existing_jpg == valid_count:
                    skipped += 1
                    continue
            except Exception:
                pass
        pending_clips.append(clip)

    print(f"スキップ: {skipped} / 処理対象: {len(pending_clips)}", flush=True)
    if not pending_clips:
        print("全クリップ処理済みです。")
        return

    print("総フレーム数を集計中...", end=" ", flush=True)
    total_frames = _count_frames_fast(pending_clips)
    print(f"{total_frames:,} frames", flush=True)

    ctx = mp.get_context("spawn")

    # クリップをラウンドロビンでワーカーに分配
    chunks = [[] for _ in range(args.workers)]
    for i, clip in enumerate(pending_clips):
        chunks[i % args.workers].append(str(clip))

    # 共有カウンタは必ず ctx.Value で作成する
    # (mp.Value はデフォルトの fork コンテキストを使うため spawn プロセスでは lock がクラッシュする)
    frames_counter = ctx.Value('i', 0)
    clips_counter  = ctx.Value('i', skipped)

    done_event     = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor,
        args=(frames_counter, clips_counter, total_frames, len(all_clips), done_event),
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
                wid, chunk, args.out_dir, args.yunet,
                args.det_scale, args.bbox_margin, args.face_size, args.jpeg_quality,
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


if __name__ == "__main__":
    main()
