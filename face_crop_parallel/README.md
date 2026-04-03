# face_crop_parallel

動画クリップ群から顔領域を並列検出・クロップして正方形 JPEG として保存する前処理ツール。

## 特徴

- **YuNet** (OpenCV 組み込み, CPU のみ) で高速顔検出 ─ GPU 不要
- **マルチプロセス (spawn)** で並列化 ─ GIL を回避し CPU コア数に比例してスケール
- **ストリーミングデコード** ─ 1 フレームずつ処理して大容量動画のメモリ使用量を抑制
- **再開可能** ─ `coords.npy` + JPEG 枚数が一致するタスクを自動スキップ
- **tqdm** でフレーム単位のスループット / ETA をリアルタイム表示
- **VideoSource プラグイン** ─ 入力形式を抽象化。クリップ群 / VAD セグメント / カスタムソースを切替可能

## 入力モード

### Clip モード (デフォルト)

事前に分割済みのクリップ `.mp4` 群を処理する。

```
clips_dir/{video_name}/{clip_name}.mp4
```

### VAD モード

元動画 + VAD (Voice Activity Detection) の結果 JSON から、発話区間のみを切り出して処理する。
ffmpeg による中間クリップ生成が不要になり、ディスク使用量と処理時間を大幅に削減できる。

```
videos_dir/{stem}.mp4
vad_dir/{stem}/vad.json
```

**vad.json フォーマット:**

```json
{
  "segments": [
    {"start": 0.5, "end": 3.2},
    {"start": 10.1, "end": 15.8}
  ],
  "total_duration": 120.0
}
```

## 出力形式

```
out_dir/
  {video_name}/
    {clip_id}/
      0.jpg, 1.jpg, ...    # 顔検出に成功したフレームのみ (0-indexed)
      coords.npy           # 全フレームの座標キャッシュ
```

### coords.npy

```
shape = (N_frames, 4), dtype = int32
coords[i] = [x1, y1, x2, y2]   # 元フレーム内の顔バウンディングボックス (マージン適用済み)
coords[i] = [-1, -1, -1, -1]   # 顔未検出
```

## インストール

```bash
# GitHub から直接インストール
pip install "git+https://github.com/zenryokukikai/ml-tools.git#subdirectory=face_crop_parallel"

# ローカルからインストール (開発時)
cd face_crop_parallel
pip install -e .
```

YuNet ONNX モデルをダウンロード:

```bash
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## 使い方

### CLI

```bash
# Clip モード
face-crop-parallel \
  --clips_dir /path/to/clips \
  --out_dir   /path/to/output \
  --yunet     /path/to/face_detection_yunet_2023mar.onnx \
  --workers   16

# VAD モード
face-crop-parallel \
  --videos_dir /path/to/videos \
  --vad_dir    /path/to/vad_output \
  --out_dir    /path/to/output \
  --yunet      /path/to/face_detection_yunet_2023mar.onnx \
  --workers    16 \
  --target_fps 25
```

### Python API

```python
from face_crop_parallel import run, DefaultClipSource, VADSource

# Clip モード
source = DefaultClipSource("/path/to/clips")
run(source=source, out_dir="/path/to/output", yunet="model.onnx", workers=16)

# VAD モード
source = VADSource("/path/to/videos", "/path/to/vad_output", target_fps=25.0)
run(source=source, out_dir="/path/to/output", yunet="model.onnx", workers=16)
```

### カスタム VideoSource

`VideoSource` を継承して独自の入力形式に対応できる。

```python
from face_crop_parallel import ClipTask, VideoSource, run

class MyCustomSource(VideoSource):
    def discover(self, out_dir):
        # 処理対象の ClipTask リストを返す
        tasks = [ClipTask(video_path="...", clip_id="...", video_name="...")]
        return tasks, 0, len(tasks)  # (pending, skipped, total)

    def iter_frames(self, task):
        # BGR フレームを yield
        cap = cv2.VideoCapture(task.video_path)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()

    def count_frames(self, tasks):
        return sum(...)  # 推定フレーム数

run(source=MyCustomSource(), out_dir="output", yunet="model.onnx")
```

### オプション

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--clips_dir` | - | Clip モード: 入力動画クリップのルートディレクトリ |
| `--videos_dir` | - | VAD モード: 元動画のディレクトリ |
| `--vad_dir` | - | VAD モード: `{stem}/vad.json` を含むディレクトリ |
| `--target_fps` | 25.0 | VAD モード: 出力フレームレート (fps 変換) |
| `--out_dir` | (必須) | 出力先ディレクトリ |
| `--yunet` | (必須) | YuNet ONNX モデルのパス |
| `--workers` | 8 | 並列ワーカー数 |
| `--det_scale` | 0.5 | 検出時のフレーム縮小率 (小さいほど高速・低精度) |
| `--bbox_margin` | 0.25 | 検出ボックスの拡張率 (各辺, 0.25 = 25% 拡張) |
| `--face_size` | 192 | 出力 JPEG のサイズ (正方形, px) |
| `--jpeg_quality` | 95 | JPEG 品質 (0–100) |

## パフォーマンスの実測値

実測環境: **Intel Xeon (20 物理コア), Docker コンテナ内, Ubuntu 22.04**

| workers | 処理開始直後 | 安定後 (ウォームアップ後) |
|---------|------------|------------------------|
| 16 | ~800 fps | ~1,800–2,000 fps |

- 処理開始直後は OS のページキャッシュが冷えているため低め。数十秒経過後、ディスクキャッシュや JIT 最適化が効いてスループットが大きく向上する。
- `--workers` は物理コア数の 80% 程度が目安。増やしすぎると CPU 競合で逆に遅くなる。
- 上記環境では `--workers 16` が最速。コア数の異なる環境では適宜調整すること。

## 設計上の注意

### VideoSource プラグインアーキテクチャ

`VideoSource` は動画入力を抽象化するインターフェース。以下の制約を満たす必要がある:

- **pickle 可能** であること (spawn コンテキストでワーカーに渡すため)
- `iter_frames()` 内で `cv2.VideoCapture` を開くこと (ファイルハンドルを保持しない)

組み込み実装:
- `DefaultClipSource`: クリップ `.mp4` 群を処理 (v0.1 互換)
- `VADSource`: 元動画 + `vad.json` から発話区間のみを切り出し + fps 変換

### CPU 競合の回避

各ワーカープロセス内で `cv2.setNumThreads(1)` と環境変数
(`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) を `1` に設定する。
設定しないと各ワーカーが全コアを使おうとしてスラッシングが発生する。

### spawn コンテキストと共有カウンタ

`multiprocessing.Value` はデフォルトの fork コンテキストで作成されるため、
spawn プロセスに渡すと `get_lock()` でクラッシュする。
必ず `ctx = mp.get_context("spawn")` の `ctx.Value` で作成すること。

```python
ctx            = mp.get_context("spawn")
frames_counter = ctx.Value('i', 0)   # NG: mp.Value('i', 0)
```

### 整合性チェック

前処理を中断・再開する場合は先に整合性チェックを行うことを推奨:

```python
import shutil, numpy as np
from pathlib import Path

out = Path("/path/to/out_dir")
for clip_dir in [d for vd in out.iterdir() for d in vd.iterdir() if d.is_dir()]:
    coords = clip_dir / "coords.npy"
    jpgs   = list(clip_dir.glob("*.jpg"))
    bad    = not coords.exists()
    if not bad:
        arr = np.load(str(coords))
        bad = len(jpgs) != int((arr[:, 0] >= 0).sum())
    if bad:
        shutil.rmtree(clip_dir)
        print(f"削除: {clip_dir}")
```
