# face_crop_parallel

動画クリップ群から顔領域を並列検出・クロップして正方形 JPEG として保存する前処理ツール。

## 特徴

- **YuNet** (OpenCV 組み込み, CPU のみ) で高速顔検出 ─ GPU 不要
- **マルチプロセス (spawn)** で並列化 ─ GIL を回避し CPU コア数に比例してスケール
- **ストリーミングデコード** ─ 1 フレームずつ処理して大容量動画のメモリ使用量を抑制
- **再開可能** ─ `coords.npy` + JPEG 枚数が一致するクリップを自動スキップ
- **tqdm** でフレーム単位のスループット / ETA をリアルタイム表示

## 出力形式

```
out_dir/
  {video_name}/
    {clip_name}/
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

```bash
# コマンドとして実行 (pip install 後)
face-crop-parallel \
  --clips_dir /path/to/clips \
  --out_dir   /path/to/output \
  --yunet     /path/to/face_detection_yunet_2023mar.onnx \
  --workers   16

# モジュールとして実行
python -m face_crop_parallel \
  --clips_dir /path/to/clips \
  --out_dir   /path/to/output \
  --yunet     /path/to/face_detection_yunet_2023mar.onnx \
  --workers   16
```

### 入力ディレクトリ構造

```
clips_dir/
  {video_name}/
    {clip_name}.mp4
    ...
```

### オプション

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--clips_dir` | (必須) | 入力動画クリップのルートディレクトリ |
| `--out_dir` | (必須) | 出力先ディレクトリ |
| `--yunet` | (必須) | YuNet ONNX モデルのパス |
| `--workers` | 8 | 並列ワーカー数 |
| `--det_scale` | 0.5 | 検出時のフレーム縮小率 (小さいほど高速・低精度) |
| `--bbox_margin` | 0.25 | 検出ボックスの拡張率 (各辺, 0.25 = 25% 拡張) |
| `--face_size` | 192 | 出力 JPEG のサイズ (正方形, px) |
| `--jpeg_quality` | 95 | JPEG 品質 (0–100) |

## パフォーマンスの目安

| CPU | workers | スループット |
|-----|---------|------------|
| 20 コア (Xeon など) | 16 | ~750 fps |

`--workers` は物理コア数の 80% 程度が目安。増やしすぎると CPU 競合で逆に遅くなる。

## 設計上の注意

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
# coords.npy なし、または JPEG 枚数不一致のクリップディレクトリを削除して再処理
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
