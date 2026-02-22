"""
Dataset Seeder — Downloads and preprocesses sign language datasets from the internet.
Sources:
  1. WLASL (Word-Level American Sign Language) — https://github.com/dxli94/WLASL
  2. MS-ASL  — https://www.microsoft.com/en-us/research/project/ms-asl/
  3. LSA64   — Argentinian Sign Language (64 signs)
  4. Custom YouTube ASL clips via yt-dlp (optional)
  5. Kaggle datasets (ASL Alphabet, WLASL100, ASL Signs) via kaggle API

Usage:
    python seed_dataset.py --source wlasl --signs hello thanks iloveyou
    python seed_dataset.py --source youtube --signs hello thanks iloveyou
    python seed_dataset.py --source kaggle --dataset asl-alphabet
    python seed_dataset.py --source kaggle --dataset wlasl100
    python seed_dataset.py --source all

Kaggle Setup:
    1. pip install kaggle
    2. Get API key from https://www.kaggle.com/settings → API → Create New Token
    3. Place kaggle.json at ~/.kaggle/kaggle.json  (or set KAGGLE_USERNAME + KAGGLE_KEY env vars)
"""

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("../MP_Data")
SEQUENCE_LENGTH = 30
NUM_SEQUENCES = 30  # sequences to extract per sign
SIGNS_DEFAULT = ["hello", "thanks", "iloveyou", "one", "two", "three"]

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ─── WLASL JSON source ────────────────────────────────────────────────────────
WLASL_JSON_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)

# YouTube search queries per sign (for yt-dlp fallback)
YT_QUERIES = {
    "hello":     "ASL sign language hello tutorial",
    "thanks":    "ASL sign language thank you tutorial",
    "iloveyou":  "ASL I love you hand sign tutorial",
    "one":       "ASL number one finger sign",
    "two":       "ASL number two fingers sign",
    "three":     "ASL number three fingers sign",
}


# ─── Kaggle Dataset Configs ───────────────────────────────────────────────────
# Each entry: { "owner/dataset", label_map, video_glob or image_glob, type }
KAGGLE_DATASETS = {
    # ── ASL Alphabet (static images, A-Z + space/del/nothing) ──────────────
    "asl-alphabet": {
        "slug": "grassknoted/asl-alphabet",
        "type": "image_classes",  # folder-per-class structure
        "split": "asl_alphabet_train/asl_alphabet_train",
        # Map folder name → our sign label (only the ones we care about)
        "label_map": {
            "A": "a", "B": "b", "C": "c", "D": "d", "E": "e",
            "F": "f", "G": "g", "H": "h", "I": "i", "J": "j",
            "K": "k", "L": "l", "M": "m", "N": "n", "O": "o",
            "P": "p", "Q": "q", "R": "r", "S": "s", "T": "t",
            "U": "u", "V": "v", "W": "w", "X": "x", "Y": "y",
            "Z": "z",
        },
        "img_exts": {".jpg", ".jpeg", ".png"},
        "max_per_class": 100,  # images to process per letter
    },
    # ── WLASL-100 (video clips, top-100 ASL words) ──────────────────────────
    "wlasl100": {
        "slug": "risangbaskoro/wlasl100-dataset",
        "type": "video_classes",  # folder-per-class structure of mp4 files
        "split": ".",            # root of extracted zip
        "label_map": {},         # auto-detect: folder name = sign
        "vid_exts": {".mp4", ".avi", ".mov"},
        "max_per_class": 20,     # videos to process per sign
    },
    # ── ASL Signs (Kaggle competition dataset, word-level) ──────────────────
    "asl-signs": {
        "slug": "competitions/asl-signs",
        "type": "parquet_landmarks",  # pre-extracted MediaPipe landmarks as parquet
        "split": "train_landmark_files",
        "label_csv": "train.csv",
        "label_map": {},  # auto-detect from CSV
        "max_per_class": 50,
    },
}

# ─── Keypoint extraction ──────────────────────────────────────────────────────
def extract_keypoints(results) -> np.ndarray:
    _zero = np.zeros(63, dtype=np.float32)
    lh = (np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark],
                   dtype=np.float32).ravel()
          if results.left_hand_landmarks else _zero)
    rh = (np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark],
                   dtype=np.float32).ravel()
          if results.right_hand_landmarks else _zero)
    return np.concatenate([lh, rh])


def process_video(video_path: str, sign: str, start_seq: int) -> int:
    """Extract keypoint sequences from a video file. Returns number of sequences saved."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < SEQUENCE_LENGTH:
        print(f"  [SKIP] Too short ({total_frames} frames): {video_path}")
        cap.release()
        return 0

    seqs_saved = 0
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
    ) as holistic:

        # Extract up to NUM_SEQUENCES non-overlapping chunks
        step = max(1, total_frames // NUM_SEQUENCES)
        for seq_idx in range(NUM_SEQUENCES):
            start_frame = seq_idx * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames_kp = []
            for _ in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                frames_kp.append(extract_keypoints(results))

            if len(frames_kp) < SEQUENCE_LENGTH:
                break  # video ended

            # Save sequence
            seq_dir = DATA_DIR / sign / str(start_seq + seqs_saved)
            seq_dir.mkdir(parents=True, exist_ok=True)
            for i, kp in enumerate(frames_kp):
                np.save(str(seq_dir / f"{i}.npy"), kp)
            seqs_saved += 1

    cap.release()
    print(f"  [OK] Saved {seqs_saved} sequences for '{sign}'")
    return seqs_saved


# ─── WLASL Source ─────────────────────────────────────────────────────────────
def seed_from_wlasl(signs: list[str]):
    """Download WLASL JSON and extract video clips for given signs."""
    print("\n[WLASL] Downloading dataset index...")

    tmp_json = Path("_wlasl_tmp.json")
    try:
        urllib.request.urlretrieve(WLASL_JSON_URL, str(tmp_json))
    except Exception as e:
        print(f"[ERROR] Cannot download WLASL JSON: {e}")
        print("  → Check your internet connection or use --source youtube")
        return

    with open(tmp_json) as f:
        wlasl_data = json.load(f)
    tmp_json.unlink(missing_ok=True)

    signs_lower = [s.lower() for s in signs]

    for entry in wlasl_data:
        gloss = entry.get("gloss", "").lower()
        if gloss not in signs_lower:
            continue

        print(f"\n[WLASL] Processing sign: '{gloss}'")
        instances = entry.get("instances", [])
        saved = 0

        for inst in instances:
            url = inst.get("url", "")
            if not url:
                continue

            # Download video to temp file
            tmp_video = Path(f"_tmp_{gloss}.mp4")
            try:
                urllib.request.urlretrieve(url, str(tmp_video))
            except Exception as e:
                print(f"  [SKIP] Download failed: {e}")
                continue

            # Existing sequences count for this sign
            existing = len(list((DATA_DIR / gloss).glob("*/0.npy"))) if (DATA_DIR / gloss).exists() else 0
            n = process_video(str(tmp_video), gloss, existing)
            saved += n
            tmp_video.unlink(missing_ok=True)

            if saved >= NUM_SEQUENCES:
                break

        print(f"[WLASL] '{gloss}': total {saved} sequences seeded")


# ─── YouTube Source (requires yt-dlp) ─────────────────────────────────────────
def seed_from_youtube(signs: list[str]):
    """Download YouTube ASL tutorial clips using yt-dlp."""
    try:
        import yt_dlp
    except ImportError:
        print("[ERROR] yt-dlp not installed. Run: pip install yt-dlp")
        return

    for sign in signs:
        query = YT_QUERIES.get(sign.lower(), f"ASL sign language {sign} tutorial")
        print(f"\n[YouTube] Searching for: '{query}'")

        tmp_dir = Path(f"_yt_tmp_{sign}")
        tmp_dir.mkdir(exist_ok=True)

        ydl_opts = {
            "format": "bestvideo[height<=480][ext=mp4]+bestaudio/best[height<=480]",
            "outtmpl": str(tmp_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "max_downloads": 3,
            "quiet": True,
            "no_warnings": True,
            "default_search": f"ytsearch3:{query}",
            "merge_output_format": "mp4",
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"ytsearch3:{query}"])
        except Exception as e:
            print(f"  [WARN] yt-dlp error: {e}")

        # Process downloaded videos
        for video_file in tmp_dir.glob("*.mp4"):
            existing = len(list((DATA_DIR / sign).glob("*/0.npy"))) if (DATA_DIR / sign).exists() else 0
            process_video(str(video_file), sign.lower(), existing)

        shutil.rmtree(str(tmp_dir), ignore_errors=True)


# ─── Kaggle Source ────────────────────────────────────────────────────────────
def _kaggle_api():
    """Return an authenticated kaggle.KaggleApi instance, or None on failure."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        return api
    except ImportError:
        print("[ERROR] kaggle package not installed. Run: pip install kaggle")
        return None
    except Exception as e:
        print(f"[ERROR] Kaggle auth failed: {e}")
        print("  → Get your API key: https://www.kaggle.com/settings → API → Create New Token")
        print("  → Place kaggle.json at ~/.kaggle/kaggle.json")
        print("  → OR set env vars: KAGGLE_USERNAME and KAGGLE_KEY")
        return None


def _download_kaggle_dataset(api, slug: str, dest: Path) -> bool:
    """Download and unzip a Kaggle dataset to dest/. Returns True on success."""
    dest.mkdir(parents=True, exist_ok=True)
    owner, name = slug.split("/", 1)
    print(f"  [Kaggle] Downloading {slug} → {dest} ...")
    try:
        api.dataset_download_files(f"{owner}/{name}", path=str(dest), unzip=True, quiet=False)
        return True
    except Exception as e:
        # Some datasets use competition slug
        try:
            api.competition_download_files(name, path=str(dest))
            # unzip any zips inside
            for zf in dest.glob("*.zip"):
                with zipfile.ZipFile(zf) as z:
                    z.extractall(dest)
                zf.unlink()
            return True
        except Exception as e2:
            print(f"  [ERROR] Download failed: {e} | {e2}")
            return False


def _process_images_for_sign(img_dir: Path, sign: str, max_imgs: int):
    """Run MediaPipe on static images and save a SEQUENCE_LENGTH-frame sequence per image."""
    imgs = []
    for ext in (".jpg", ".jpeg", ".png"):
        imgs.extend(img_dir.glob(f"*{ext}"))
    if not imgs:
        return 0

    imgs = imgs[:max_imgs]
    existing = len(list((DATA_DIR / sign).glob("*/0.npy"))) if (DATA_DIR / sign).exists() else 0
    saved = 0

    with mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=0.3,
        model_complexity=0,
    ) as holistic:
        for img_path in imgs:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            kp = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark],
                          dtype=np.float32).ravel() if results.right_hand_landmarks else np.zeros(63, dtype=np.float32)
            lkp = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark],
                           dtype=np.float32).ravel() if results.left_hand_landmarks else np.zeros(63, dtype=np.float32)
            full_kp = np.concatenate([lkp, kp])

            # Replicate single frame 30 times → static gesture sequence
            seq_dir = DATA_DIR / sign / str(existing + saved)
            seq_dir.mkdir(parents=True, exist_ok=True)
            for i in range(SEQUENCE_LENGTH):
                np.save(str(seq_dir / f"{i}.npy"), full_kp)
            saved += 1

    print(f"  [OK] '{sign}': {saved} image sequences saved")
    return saved


def _process_parquet_for_sign(parquet_path: Path, sign: str, seq_start: int) -> int:
    """Convert a pre-extracted landmark parquet file (ASL Signs competition) to .npy sequences."""
    try:
        import pandas as pd
    except ImportError:
        print("  [SKIP] pandas not installed. Run: pip install pandas pyarrow")
        return 0

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"  [SKIP] Cannot read parquet {parquet_path}: {e}")
        return 0

    # ASL Signs format: columns x, y, z, type, landmark_index, frame
    # Group by frame, extract hand landmarks
    if "frame" not in df.columns:
        return 0

    hand_df = df[df.get("type", pd.Series()).isin(["left_hand", "right_hand"])] if "type" in df.columns else df
    frames_grouped = hand_df.groupby("frame")
    frame_keys = sorted(frames_grouped.groups.keys())

    if len(frame_keys) < SEQUENCE_LENGTH:
        return 0

    # Sample SEQUENCE_LENGTH evenly-spaced frames
    step = max(1, len(frame_keys) // SEQUENCE_LENGTH)
    selected = [frame_keys[i * step] for i in range(SEQUENCE_LENGTH)]

    keypoints = []
    for fk in selected:
        frame_data = frames_grouped.get_group(fk)
        lh_rows = frame_data[frame_data["type"] == "left_hand"] if "type" in frame_data else pd.DataFrame()
        rh_rows = frame_data[frame_data["type"] == "right_hand"] if "type" in frame_data else pd.DataFrame()

        def rows_to_kp(rows):
            if len(rows) >= 21:
                rows = rows.sort_values("landmark_index") if "landmark_index" in rows else rows
                return rows[["x", "y", "z"]].values[:21].astype(np.float32).ravel()
            return np.zeros(63, dtype=np.float32)

        keypoints.append(np.concatenate([rows_to_kp(lh_rows), rows_to_kp(rh_rows)]))

    if len(keypoints) < SEQUENCE_LENGTH:
        return 0

    seq_dir = DATA_DIR / sign / str(seq_start)
    seq_dir.mkdir(parents=True, exist_ok=True)
    for i, kp in enumerate(keypoints):
        np.save(str(seq_dir / f"{i}.npy"), kp)
    return 1


def seed_from_kaggle(dataset_key: str, signs: list[str]):
    """
    Download a Kaggle dataset and extract keypoint sequences for given signs.

    dataset_key: one of 'asl-alphabet', 'wlasl100', 'asl-signs'
    """
    if dataset_key not in KAGGLE_DATASETS:
        valid = ", ".join(KAGGLE_DATASETS.keys())
        print(f"[ERROR] Unknown dataset '{dataset_key}'. Valid options: {valid}")
        return

    api = _kaggle_api()
    if api is None:
        return

    cfg = KAGGLE_DATASETS[dataset_key]
    tmp_dir = Path(f"_kaggle_tmp_{dataset_key}")

    print(f"\n[Kaggle] Dataset: {cfg['slug']}")
    ok = _download_kaggle_dataset(api, cfg["slug"], tmp_dir)
    if not ok:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        return

    # ── Image-per-class dataset (e.g. ASL Alphabet) ──────────────────────────
    if cfg["type"] == "image_classes":
        base = tmp_dir / cfg["split"] if cfg["split"] != "." else tmp_dir
        label_map = cfg["label_map"]
        signs_lower = [s.lower() for s in signs]

        for folder in sorted(base.iterdir()):
            if not folder.is_dir():
                continue
            mapped = label_map.get(folder.name, folder.name.lower())
            if signs_lower and mapped not in signs_lower:
                continue
            print(f"\n[Kaggle] Processing class '{folder.name}' → '{mapped}'")
            _process_images_for_sign(folder, mapped, cfg["max_per_class"])

    # ── Video-per-class dataset (e.g. WLASL100) ───────────────────────────────
    elif cfg["type"] == "video_classes":
        base = tmp_dir / cfg["split"] if cfg["split"] != "." else tmp_dir
        signs_lower = [s.lower() for s in signs]

        for folder in sorted(base.iterdir()):
            if not folder.is_dir():
                continue
            sign_name = folder.name.lower()
            if signs_lower and sign_name not in signs_lower:
                continue
            print(f"\n[Kaggle] Processing videos for '{sign_name}'")
            vid_exts = cfg.get("vid_exts", {".mp4", ".avi"})
            videos = [v for v in folder.iterdir() if v.suffix.lower() in vid_exts]
            videos = videos[:cfg["max_per_class"]]
            for vf in videos:
                existing = len(list((DATA_DIR / sign_name).glob("*/0.npy"))) \
                    if (DATA_DIR / sign_name).exists() else 0
                process_video(str(vf), sign_name, existing)

    # ── Parquet landmark dataset (e.g. ASL Signs competition) ─────────────────
    elif cfg["type"] == "parquet_landmarks":
        try:
            import pandas as pd
        except ImportError:
            print("[SKIP] pandas not installed. Run: pip install pandas pyarrow")
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
            return

        label_csv = tmp_dir / cfg["label_csv"]
        if not label_csv.exists():
            print(f"[ERROR] Label CSV not found: {label_csv}")
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
            return

        meta = pd.read_csv(label_csv)
        signs_lower = [s.lower() for s in signs]

        for _, row in meta.iterrows():
            sign_name = str(row.get("sign", "")).lower()
            if signs_lower and sign_name not in signs_lower:
                continue
            parquet_file = tmp_dir / str(row.get("path", ""))
            if not parquet_file.exists():
                continue
            existing = len(list((DATA_DIR / sign_name).glob("*/0.npy"))) \
                if (DATA_DIR / sign_name).exists() else 0
            _process_parquet_for_sign(parquet_file, sign_name, existing)

    shutil.rmtree(str(tmp_dir), ignore_errors=True)
    print(f"\n[Kaggle] Done processing '{dataset_key}'")


# ─── Synthetic Augmentation ───────────────────────────────────────────────────
def augment_existing(signs: list[str], factor: int = 3):
    """
    Augment existing MP_Data sequences by adding:
    - Gaussian noise
    - Temporal jitter (random frame drop/repeat)
    - Scale perturbation

    Multiplies dataset size by `factor`.
    """
    print("\n[Augment] Augmenting existing sequences...")

    rng = np.random.default_rng(42)

    for sign in signs:
        sign_dir = DATA_DIR / sign
        if not sign_dir.exists():
            print(f"  [SKIP] No data for '{sign}'")
            continue

        existing_seqs = sorted([d for d in sign_dir.iterdir() if d.is_dir()],
                                key=lambda x: int(x.name))
        if not existing_seqs:
            continue

        start_idx = len(existing_seqs)
        aug_count = 0

        for i, seq_dir in enumerate(existing_seqs):
            frames = []
            for f_idx in range(SEQUENCE_LENGTH):
                npy = seq_dir / f"{f_idx}.npy"
                if npy.exists():
                    frames.append(np.load(str(npy)))
            if len(frames) < SEQUENCE_LENGTH:
                continue

            seq = np.array(frames)  # shape: (30, 126)

            for aug_idx in range(factor):
                augmented = seq.copy()

                # 1. Gaussian noise
                augmented += rng.normal(0, 0.01, augmented.shape).astype(np.float32)

                # 2. Scale perturbation (±5%)
                scale = rng.uniform(0.95, 1.05)
                augmented *= scale

                # 3. Temporal jitter: randomly drop 1 frame and repeat a neighbor
                drop_idx = rng.integers(1, SEQUENCE_LENGTH - 1)
                augmented = np.delete(augmented, drop_idx, axis=0)
                dup_idx = rng.integers(0, len(augmented))
                augmented = np.insert(augmented, dup_idx, augmented[dup_idx], axis=0)

                # Save
                new_seq_dir = DATA_DIR / sign / str(start_idx + aug_count)
                new_seq_dir.mkdir(parents=True, exist_ok=True)
                for f_idx, frame in enumerate(augmented):
                    np.save(str(new_seq_dir / f"{f_idx}.npy"), frame)

                aug_count += 1

        print(f"  [OK] '{sign}': {aug_count} augmented sequences added (total: {start_idx + aug_count})")


# ─── Summary ──────────────────────────────────────────────────────────────────
def print_summary(signs: list[str]):
    print("\n─── Dataset Summary ───────────────────────────────")
    for sign in signs:
        sign_dir = DATA_DIR / sign
        if sign_dir.exists():
            seqs = list(sign_dir.glob("*/0.npy"))
            print(f"  {sign:15s}: {len(seqs):3d} sequences")
        else:
            print(f"  {sign:15s}:   0 sequences (missing)")
    print("────────────────────────────────────────────────────")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Seed sign language dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Augment existing local data (3x):
    python seed_dataset.py --source augment --augment-factor 5

  Download from WLASL:
    python seed_dataset.py --source wlasl --signs hello thanks iloveyou

  Download from YouTube (needs yt-dlp):
    python seed_dataset.py --source youtube --signs hello thanks

  Download from Kaggle — ASL Alphabet (static images, A-Z):
    python seed_dataset.py --source kaggle --dataset asl-alphabet --signs hello

  Download from Kaggle — WLASL100 (videos, 100 words):
    python seed_dataset.py --source kaggle --dataset wlasl100 --signs hello thanks iloveyou

  Download from Kaggle — ASL Signs competition (parquet landmarks):
    python seed_dataset.py --source kaggle --dataset asl-signs

  Download everything:
    python seed_dataset.py --source all --dataset asl-alphabet

Kaggle Setup:
  pip install kaggle
  # Place ~/.kaggle/kaggle.json  OR  set KAGGLE_USERNAME + KAGGLE_KEY env vars
        """
    )
    parser.add_argument(
        "--source", choices=["wlasl", "youtube", "kaggle", "augment", "all"],
        default="augment",
        help="Data source to use (default: augment existing data)"
    )
    parser.add_argument(
        "--signs", nargs="+", default=SIGNS_DEFAULT,
        help="Signs to download/augment (default: hello thanks iloveyou one two three)"
    )
    parser.add_argument(
        "--augment-factor", type=int, default=3,
        help="Augmentation factor (default: 3x)"
    )
    parser.add_argument(
        "--dataset",
        choices=list(KAGGLE_DATASETS.keys()),
        default="asl-alphabet",
        help="Kaggle dataset to download (used with --source kaggle or all)"
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Seeder] Target directory: {DATA_DIR.resolve()}")
    print(f"[Seeder] Signs: {args.signs}")
    print(f"[Seeder] Source: {args.source}\n")

    if args.source in ("wlasl", "all"):
        seed_from_wlasl(args.signs)

    if args.source in ("youtube", "all"):
        seed_from_youtube(args.signs)

    if args.source in ("kaggle", "all"):
        seed_from_kaggle(args.dataset, args.signs)

    if args.source in ("augment", "all"):
        augment_existing(args.signs, factor=args.augment_factor)

    print_summary(args.signs)
    print("\n[Done] Run your training notebook to retrain the model.")


if __name__ == "__main__":
    main()
