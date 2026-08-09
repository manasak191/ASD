from pathlib import Path
import numpy as np
import librosa
from pydub import AudioSegment
import uuid

# === MUST match training ===
SR      = 16000
HOP     = 512     # if you retrained with 160 (10ms), change here too
N_MFCC  = 20

def resample_compat(y, sr_in, sr_out):
    if sr_in == sr_out:
        return y
    try:
        return librosa.resample(y, orig_sr=sr_in, target_sr=sr_out)  # librosa >= 0.10
    except TypeError:
        return librosa.resample(y, sr_in, sr_out)                    # librosa < 0.10

def _safe_delta(M, base_width=9):
    T = M.shape[1]
    w = min(base_width, T if T % 2 else T-1)
    w = max(3, w)
    return librosa.feature.delta(M, width=w, mode="interp")

def mfcc_stats_from_array(y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP):
    if y is None or len(y) == 0:
        y = np.zeros(int(0.5*sr), dtype=np.float32)
    # peak normalize
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    # MFCC + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    d1   = _safe_delta(mfcc, base_width=9)
    d2   = _safe_delta(mfcc, base_width=9)
    F    = np.concatenate([mfcc, d1, d2], axis=0)  # (60, T)
    mu, sd = F.mean(axis=1), F.std(axis=1)
    return np.concatenate([mu, sd])                # 120-D

def file_to_vector(wav_path: str|Path):
    y, sr0 = librosa.load(str(wav_path), sr=None, mono=True)
    y = resample_compat(y, sr0, SR)
    return mfcc_stats_from_array(y, sr=SR)

def convert_to_wav(in_path: Path, out_dir: Path) -> Path:
    """
    Convert mp3/ogg/webm/flac/m4a -> WAV. Requires ffmpeg on PATH.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(in_path).stem}_{uuid.uuid4().hex}.wav"
    audio = AudioSegment.from_file(str(in_path))
    audio.set_frame_rate(SR).set_channels(1).export(str(out_path), format="wav")
    return out_path

def get_duration_sec(path: str) -> float:
    try:
        return float(librosa.get_duration(path=path))
    except Exception:
        return 0.0
