import wave
import numpy as np
import pandas as pd
from pathlib import Path
INPUT_CSV_ROOT = 'D:\\code\\teacherT2S\\yolo\\pose_csv'
INPUT_WAV_ROOT = 'D:\\code\\teacherT2S\\yolo\\audio_wav'
OUTPUT_CSV_ROOT = 'D:\\code\\teacherT2S\\yolo\\pose_audio_csv'
TIME_COL = 'time_sec'
CENTER_WINDOW_SEC = 0.5
SILENCE_RMS_THRES = 0.01
TARGET_SR_EXPECTED = 16000
SKIP_EXISTING = True

def is_valid_existing_file(path):
    path = Path(path)
    return path.exists() and path.is_file() and (path.stat().st_size > 0)

def collect_all_csv_files(input_root):
    input_root = Path(input_root)
    files = [p for p in input_root.rglob('*.csv') if p.is_file()]
    files.sort()
    return files

def make_corresponding_wav_path(csv_path, csv_root, wav_root):
    csv_path = Path(csv_path)
    csv_root = Path(csv_root)
    wav_root = Path(wav_root)
    rel_path = csv_path.relative_to(csv_root)
    return wav_root / rel_path.with_suffix('.wav')

def make_output_csv_path(csv_path, csv_root, output_root):
    csv_path = Path(csv_path)
    csv_root = Path(csv_root)
    output_root = Path(output_root)
    rel_path = csv_path.relative_to(csv_root)
    out_csv = output_root / rel_path
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    return out_csv

def read_wav_mono_float32(wav_path):
    with wave.open(str(wav_path), 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        x = x / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        x = x / 2147483648.0
    else:
        raise ValueError(f'不支持的 wav sampwidth={sampwidth} bytes: {wav_path}')
    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return (x, int(sr))

def safe_mean(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.mean(arr))

def window_audio_features(y, sr, center_t, half_w):
    start = max(0, int(round((center_t - half_w) * sr)))
    end = min(len(y), int(round((center_t + half_w) * sr)))
    if end <= start:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    win = y[start:end]
    if len(win) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    rms = float(np.sqrt(np.mean(win * win) + 1e-12))
    if len(win) >= 2:
        signs = np.signbit(win)
        zcr = float(np.mean(signs[1:] != signs[:-1]))
    else:
        zcr = np.nan
    frame_len = max(1, int(round(0.025 * sr)))
    hop = max(1, int(round(0.01 * sr)))
    frame_rms = []
    for s in range(0, max(1, len(win) - frame_len + 1), hop):
        fr = win[s:s + frame_len]
        if len(fr) == 0:
            continue
        frame_rms.append(float(np.sqrt(np.mean(fr * fr) + 1e-12)))
    if len(frame_rms) == 0:
        silence_ratio = float(rms < SILENCE_RMS_THRES)
        voiced_ratio = float(rms >= SILENCE_RMS_THRES)
    else:
        frame_rms = np.asarray(frame_rms, dtype=np.float32)
        silence_ratio = float(np.mean(frame_rms < SILENCE_RMS_THRES))
        voiced_th = max(SILENCE_RMS_THRES, float(np.quantile(frame_rms, 0.35)))
        voiced_ratio = float(np.mean(frame_rms >= voiced_th))
    f0_mean = np.nan
    return (rms, zcr, voiced_ratio, f0_mean, silence_ratio)

def process_one_pair(input_csv, input_wav, output_csv):
    print('=' * 80)
    print(f'开始处理视觉 CSV: {input_csv}')
    print(f'对应音频 WAV: {input_wav}')
    print(f'输出路径: {output_csv}')
    df = pd.read_csv(input_csv)
    if TIME_COL not in df.columns:
        raise ValueError(f'CSV 中找不到时间列: {TIME_COL}')
    times = df[TIME_COL].astype(float).values
    y, sr = read_wav_mono_float32(input_wav)
    print(f'音频长度: {len(y) / max(sr, 1):.2f} 秒, sr={sr}')
    if sr != TARGET_SR_EXPECTED:
        print(f'[提示] 当前 wav sr={sr}，不是预期 {TARGET_SR_EXPECTED}。仍按实际 sr 计算。')
    half_w = CENTER_WINDOW_SEC / 2.0
    audio_rms_list = []
    audio_zcr_list = []
    audio_voiced_ratio_list = []
    audio_f0_mean_list = []
    audio_silence_ratio_list = []
    for t in times:
        audio_rms, audio_zcr, audio_voiced_ratio, audio_f0_mean, audio_silence_ratio = window_audio_features(y=y, sr=sr, center_t=float(t), half_w=half_w)
        audio_rms_list.append(audio_rms)
        audio_zcr_list.append(audio_zcr)
        audio_voiced_ratio_list.append(audio_voiced_ratio)
        audio_f0_mean_list.append(audio_f0_mean)
        audio_silence_ratio_list.append(audio_silence_ratio)
    df['audio_rms'] = audio_rms_list
    df['audio_zcr'] = audio_zcr_list
    df['audio_voiced_ratio'] = audio_voiced_ratio_list
    df['audio_f0_mean'] = audio_f0_mean_list
    df['audio_silence_ratio'] = audio_silence_ratio_list
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f'完成，已输出: {output_csv}')
    print('新增列：audio_rms, audio_zcr, audio_voiced_ratio, audio_f0_mean, audio_silence_ratio')

def main():
    csv_root = Path(INPUT_CSV_ROOT)
    wav_root = Path(INPUT_WAV_ROOT)
    output_root = Path(OUTPUT_CSV_ROOT)
    if not csv_root.exists():
        raise FileNotFoundError(f'视觉 CSV 根目录不存在: {csv_root}')
    if not wav_root.exists():
        raise FileNotFoundError(f'音频 WAV 根目录不存在: {wav_root}')
    output_root.mkdir(parents=True, exist_ok=True)
    csv_files = collect_all_csv_files(csv_root)
    if len(csv_files) == 0:
        print(f'没有在 {csv_root} 下找到 CSV 文件')
        return
    print(f'共找到 {len(csv_files)} 个视觉 CSV：')
    for i, p in enumerate(csv_files, 1):
        print(f'{i:03d}. {p}')
    success_count = 0
    fail_count = 0
    skip_missing_wav_count = 0
    skip_existing_count = 0
    for csv_path in csv_files:
        wav_path = make_corresponding_wav_path(csv_path, csv_root, wav_root)
        output_csv = make_output_csv_path(csv_path, csv_root, output_root)
        if SKIP_EXISTING and is_valid_existing_file(output_csv):
            skip_existing_count += 1
            print('=' * 80)
            print(f'[跳过] 已存在融合 CSV: {output_csv}')
            continue
        if not wav_path.exists():
            skip_missing_wav_count += 1
            print('=' * 80)
            print(f'[跳过] 找不到对应音频: {wav_path}')
            continue
        try:
            process_one_pair(csv_path, wav_path, output_csv)
            success_count += 1
        except Exception as e:
            fail_count += 1
            print('=' * 80)
            print(f'[失败] {csv_path}')
            print(f'原因: {e}')
    print('=' * 80)
    print('全部处理完成')
    print(f'成功生成: {success_count}')
    print(f'失败: {fail_count}')
    print(f'跳过(缺少wav): {skip_missing_wav_count}')
    print(f'跳过已有: {skip_existing_count}')
    print(f'输出根目录: {output_root}')
if __name__ == '__main__':
    main()
