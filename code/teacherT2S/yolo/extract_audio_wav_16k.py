from moviepy import VideoFileClip
from pathlib import Path
import os
INPUT_ROOT = 'D:\\code\\teacherT2S\\yolo\\input'
OUTPUT_ROOT = 'D:\\code\\teacherT2S\\yolo\\audio_wav'
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}
OUTPUT_FPS = 16000
OUTPUT_NBYTES = 2
OUTPUT_CODEC = 'pcm_s16le'
SKIP_EXISTING = True

def is_valid_existing_file(path):
    path = Path(path)
    return path.exists() and path.is_file() and (path.stat().st_size > 0)

def collect_all_videos(input_root):
    input_root = Path(input_root)
    videos = []
    for p in input_root.rglob('*'):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
    videos.sort()
    return videos

def make_output_audio_path(video_path, input_root, output_root):
    video_path = Path(video_path)
    input_root = Path(input_root)
    output_root = Path(output_root)
    rel_path = video_path.relative_to(input_root)
    out_wav = output_root / rel_path.with_suffix('.wav')
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    return out_wav

def extract_audio_from_video(input_video, output_audio):
    print('=' * 80)
    print(f'开始处理视频: {input_video}')
    print(f'输出音频路径: {output_audio}')
    clip = None
    try:
        clip = VideoFileClip(str(input_video))
        if clip.audio is None:
            raise RuntimeError('该视频没有音轨')
        clip.audio.write_audiofile(str(output_audio), fps=OUTPUT_FPS, nbytes=OUTPUT_NBYTES, codec=OUTPUT_CODEC)
        print(f'完成，输出：{output_audio}')
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass

def main():
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    if not input_root.exists():
        raise FileNotFoundError(f'输入根目录不存在: {input_root}')
    videos = collect_all_videos(input_root)
    if len(videos) == 0:
        print(f'没有在 {input_root} 下找到视频文件')
        return
    print(f'共找到 {len(videos)} 个视频：')
    for i, v in enumerate(videos, 1):
        print(f'{i:03d}. {v}')
    success_count = 0
    fail_count = 0
    skip_existing_count = 0
    for video_path in videos:
        output_audio = make_output_audio_path(video_path, input_root, output_root)
        if SKIP_EXISTING and is_valid_existing_file(output_audio):
            skip_existing_count += 1
            print('=' * 80)
            print(f'[跳过] 已存在音频 WAV: {output_audio}')
            continue
        try:
            extract_audio_from_video(video_path, output_audio)
            success_count += 1
        except Exception as e:
            fail_count += 1
            print('=' * 80)
            print(f'[失败] {video_path}')
            print(f'原因: {e}')
    print('=' * 80)
    print('全部处理完成')
    print(f'成功生成: {success_count}')
    print(f'失败: {fail_count}')
    print(f'跳过已有: {skip_existing_count}')
    print(f'输出根目录: {output_root}')
if __name__ == '__main__':
    main()
