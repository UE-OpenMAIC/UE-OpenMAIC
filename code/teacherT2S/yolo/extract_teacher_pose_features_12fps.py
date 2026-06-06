from ultralytics import YOLO
import cv2
import csv
import numpy as np
import os
from pathlib import Path
INPUT_ROOT = 'D:\\code\\teacherT2S\\yolo\\input'
OUTPUT_ROOT = 'D:\\code\\teacherT2S\\yolo\\pose_csv'
MODEL_NAME = 'yolo11n-pose.pt'
SAMPLE_FPS = 12
MAX_SECONDS = None
PERSON_CONF_THRES = 0.25
KPT_CONF_THRES = 0.4
BOX_MARGIN = 5
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}
KPT_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
ARM_POINTS = ['left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist']

def point_in_box(x, y, x1, y1, x2, y2, margin=0):
    return x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin

def build_csv_header():
    return ['frame_id', 'time_sec', 'center_x', 'center_y', 'orientation_score', 'left_shoulder_x', 'left_shoulder_y', 'left_elbow_x', 'left_elbow_y', 'left_wrist_x', 'left_wrist_y', 'right_shoulder_x', 'right_shoulder_y', 'right_elbow_x', 'right_elbow_y', 'right_wrist_x', 'right_wrist_y']

def normalize_x(x, frame_width):
    if x is None:
        return ''
    return round(float(x) / frame_width, 6)

def normalize_y(y, frame_height):
    if y is None:
        return ''
    return round(float(y) / frame_height, 6)

def compute_orientation_score(valid_points):
    needed = ['nose', 'left_shoulder', 'right_shoulder']
    for name in needed:
        if name not in valid_points:
            return ''
    nose_x, nose_y = valid_points['nose']
    ls_x, ls_y = valid_points['left_shoulder']
    rs_x, rs_y = valid_points['right_shoulder']
    shoulder_mid_x = (ls_x + rs_x) / 2.0
    shoulder_width = np.sqrt((ls_x - rs_x) ** 2 + (ls_y - rs_y) ** 2)
    if shoulder_width < 1e-06:
        return ''
    score = (nose_x - shoulder_mid_x) / shoulder_width
    return round(float(score), 6)

def empty_row(frame_id, time_sec):
    return [frame_id, round(time_sec, 6), '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

def collect_all_videos(input_root):
    input_root = Path(input_root)
    videos = []
    for p in input_root.rglob('*'):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
    videos.sort()
    return videos

def make_output_csv_path(video_path, input_root, output_root):
    video_path = Path(video_path)
    input_root = Path(input_root)
    output_root = Path(output_root)
    rel_path = video_path.relative_to(input_root)
    out_csv = output_root / rel_path.with_suffix('.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    return out_csv

def choose_mode():
    print('\n请选择本次运行模式：')
    print('1 = 对目录下所有视频生成 pose CSV；如果 CSV 已存在则跳过')
    print('2 = 对目录下所有视频生成 pose CSV；即使 CSV 已存在也覆盖')
    print('4 = 只更新指定编号视频的 pose CSV；覆盖已有 CSV')
    print('5 = 只处理指定编号视频的 pose CSV；如果 CSV 已存在则跳过')
    while True:
        mode = input('请输入 1 / 2 / 4 / 5 : ').strip()
        if mode in {'1', '2', '4', '5'}:
            return mode
        print('输入无效，请重新输入。')

def parse_index_selection(s, max_n):
    s = str(s).strip().replace('，', ',')
    if not s:
        return []
    selected = set()
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            if not a.strip().isdigit() or not b.strip().isdigit():
                raise ValueError(f'编号范围格式错误: {part}')
            a = int(a.strip())
            b = int(b.strip())
            if a > b:
                a, b = (b, a)
            for x in range(a, b + 1):
                if 1 <= x <= max_n:
                    selected.add(x - 1)
                else:
                    raise ValueError(f'编号超出范围: {x}，应在 1 ~ {max_n} 之间')
        else:
            if not part.isdigit():
                raise ValueError(f'编号格式错误: {part}')
            x = int(part)
            if 1 <= x <= max_n:
                selected.add(x - 1)
            else:
                raise ValueError(f'编号超出范围: {x}，应在 1 ~ {max_n} 之间')
    return sorted(selected)

def choose_target_videos_by_index(videos):
    print('\n请输入要处理的视频编号（与上面列表一致，1-based）：')
    print('可以输入单个编号，例如：1')
    print('也可以输入多个编号，例如：1,3,5 或 1-5')
    while True:
        s = input('视频编号: ').strip()
        try:
            idxs = parse_index_selection(s, len(videos))
            if not idxs:
                print('没有选择任何视频，请重新输入。')
                continue
            targets = [videos[i] for i in idxs]
            print('已选择：')
            for v in targets:
                print(f'  {v}')
            return targets
        except Exception as e:
            print(f'输入无效：{e}')
            print('请重新输入。')

def process_one_video(model, input_video, output_csv):
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {input_video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        cap.release()
        raise RuntimeError(f'视频 FPS 读取失败: {input_video}')
    duration = total_frames / fps
    print('=' * 80)
    print(f'开始处理: {input_video}')
    print(f'输出路径: {output_csv}')
    print(f'视频 FPS: {fps:.4f}')
    print(f'总帧数: {total_frames}')
    print(f'视频时长: {duration:.2f} 秒')
    print(f'分辨率: {frame_width} x {frame_height}')
    sample_interval = 1.0 / SAMPLE_FPS
    next_sample_time = 0.0
    rows = []
    frame_id = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = frame_id / fps
        if MAX_SECONDS is not None and current_time > MAX_SECONDS:
            break
        if current_time + 1e-09 < next_sample_time:
            frame_id += 1
            continue
        next_sample_time += sample_interval
        results = model(frame, verbose=False)
        r = results[0]
        row = empty_row(frame_id, current_time)
        if r.boxes is not None and len(r.boxes) > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            valid_person_indices = np.where(confs >= PERSON_CONF_THRES)[0]
            if len(valid_person_indices) > 0:
                best_local = valid_person_indices[np.argmax(confs[valid_person_indices])]
                box = boxes_xyxy[best_local]
                x1, y1, x2, y2 = box.tolist()
                bbox_cx = (x1 + x2) / 2.0
                bbox_cy = (y1 + y2) / 2.0
                center_x = normalize_x(bbox_cx, frame_width)
                center_y = normalize_y(bbox_cy, frame_height)
                valid_points = {}
                if r.keypoints is not None and len(r.keypoints.xy) > best_local:
                    kpts_xy = r.keypoints.xy.cpu().numpy()[best_local]
                    if hasattr(r.keypoints, 'conf') and r.keypoints.conf is not None:
                        kpts_conf = r.keypoints.conf.cpu().numpy()[best_local]
                    else:
                        kpts_conf = np.ones(len(kpts_xy), dtype=np.float32)
                    for i, (x, y) in enumerate(kpts_xy):
                        conf_i = float(kpts_conf[i])
                        name = KPT_NAMES[i]
                        valid = conf_i >= KPT_CONF_THRES and point_in_box(x, y, x1, y1, x2, y2, margin=BOX_MARGIN)
                        if valid:
                            valid_points[name] = (float(x), float(y))
                orientation_score = compute_orientation_score(valid_points)
                row = [frame_id, round(current_time, 6), center_x, center_y, orientation_score]
                for name in ARM_POINTS:
                    if name in valid_points:
                        px, py = valid_points[name]
                        row.extend([normalize_x(px, frame_width), normalize_y(py, frame_height)])
                    else:
                        row.extend(['', ''])
        rows.append(row)
        saved_count += 1
        if saved_count % 50 == 0:
            print(f'  已写入 {saved_count} 行采样数据')
        frame_id += 1
    cap.release()
    header = build_csv_header()
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f'完成，CSV 已保存到: {output_csv}')
    print(f'总共写入 {len(rows)} 行')
    if MAX_SECONDS is not None:
        print(f'处理时长限制: 前 {MAX_SECONDS} 秒')
    else:
        print('处理范围: 全视频')
    print('输出特征维度（不含 frame_id/time_sec）: 15')

def main():
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)
    if not input_root.exists():
        raise FileNotFoundError(f'输入根目录不存在: {input_root}')
    videos_all = collect_all_videos(input_root)
    if len(videos_all) == 0:
        print(f'没有在 {input_root} 下找到视频文件')
        return
    print(f'共找到 {len(videos_all)} 个视频')
    for i, v in enumerate(videos_all, 1):
        print(f'{i:03d}. {v}')
    mode = choose_mode()
    if mode in {'4', '5'}:
        videos = choose_target_videos_by_index(videos_all)
    else:
        videos = videos_all
    if mode == '1':
        print('模式1：批量生成所有视频 pose CSV；已有 CSV 则跳过。')
        skip_existing = True
    elif mode == '2':
        print('模式2：批量生成所有视频 pose CSV；已有 CSV 也覆盖。')
        skip_existing = False
    elif mode == '4':
        print('模式4：只更新指定编号视频的 pose CSV；覆盖已有 CSV。')
        skip_existing = False
    elif mode == '5':
        print('模式5：只处理指定编号视频的 pose CSV；已有 CSV 则跳过。')
        skip_existing = True
    else:
        raise ValueError(f'未知模式: {mode}')
    print('=' * 80)
    print(f'本次实际处理视频数: {len(videos)}')
    for i, v in enumerate(videos, 1):
        print(f'{i:03d}. {v}')
    print('=' * 80)
    model = YOLO(MODEL_NAME)
    success_count = 0
    fail_count = 0
    skip_count = 0
    for video_path in videos:
        output_csv = make_output_csv_path(video_path, input_root, output_root)
        if skip_existing and output_csv.exists():
            skip_count += 1
            print('=' * 80)
            print(f'[跳过] 输出 CSV 已存在: {output_csv}')
            continue
        try:
            process_one_video(model, video_path, output_csv)
            success_count += 1
        except Exception as e:
            fail_count += 1
            print(f'[失败] {video_path}')
            print(f'原因: {e}')
    print('=' * 80)
    print('全部处理完成')
    print(f'成功: {success_count}')
    print(f'失败: {fail_count}')
    print(f'跳过: {skip_count}')
    print(f'输出根目录: {output_root}')
if __name__ == '__main__':
    main()
