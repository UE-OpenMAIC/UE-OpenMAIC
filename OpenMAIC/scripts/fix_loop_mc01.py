import bpy
import os
import sys


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    if len(argv) < 2:
        raise RuntimeError(
            "Usage: blender -b -P fix_loop_mc01.py -- <input_fbx> <output_fbx> [blend_frames] [end_frame] [start_frame]"
        )
    input_fbx = argv[0]
    output_fbx = argv[1]
    blend_frames = int(argv[2]) if len(argv) >= 3 else 12
    end_frame = int(argv[3]) if len(argv) >= 4 else None
    start_frame = int(argv[4]) if len(argv) >= 5 else None
    return input_fbx, output_fbx, blend_frames, end_frame, start_frame


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_fbx(path):
    bpy.ops.import_scene.fbx(filepath=path)


def get_action():
    actions = list(bpy.data.actions)
    if not actions:
        return None
    return actions[0]


def make_loop(action, blend_frames=12, clamp_end=None, clamp_start=None):
    start, end = action.frame_range
    start = int(round(start))
    end = int(round(end))
    if clamp_start is not None:
        start = int(clamp_start)
    if clamp_end is not None:
        end = max(start + 1, min(end, int(clamp_end)))
    total = max(1, end - start)
    blend_frames = max(2, min(blend_frames, total))

    for fcurve in action.fcurves:
        # Ensure explicit key at start frame when forcing an earlier start.
        fcurve.keyframe_points.insert(frame=start, value=fcurve.evaluate(start), options={"FAST"})
        # 1) Hard-trim: remove all keys after target end frame
        for i in range(len(fcurve.keyframe_points) - 1, -1, -1):
            if fcurve.keyframe_points[i].co.x > end:
                fcurve.keyframe_points.remove(fcurve.keyframe_points[i], fast=True)

        first_kp = None
        for kp in fcurve.keyframe_points:
            if int(round(kp.co.x)) == start:
                first_kp = kp
                break
        if first_kp is None:
            first_val = fcurve.evaluate(start)
        else:
            first_val = first_kp.co.y

        # Ensure final frame equals first frame value
        fcurve.keyframe_points.insert(frame=end, value=first_val, options={"FAST"})

        # Smoothly pull the tail segment toward first frame over N frames
        tail_start = end - blend_frames + 1
        for frame in range(tail_start, end):
            t = (frame - tail_start + 1) / blend_frames
            current = fcurve.evaluate(frame)
            blended = current * (1.0 - t) + first_val * t
            fcurve.keyframe_points.insert(frame=frame, value=blended, options={"FAST"})

        for kp in fcurve.keyframe_points:
            kp.interpolation = "BEZIER"

        fcurve.update()

    # Force action range to trimmed range
    action.frame_start = start
    action.frame_end = end

    return start, end


def assign_action_to_armatures(action):
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        if obj.animation_data is None:
            obj.animation_data_create()
        obj.animation_data.action = action


def export_fbx(path, start, end):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=False,
        bake_anim=True,
        bake_anim_use_all_actions=False,
        bake_anim_simplify_factor=0.0,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_use_nla_strips=False,
        add_leaf_bones=False,
    )


def main():
    input_fbx, output_fbx, blend_frames, end_frame, start_frame = parse_args()
    clear_scene()
    import_fbx(input_fbx)

    action = get_action()
    if action is None:
        raise RuntimeError("No action found in imported FBX.")

    start, end = make_loop(
        action,
        blend_frames=blend_frames,
        clamp_end=end_frame,
        clamp_start=start_frame,
    )
    assign_action_to_armatures(action)
    export_fbx(output_fbx, start, end)
    print(f"[OK] loop fixed: {input_fbx} -> {output_fbx}, frames {start}-{end}, blend={blend_frames}")


if __name__ == "__main__":
    main()
