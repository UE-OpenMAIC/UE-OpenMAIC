import bpy
import os
import sys


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    if len(argv) < 4:
        raise RuntimeError(
            "Usage: blender -b -P fix_transition_and_loop.py -- <prev_fbx> <current_fbx> <output_fbx> <end_frame> [blend_head] [blend_tail]"
        )
    prev_fbx = argv[0]
    current_fbx = argv[1]
    output_fbx = argv[2]
    end_frame = int(argv[3])
    blend_head = int(argv[4]) if len(argv) >= 5 else 10
    blend_tail = int(argv[5]) if len(argv) >= 6 else 12
    return prev_fbx, current_fbx, output_fbx, end_frame, blend_head, blend_tail


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_fbx(path):
    bpy.ops.import_scene.fbx(filepath=path)


def get_first_action():
    actions = list(bpy.data.actions)
    if not actions:
        raise RuntimeError("No action found.")
    return actions[0]


def sample_action_values(action, frame):
    values = {}
    for fcurve in action.fcurves:
        # Only sample pose bone rotations for cross-action transition matching.
        if not fcurve.data_path.startswith('pose.bones["'):
            continue
        if ".rotation_quaternion" not in fcurve.data_path and ".rotation_euler" not in fcurve.data_path:
            continue
        values[fcurve.data_path, fcurve.array_index] = fcurve.evaluate(frame)
    return values


def trim_action_to_end(action, end_frame):
    start, end = action.frame_range
    start = int(round(start))
    end = int(round(end))
    target_end = max(start + 1, min(end, int(end_frame)))
    for fcurve in action.fcurves:
        for i in range(len(fcurve.keyframe_points) - 1, -1, -1):
            if fcurve.keyframe_points[i].co.x > target_end:
                fcurve.keyframe_points.remove(fcurve.keyframe_points[i], fast=True)
        fcurve.update()
    action.frame_start = start
    action.frame_end = target_end
    return start, target_end


def blend_head_to_prev_tail(action, prev_tail_values, start, blend_head):
    total = max(2, blend_head)
    head_end = start + total - 1
    for fcurve in action.fcurves:
        # Only blend pose bone rotations; do NOT touch object/root location/scale.
        if not fcurve.data_path.startswith('pose.bones["'):
            continue
        if ".rotation_quaternion" not in fcurve.data_path and ".rotation_euler" not in fcurve.data_path:
            continue
        key = (fcurve.data_path, fcurve.array_index)
        if key not in prev_tail_values:
            continue
        target = prev_tail_values[key]
        for frame in range(start, head_end + 1):
            t = (frame - start + 1) / total
            cur = fcurve.evaluate(frame)
            blended = target * (1.0 - t) + cur * t
            fcurve.keyframe_points.insert(frame=frame, value=blended, options={"FAST"})
        fcurve.update()


def blend_tail_to_head_for_loop(action, start, end, blend_tail):
    total = max(2, min(blend_tail, end - start))
    head_values = {}
    for fcurve in action.fcurves:
        # Only blend pose bone rotations; keep translation/scale intact.
        if not fcurve.data_path.startswith('pose.bones["'):
            continue
        if ".rotation_quaternion" not in fcurve.data_path and ".rotation_euler" not in fcurve.data_path:
            continue
        head_values[(fcurve.data_path, fcurve.array_index)] = fcurve.evaluate(start)

    tail_start = end - total + 1
    for fcurve in action.fcurves:
        if not fcurve.data_path.startswith('pose.bones["'):
            continue
        if ".rotation_quaternion" not in fcurve.data_path and ".rotation_euler" not in fcurve.data_path:
            continue
        key = (fcurve.data_path, fcurve.array_index)
        target = head_values[key]
        for frame in range(tail_start, end):
            t = (frame - tail_start + 1) / total
            cur = fcurve.evaluate(frame)
            blended = cur * (1.0 - t) + target * t
            fcurve.keyframe_points.insert(frame=frame, value=blended, options={"FAST"})
        fcurve.keyframe_points.insert(frame=end, value=target, options={"FAST"})
        for kp in fcurve.keyframe_points:
            kp.interpolation = "BEZIER"
        fcurve.update()


def assign_action_to_armatures(action):
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            if obj.animation_data is None:
                obj.animation_data_create()
            obj.animation_data.action = action


def export_fbx(path):
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
    prev_fbx, current_fbx, output_fbx, end_frame, blend_head, blend_tail = parse_args()

    # Read previous action tail values
    clear_scene()
    import_fbx(prev_fbx)
    prev_action = get_first_action()
    _, prev_end = prev_action.frame_range
    prev_tail_values = sample_action_values(prev_action, int(round(prev_end)))

    # Process current action
    clear_scene()
    import_fbx(current_fbx)
    action = get_first_action()
    start, end = trim_action_to_end(action, end_frame)
    blend_head_to_prev_tail(action, prev_tail_values, start, blend_head)
    blend_tail_to_head_for_loop(action, start, end, blend_tail)
    assign_action_to_armatures(action)
    export_fbx(output_fbx)
    print(
        f"[OK] transition+loop fixed: {current_fbx} -> {output_fbx}, frames {start}-{end}, head_blend={blend_head}, tail_blend={blend_tail}"
    )


if __name__ == "__main__":
    main()
