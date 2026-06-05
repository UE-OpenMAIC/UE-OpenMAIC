import bpy
import os
import sys


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    if len(argv) < 3:
        raise RuntimeError(
            "Usage: blender -b -P remap_trim_loop_fbx.py -- <input_fbx> <output_fbx> <end_frame> [blend_frames]"
        )
    input_fbx = argv[0]
    output_fbx = argv[1]
    end_frame = int(argv[2])
    blend_frames = int(argv[3]) if len(argv) >= 4 else 12
    return input_fbx, output_fbx, end_frame, blend_frames


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_fbx(path):
    bpy.ops.import_scene.fbx(filepath=path)


def get_action():
    if not bpy.data.actions:
        raise RuntimeError("No action found.")
    return bpy.data.actions[0]


def process_action(action, end_frame, blend_frames):
    src_start, src_end = action.frame_range
    src_start = int(round(src_start))
    src_end = int(round(src_end))
    cut_end = max(src_start + 1, min(src_end, end_frame))

    # Shift kept range so src_start -> 1
    shift = src_start - 1
    new_end = cut_end - shift

    for fc in action.fcurves:
        # remove keys outside [src_start, cut_end]
        for i in range(len(fc.keyframe_points) - 1, -1, -1):
            x = fc.keyframe_points[i].co.x
            if x < src_start or x > cut_end:
                fc.keyframe_points.remove(fc.keyframe_points[i], fast=True)

        # shift keys to start at frame 1
        for kp in fc.keyframe_points:
            kp.co.x -= shift
            kp.handle_left.x -= shift
            kp.handle_right.x -= shift

        fc.update()

    # loop smooth (tail -> head)
    blend = max(2, min(blend_frames, new_end - 1))
    tail_start = new_end - blend + 1
    for fc in action.fcurves:
        head_val = fc.evaluate(1)
        for frame in range(tail_start, new_end):
            t = (frame - tail_start + 1) / blend
            cur = fc.evaluate(frame)
            v = cur * (1.0 - t) + head_val * t
            fc.keyframe_points.insert(frame=frame, value=v, options={"FAST"})
        fc.keyframe_points.insert(frame=new_end, value=head_val, options={"FAST"})
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
        fc.update()

    action.frame_start = 1
    action.frame_end = new_end
    return 1, new_end, src_start, cut_end


def assign_action(action):
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            if obj.animation_data is None:
                obj.animation_data_create()
            obj.animation_data.action = action


def export_fbx(path, frame_start, frame_end):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
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
    input_fbx, output_fbx, end_frame, blend_frames = parse_args()
    clear_scene()
    import_fbx(input_fbx)
    action = get_action()
    start, end, src_start, cut_end = process_action(action, end_frame, blend_frames)
    assign_action(action)
    export_fbx(output_fbx, start, end)
    print(
        f"[OK] remap+trim+loop: src {src_start}-{cut_end} -> out {start}-{end}, blend={blend_frames}"
    )


if __name__ == "__main__":
    main()
