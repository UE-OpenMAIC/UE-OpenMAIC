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
            "Usage: blender -b -P trim_to_1_end_loop_fbx.py -- <input_fbx> <output_fbx> <end_frame> [blend_frames]"
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
    end = max(2, min(src_end, end_frame))
    start = 1

    for fc in action.fcurves:
        # remove keys after requested end
        for i in range(len(fc.keyframe_points) - 1, -1, -1):
            if fc.keyframe_points[i].co.x > end:
                fc.keyframe_points.remove(fc.keyframe_points[i], fast=True)

        # ensure key at frame 1 to make explicit 1~end range
        v1 = fc.evaluate(src_start)
        fc.keyframe_points.insert(frame=1, value=v1, options={"FAST"})
        fc.update()

    # tail -> head smoothing
    blend = max(2, min(blend_frames, end - start))
    tail_start = end - blend + 1
    for fc in action.fcurves:
        head_val = fc.evaluate(start)
        for frame in range(tail_start, end):
            t = (frame - tail_start + 1) / blend
            cur = fc.evaluate(frame)
            v = cur * (1.0 - t) + head_val * t
            fc.keyframe_points.insert(frame=frame, value=v, options={"FAST"})
        fc.keyframe_points.insert(frame=end, value=head_val, options={"FAST"})
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
        fc.update()

    action.frame_start = start
    action.frame_end = end
    return start, end


def assign_action(action):
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            if obj.animation_data is None:
                obj.animation_data_create()
            obj.animation_data.action = action


def export_fbx(path, start, end):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_start = start
    scene.frame_end = end
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
    start, end = process_action(action, end_frame, blend_frames)
    assign_action(action)
    export_fbx(output_fbx, start, end)
    print(f"[OK] trim1-end loop: out {start}-{end}, blend={blend_frames}")


if __name__ == "__main__":
    main()
