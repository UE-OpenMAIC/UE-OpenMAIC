import bpy
import sys


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    if len(argv) < 1:
        raise RuntimeError("Usage: blender -b -P inspect_fbx_action_range.py -- <fbx_path>")

    fbx_path = argv[0]
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    if not bpy.data.actions:
        print("[ERR] no action")
        return
    for idx, action in enumerate(bpy.data.actions):
        start, end = action.frame_range
        min_k = 10**9
        max_k = -10**9
        for fc in action.fcurves:
            for kp in fc.keyframe_points:
                x = kp.co.x
                if x < min_k:
                    min_k = x
                if x > max_k:
                    max_k = x
        print(
            f"[ACTION {idx}] name={action.name} frame_range={start:.3f}-{end:.3f} keys={min_k:.3f}-{max_k:.3f}"
        )


if __name__ == "__main__":
    main()
