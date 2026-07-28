#!/usr/bin/env python3
"""
Headless QRemeshify runner for Blender 4.2+.

Usage:
  blender --background --python scripts/run_qremeshify.py -- \\
    --input /path/to/mesh.glb \\
    --output /path/to/output.glb
"""

import argparse
import sys


def parse_script_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preprocess", action="store_true", default=True)
    parser.add_argument("--no-preprocess", dest="preprocess", action="store_false")
    parser.add_argument("--smoothing", action="store_true", default=True)
    parser.add_argument("--no-smoothing", dest="smoothing", action="store_false")
    parser.add_argument("--sharp-angle", type=float, default=35.0)
    parser.add_argument("--scale-fact", type=float, default=1.0)
    parser.add_argument("--merge-distance", type=float, default=0.002)
    parser.add_argument("--target-tris", type=int, default=100000)
    return parser.parse_args(argv)


def main():
    import bpy
    import bmesh

    args = parse_script_args()

    bpy.ops.wm.read_factory_settings(use_empty=True)

    ext = args.input.lower()
    if ext.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=args.input)
    elif ext.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=args.input)
    elif ext.endswith(".stl"):
        bpy.ops.wm.stl_import(filepath=args.input)
    else:
        raise RuntimeError(f"Unsupported input format: {args.input}")

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError("No mesh objects found after import")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    print(f"Working mesh: {obj.name}, faces={len(obj.data.polygons)}")

    # Basic cleanup for fragmented AI meshes.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=args.merge_distance)
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    face_count = len(obj.data.polygons)
    if face_count > args.target_tris:
        ratio = args.target_tris / face_count
        decimate = obj.modifiers.new(name="Decimate", type="DECIMATE")
        decimate.ratio = ratio
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=decimate.name)
        print(f"Decimated to {len(obj.data.polygons)} faces")

    props = bpy.context.scene.quadwild_props
    props.useCache = False
    props.enableRemesh = args.preprocess
    props.enableSmoothing = args.smoothing
    props.enableSharp = True
    props.sharpAngle = args.sharp_angle
    props.symmetryX = False
    props.symmetryY = False
    props.symmetryZ = False

    qr_props = bpy.context.scene.quadpatches_props
    qr_props.scaleFact = args.scale_fact

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    print("Running QRemeshify...")
    result = bpy.ops.qremeshify.remesh()
    if result != {"FINISHED"}:
        raise RuntimeError(f"QRemeshify failed: {result}")

    remeshed = bpy.context.view_layer.objects.active
    print(f"Remeshed result: {remeshed.name}, faces={len(remeshed.data.polygons)}")

    out = args.output.lower()
    if out.endswith((".glb", ".gltf")):
        bpy.ops.export_scene.gltf(filepath=args.output, export_format="GLB")
    elif out.endswith(".obj"):
        bpy.ops.wm.obj_export(filepath=args.output)
    else:
        raise RuntimeError(f"Unsupported output format: {args.output}")

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
