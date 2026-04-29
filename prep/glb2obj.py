"""
Blender script: convert all GLB files in a directory to OBJ.
Usage (called by run_hy3d_recon.py):
    blender -b -P glb2obj.py -- <glb_dir> <out_dir>
Produces: <out_dir>/<glb_basename>/<glb_basename>.obj
"""

import bpy
import sys
import os


def process_glb_file_with_decimation(glb_dir, out_dir):
    glb_files = [f for f in os.listdir(glb_dir) if f.endswith('.glb')]
    if not glb_files:
        print(f"No GLB files found in {glb_dir}")
        return

    for glb_file in glb_files:
        glb_path = os.path.join(glb_dir, glb_file)
        glb_basename = os.path.splitext(glb_file)[0]
        obj_subdir = os.path.join(out_dir, glb_basename)
        os.makedirs(obj_subdir, exist_ok=True)
        obj_path = os.path.join(obj_subdir, f"{glb_basename}.obj")

        print(f"Converting {glb_path} -> {obj_path}")

        # Reset scene
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Import GLB
        bpy.ops.import_scene.gltf(filepath=glb_path)

        # Select all mesh objects
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj

        # Export OBJ
        bpy.ops.wm.obj_export(
            filepath=obj_path,
            export_selected_objects=False,
            export_materials=True,
            export_triangulated_mesh=True,
        )
        print(f"Saved OBJ: {obj_path}")


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        print("No arguments provided after '--'")
        sys.exit(1)

    glb_dir = argv[0]
    out_dir = argv[1]
    process_glb_file_with_decimation(glb_dir, out_dir)