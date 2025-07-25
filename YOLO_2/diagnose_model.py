#!/usr/bin/env python3
"""
Diagnostic script to understand the Sketchfab Coke Can model structure
Run this with: blender --background --python diagnose_model.py
"""

import bpy
import sys
from pathlib import Path
from mathutils import Vector

def clear_scene():
    """Clear all objects from scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def diagnose_model():
    """Analyze the FBX model structure"""
    print("\n" + "="*60)
    print("COCA-COLA CAN MODEL DIAGNOSTICS")
    print("="*60)
    
    # Clear scene first
    clear_scene()
    
    # Import the FBX
    fbx_path = Path("source") / "Coke Can.fbx"
    if not fbx_path.exists():
        print(f"❌ Error: Model not found at {fbx_path}")
        return
    
    print(f"\n📁 Loading model: {fbx_path}")
    
    # Track what's in scene before import
    before_import = set(bpy.data.objects[:])
    
    try:
        bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    except Exception as e:
        print(f"❌ Error importing FBX: {e}")
        return
    
    # Find what was imported
    after_import = set(bpy.data.objects[:])
    imported_objects = list(after_import - before_import)
    
    print(f"\n📊 Imported {len(imported_objects)} objects:")
    
    # Analyze each object
    mesh_objects = []
    for i, obj in enumerate(imported_objects):
        print(f"\n{i+1}. Object: '{obj.name}'")
        print(f"   Type: {obj.type}")
        print(f"   Location: ({obj.location.x:.3f}, {obj.location.y:.3f}, {obj.location.z:.3f})")
        print(f"   Rotation: ({obj.rotation_euler.x:.3f}, {obj.rotation_euler.y:.3f}, {obj.rotation_euler.z:.3f})")
        print(f"   Scale: ({obj.scale.x:.3f}, {obj.scale.y:.3f}, {obj.scale.z:.3f})")
        
        if obj.type == 'MESH':
            mesh_objects.append(obj)
            mesh = obj.data
            print(f"   Vertices: {len(mesh.vertices)}")
            print(f"   Faces: {len(mesh.polygons)}")
            print(f"   Materials: {len(obj.material_slots)}")
            
            # Analyze bounding box
            bbox = obj.bound_box
            width = max(v[0] for v in bbox) - min(v[0] for v in bbox)
            height = max(v[2] for v in bbox) - min(v[2] for v in bbox)
            depth = max(v[1] for v in bbox) - min(v[1] for v in bbox)
            
            print(f"   Dimensions: W={width:.3f} x D={depth:.3f} x H={height:.3f}")
            
            # World space bounds
            world_verts = [obj.matrix_world @ Vector(v) for v in bbox]
            world_min = Vector((
                min(v.x for v in world_verts),
                min(v.y for v in world_verts),
                min(v.z for v in world_verts)
            ))
            world_max = Vector((
                max(v.x for v in world_verts),
                max(v.y for v in world_verts),
                max(v.z for v in world_verts)
            ))
            
            print(f"   World bounds: Min={world_min}, Max={world_max}")
            
            # List materials
            if obj.material_slots:
                print(f"   Material slots:")
                for j, slot in enumerate(obj.material_slots):
                    if slot.material:
                        print(f"     {j}: {slot.material.name}")
        
        elif obj.type == 'EMPTY':
            print(f"   Empty type: {obj.empty_display_type}")
        elif obj.type == 'LIGHT':
            print(f"   Light type: {obj.data.type}")
        elif obj.type == 'CAMERA':
            print(f"   Camera lens: {obj.data.lens}mm")
    
    print(f"\n📊 Summary:")
    print(f"   Total objects: {len(imported_objects)}")
    print(f"   Mesh objects: {len(mesh_objects)}")
    
    # Find the actual can
    print("\n🔍 Identifying the can object...")
    
    if len(mesh_objects) == 0:
        print("❌ No mesh objects found!")
        return
    elif len(mesh_objects) == 1:
        can_object = mesh_objects[0]
        print(f"✅ Single mesh found: '{can_object.name}'")
    else:
        # Multiple meshes - find the largest or most likely can
        print(f"⚠️  Multiple meshes found, analyzing...")
        
        # Sort by vertex count (likely the can has the most geometry)
        mesh_objects.sort(key=lambda obj: len(obj.data.vertices), reverse=True)
        
        print("\nMesh objects by vertex count:")
        for obj in mesh_objects:
            print(f"  - {obj.name}: {len(obj.data.vertices)} vertices")
        
        can_object = mesh_objects[0]
        print(f"\n✅ Assuming largest mesh is the can: '{can_object.name}'")
    
    # Test render setup
    print("\n🎬 Setting up test render...")
    
    # Setup camera
    bpy.ops.object.camera_add(location=(0, -3, 1))
    camera = bpy.context.object
    camera.rotation_euler = (1.4, 0, 0)  # About 80 degrees
    bpy.context.scene.camera = camera
    
    # Point camera at can
    if can_object:
        # Get can center
        bbox = can_object.bound_box
        center = Vector((0, 0, 0))
        for v in bbox:
            center += Vector(v)
        center /= len(bbox)
        center = can_object.matrix_world @ center
        
        print(f"\n📍 Can center (world space): {center}")
        
        # Point camera at center
        direction = center - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
    
    # Add light
    bpy.ops.object.light_add(type='SUN', location=(2, -2, 3))
    sun = bpy.context.object
    sun.data.energy = 3.0
    
    # Render settings
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    
    # Save diagnostic info
    diag_file = Path("model_diagnostics.txt")
    with open(diag_file, 'w') as f:
        f.write("COCA-COLA CAN MODEL DIAGNOSTICS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Model file: {fbx_path}\n")
        f.write(f"Total objects: {len(imported_objects)}\n")
        f.write(f"Mesh objects: {len(mesh_objects)}\n\n")
        
        f.write("Object details:\n")
        for obj in imported_objects:
            f.write(f"\n- {obj.name} ({obj.type})\n")
            if obj.type == 'MESH':
                f.write(f"  Vertices: {len(obj.data.vertices)}\n")
                f.write(f"  Materials: {len(obj.material_slots)}\n")
                bbox = obj.bound_box
                width = max(v[0] for v in bbox) - min(v[0] for v in bbox)
                height = max(v[2] for v in bbox) - min(v[2] for v in bbox)
                f.write(f"  Size: {width:.3f} x {height:.3f}\n")
    
    print(f"\n📄 Diagnostics saved to: {diag_file}")
    
    # Do a test render
    print("\n📸 Creating diagnostic render...")
    render_path = Path("diagnostic_render.png")
    scene.render.filepath = str(render_path)
    bpy.ops.render.render(write_still=True)
    print(f"✅ Diagnostic render saved to: {render_path}")
    
    print("\n" + "="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60)
    print("\nCheck diagnostic_render.png to see what the camera sees")
    print("Check model_diagnostics.txt for detailed information")

if __name__ == "__main__":
    diagnose_model()
