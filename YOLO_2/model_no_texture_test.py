
import bpy

def main():
    """Render the FBX model with a simple material, no textures."""
    # Clear the scene
    bpy.ops.object.select_all(action='SELECT')
    if bpy.context.selected_objects:
        bpy.ops.object.delete()

    # Load the FBX model
    fbx_path = '/home/janga/YOLO_2/source/Coke Can.fbx'
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # Find the imported mesh object
    can_obj = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            can_obj = obj
            break
    
    if not can_obj:
        print("ERROR: No mesh object found after importing FBX.")
        return

    # Remove existing materials
    can_obj.data.materials.clear()

    # Create and apply a simple red material
    mat = bpy.data.materials.new(name="TestMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (1.0, 0.0, 0.0, 1.0) # Red
        bsdf.inputs['Metallic'].default_value = 0.8
        bsdf.inputs['Roughness'].default_value = 0.3
    can_obj.data.materials.append(mat)

    # Add a camera and point it at the object
    bpy.ops.object.camera_add(location=(0, -3, 1.5))
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera
    direction = can_obj.location - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    # Add a light source
    bpy.ops.object.light_add(type='SUN', location=(3, -3, 3))
    bpy.context.object.data.energy = 4

    # Set render settings
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.filepath = '/home/janga/YOLO_2/model_no_texture_render.png'

    # Render the image
    print("--- Rendering model with simple material ---")
    bpy.ops.render.render(write_still=True)
    print("--- Render complete ---")

if __name__ == "__main__":
    main()
