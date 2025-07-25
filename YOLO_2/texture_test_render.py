
import bpy
import os

def setup_materials(obj):
    """Explicitly set up PBR materials for the given object."""
    print("Setting up materials...")
    
    # Define texture paths
    textures_dir = '/home/janga/YOLO_2/textures'
    texture_files = {
        "Label": {
            "color": "Label_Base_color.png",
            "metallic": "Label_Metallic.png",
            "roughness": "Label_Roughness.png",
            "normal": "Label_Normal_OpenGL.png"
        },
        "lambert1": {
            "color": "lambert1_Base_color.png",
            "metallic": "lambert1_Metallic.png",
            "roughness": "lambert1_Roughness.png",
            "normal": "lambert1_Normal_OpenGL.png"
        }
    }

    for mat_slot in obj.material_slots:
        material = mat_slot.material
        if not material:
            continue

        print(f"Processing material: {material.name}")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        # Create core nodes
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

        # Find texture set for this material
        tex_set_name = "Label" if "label" in material.name.lower() else "lambert1"
        tex_set = texture_files.get(tex_set_name)
        if not tex_set:
            print(f"  - No texture set found for {material.name}")
            continue

        # Create and connect texture nodes
        uv_node = nodes.new(type='ShaderNodeTexCoord')

        # Base Color
        color_path = os.path.join(textures_dir, tex_set["color"])
        if os.path.exists(color_path):
            color_tex = nodes.new(type='ShaderNodeTexImage')
            color_tex.image = bpy.data.images.load(color_path)
            links.new(uv_node.outputs['UV'], color_tex.inputs['Vector'])
            links.new(color_tex.outputs['Color'], bsdf_node.inputs['Base Color'])
            print(f"  - Linked Base Color: {tex_set['color']}")

        # Metallic
        metallic_path = os.path.join(textures_dir, tex_set["metallic"])
        if os.path.exists(metallic_path):
            metallic_tex = nodes.new(type='ShaderNodeTexImage')
            metallic_tex.image = bpy.data.images.load(metallic_path)
            metallic_tex.image.colorspace_settings.name = 'Non-Color'
            links.new(uv_node.outputs['UV'], metallic_tex.inputs['Vector'])
            links.new(metallic_tex.outputs['Color'], bsdf_node.inputs['Metallic'])
            print(f"  - Linked Metallic: {tex_set['metallic']}")

        # Roughness
        roughness_path = os.path.join(textures_dir, tex_set["roughness"])
        if os.path.exists(roughness_path):
            roughness_tex = nodes.new(type='ShaderNodeTexImage')
            roughness_tex.image = bpy.data.images.load(roughness_path)
            roughness_tex.image.colorspace_settings.name = 'Non-Color'
            links.new(uv_node.outputs['UV'], roughness_tex.inputs['Vector'])
            links.new(roughness_tex.outputs['Color'], bsdf_node.inputs['Roughness'])
            print(f"  - Linked Roughness: {tex_set['roughness']}")

        # Normal
        normal_path = os.path.join(textures_dir, tex_set["normal"])
        if os.path.exists(normal_path):
            normal_tex = nodes.new(type='ShaderNodeTexImage')
            normal_tex.image = bpy.data.images.load(normal_path)
            normal_tex.image.colorspace_settings.name = 'Non-Color'
            normal_map_node = nodes.new(type='ShaderNodeNormalMap')
            links.new(uv_node.outputs['UV'], normal_tex.inputs['Vector'])
            links.new(normal_tex.outputs['Color'], normal_map_node.inputs['Color'])
            links.new(normal_map_node.outputs['Normal'], bsdf_node.inputs['Normal'])
            print(f"  - Linked Normal: {tex_set['normal']}")


def main():
    """Main function to run the rendering test."""
    # Clear the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Load the FBX model
    fbx_path = '/home/janga/YOLO_2/source/Coke Can.fbx'
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # Get the imported object (assuming it's the first mesh object)
    can_obj = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            can_obj = obj
            break
    
    if not can_obj:
        print("Error: No mesh object found in FBX.")
        return

    # Set up materials
    setup_materials(can_obj)

    # Set up a simple camera
    bpy.ops.object.camera_add(location=(0, -2.5, 1.0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera

    # Point the camera at the object
    direction = can_obj.location - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    # Add a simple light
    bpy.ops.object.light_add(type='SUN', location=(2, -2, 2))
    bpy.context.object.data.energy = 5

    # Set render settings
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.resolution_x = 800
    bpy.context.scene.render.resolution_y = 800
    bpy.context.scene.render.filepath = '/home/janga/YOLO_2/texture_test_render.png'

    # Render the image
    print("Starting render...")
    bpy.ops.render.render(write_still=True)
    print("Render complete. Image saved to texture_test_render.png")

if __name__ == "__main__":
    main()
