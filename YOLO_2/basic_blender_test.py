
import bpy

def main():
    """A minimal script to test if Blender rendering is functional."""
    # Clear the scene
    bpy.ops.object.select_all(action='SELECT')
    if bpy.context.selected_objects:
        bpy.ops.object.delete()

    # Create a simple cylinder to represent the can
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.3, depth=1.2, location=(0, 0, 0.6))
    obj = bpy.context.active_object

    # Create a simple, bright red material
    mat = bpy.data.materials.new(name="TestMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (1.0, 0.0, 0.0, 1.0) # Red
        bsdf.inputs['Metallic'].default_value = 0.8
        bsdf.inputs['Roughness'].default_value = 0.3
    obj.data.materials.append(mat)

    # Add a camera and point it at the object
    bpy.ops.object.camera_add(location=(0, -3, 1.5))
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera
    direction = obj.location - camera.location
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
    scene.render.filepath = '/home/janga/YOLO_2/basic_render.png'

    # Render the image
    print("--- Starting Basic Blender Render Test ---")
    bpy.ops.render.render(write_still=True)
    print("--- Basic Render Test Complete ---")

if __name__ == "__main__":
    main()
