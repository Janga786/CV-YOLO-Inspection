import bpy

# Clear the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Load the FBX model
fbx_path = '/home/janga/YOLO_2/source/Coke Can.fbx'
bpy.ops.import_scene.fbx(filepath=fbx_path)

# Set up a simple camera
bpy.ops.object.camera_add(location=(0, -3, 1))
camera = bpy.context.object
camera.data.lens = 50
bpy.context.scene.camera = camera

# Point the camera at the object
obj = bpy.data.objects[0]
direction = obj.location - camera.location
rot_quat = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rot_quat.to_euler()

# Add a simple light
bpy.ops.object.light_add(type='SUN', location=(2, -2, 2))
sun = bpy.context.object
sun.data.energy = 5

# Set the render path and render the image
bpy.context.scene.render.filepath = '/home/janga/YOLO_2/simple_render.png'
bpy.ops.render.render(write_still=True)
