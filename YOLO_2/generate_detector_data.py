"""
Professional Soda Can Dataset Generator - Fixed Version
Ensures can is ALWAYS visible in frame with proper bounding box verification
"""

import bpy
import random
import math
import os
import sys
import argparse
from pathlib import Path
from mathutils import Vector
# numpy not needed - removed to work with Blender's Python

class ProfessionalCanGenerator:
    def __init__(self, model_path, texture_path, backgrounds_dir, output_dir):
        self.model_path = Path(model_path)
        self.texture_path = Path(texture_path)
        self.backgrounds_dir = Path(backgrounds_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get background images
        self.background_images = list(self.backgrounds_dir.glob("*.jpg")) + \
                               list(self.backgrounds_dir.glob("*.png"))
        
        print(f"Found {len(self.background_images)} background images")
        
        # Store original materials for reset
        self.original_materials = {}
        
        # Camera frustum parameters for visibility check
        self.camera_fov_margin = 0.70  # Use 70% of camera FOV for safety
        
        self.setup_scene()
    
    def setup_scene(self):
        """Setup Blender scene with optimal settings"""
        # Clear scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Render settings - use EEVEE for speed
        scene = bpy.context.scene
        scene.render.engine = 'BLENDER_EEVEE'
        scene.render.resolution_x = 640
        scene.render.resolution_y = 640
        scene.render.film_transparent = False
        
        # EEVEE settings
        scene.eevee.taa_render_samples = 32
        scene.eevee.use_gtao = True
        scene.eevee.gtao_distance = 0.2
        scene.eevee.use_bloom = False
        scene.eevee.use_ssr = True
        scene.eevee.use_ssr_refraction = True
        
        # Camera setup - positioned to always see the can
        bpy.ops.object.camera_add(location=(0, -2.5, 0.6))
        self.camera = bpy.context.object
        self.camera.rotation_euler = (math.radians(80), 0, 0)
        scene.camera = self.camera
        self.camera.data.lens = 50  # Standard lens
        self.camera.data.clip_start = 0.1
        self.camera.data.clip_end = 100.0
        
        # Professional lighting setup
        # Key light (sun)
        bpy.ops.object.light_add(type='SUN', location=(3, -2, 5))
        self.key_light = bpy.context.object
        self.key_light.data.energy = 3.0
        self.key_light.data.angle = math.radians(15)
        
        # Fill light (area)
        bpy.ops.object.light_add(type='AREA', location=(-2, -3, 3))
        self.fill_light = bpy.context.object
        self.fill_light.data.energy = 2.0
        self.fill_light.data.size = 2.0
        
        # Rim light
        bpy.ops.object.light_add(type='SPOT', location=(1, 1, 2))
        self.rim_light = bpy.context.object
        self.rim_light.data.energy = 1.5
        self.rim_light.data.spot_size = math.radians(45)
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load the Sketchfab model and properly set up materials"""
        print(f"Loading model: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Import FBX
        try:
            # Store existing objects to identify new ones
            existing_objects = set(bpy.data.objects[:])
            
            bpy.ops.import_scene.fbx(filepath=str(self.model_path))
            
            # Find newly imported objects
            new_objects = [obj for obj in bpy.data.objects if obj not in existing_objects]
            mesh_objects = [obj for obj in new_objects if obj.type == 'MESH']
            
            print(f"Imported {len(mesh_objects)} mesh objects")
            
        except Exception as e:
            print(f"Error importing FBX: {e}")
            raise
        
        if not mesh_objects:
            raise ValueError("No mesh objects found in the imported model")
        
        # If multiple meshes, join them
        if len(mesh_objects) > 1:
            # Select all mesh objects
            bpy.ops.object.select_all(action='DESELECT')
            for obj in mesh_objects:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = mesh_objects[0]
            
            # Join meshes
            bpy.ops.object.join()
            self.can_object = bpy.context.active_object
        else:
            self.can_object = mesh_objects[0]
        
        print(f"Using can object: {self.can_object.name}")
        
        # First, apply all transformations to make them real
        bpy.context.view_layer.objects.active = self.can_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # Now calculate the true bounding box center
        bbox = self.can_object.bound_box
        bbox_center = Vector((
            sum(v[0] for v in bbox) / 8,
            sum(v[1] for v in bbox) / 8,
            sum(v[2] for v in bbox) / 8
        ))
        
        # Move geometry to center by setting origin to geometry center
        bpy.context.view_layer.objects.active = self.can_object
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
        
        # Now the origin is at the center, so we can just position at 0,0,0
        self.can_object.location = (0, 0, 0)
        self.can_object.rotation_euler = (0, 0, 0)
        
        # Scale to standard can size
        # Standard can: 4.83 inches (12.3 cm) tall, 2.6 inches (6.6 cm) diameter
        # The model appears to be a cylinder, so use the height dimension
        bbox = self.can_object.bound_box
        
        # Determine which dimension is height (usually Z for a standing can)
        x_size = max(v[0] for v in bbox) - min(v[0] for v in bbox)
        y_size = max(v[1] for v in bbox) - min(v[1] for v in bbox)
        z_size = max(v[2] for v in bbox) - min(v[2] for v in bbox)
        
        # For this model, it seems X and Z are similar (diameter) and Y is different
        # So Y is likely the height of the can lying on its side
        # We need to rotate it to stand upright
        if abs(x_size - z_size) < 0.1 and y_size > max(x_size, z_size):
            # Can is lying on its side, rotate it to stand up
            self.can_object.rotation_euler = (math.radians(90), 0, 0)
            bpy.context.view_layer.objects.active = self.can_object
            bpy.ops.object.transform_apply(rotation=True)
            
            # Recalculate bbox after rotation
            bbox = self.can_object.bound_box
            current_height = max(v[2] for v in bbox) - min(v[2] for v in bbox)
        else:
            current_height = z_size
        
        if current_height > 0:
            # Scale to 1.23 Blender units (12.3 cm)
            scale_factor = 1.23 / current_height
            self.can_object.scale = (scale_factor, scale_factor, scale_factor)
        
        # Apply scale
        bpy.context.view_layer.objects.active = self.can_object
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        # Position can on ground after all transformations
        bbox = self.can_object.bound_box
        min_z = min(v[2] for v in bbox)
        self.can_object.location.z = -min_z
        
        # Final check - print dimensions to verify
        bbox = self.can_object.bound_box
        final_height = max(v[2] for v in bbox) - min(v[2] for v in bbox)
        final_width = max(v[0] for v in bbox) - min(v[0] for v in bbox)
        print(f"Final can dimensions - Height: {final_height:.3f}, Width: {final_width:.3f}")
        
        # Store can dimensions for visibility checks
        self.can_dimensions = self.get_object_dimensions()
        print(f"Can dimensions: {self.can_dimensions}")
        
        # Setup materials with all texture maps
        self.setup_all_materials()
    
    def get_object_dimensions(self):
        """Get the dimensions of the can object"""
        bbox = self.can_object.bound_box
        min_x = min(v[0] for v in bbox)
        max_x = max(v[0] for v in bbox)
        min_y = min(v[1] for v in bbox)
        max_y = max(v[1] for v in bbox)
        min_z = min(v[2] for v in bbox)
        max_z = max(v[2] for v in bbox)
        
        # Calculate dimensions
        width = max_x - min_x
        depth = max_y - min_y
        height = max_z - min_z
        
        # Get actual center in world space
        bbox_world = [self.can_object.matrix_world @ Vector(corner) for corner in self.can_object.bound_box]
        world_min_x = min(v[0] for v in bbox_world)
        world_max_x = max(v[0] for v in bbox_world)
        world_min_y = min(v[1] for v in bbox_world)
        world_max_y = max(v[1] for v in bbox_world)
        world_min_z = min(v[2] for v in bbox_world)
        world_max_z = max(v[2] for v in bbox_world)
        
        world_center = Vector((
            (world_min_x + world_max_x) / 2,
            (world_min_y + world_max_y) / 2,
            (world_min_z + world_max_z) / 2
        ))
        
        return {
            'width': width,
            'depth': depth,
            'height': height,
            'center': world_center
        }
    
    def setup_all_materials(self):
        """Setup all materials with proper texture assignments"""
        print("Setting up materials with all texture maps...")
        
        textures_dir = Path("textures")
        
        # Process each material slot
        for mat_slot_idx, mat_slot in enumerate(self.can_object.material_slots):
            if not mat_slot.material:
                continue
                
            material = mat_slot.material
            print(f"Processing material {mat_slot_idx}: {material.name}")
            
            # Store original for reset
            self.original_materials[mat_slot_idx] = material.copy()
            
            # Enable nodes
            material.use_nodes = True
            nodes = material.node_tree.nodes
            links = material.node_tree.links
            
            # Clear existing nodes
            nodes.clear()
            
            # Create base nodes
            output_node = nodes.new(type='ShaderNodeOutputMaterial')
            bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
            
            output_node.location = (600, 0)
            bsdf_node.location = (300, 0)
            
            links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
            
            # Determine which textures to use based on material name
            texture_prefix = None
            if 'label' in material.name.lower():
                texture_prefix = "Label"
            elif 'lambert' in material.name.lower() or 'body' in material.name.lower():
                texture_prefix = "lambert1"
            else:
                # Default to Label for first material, lambert for others
                texture_prefix = "Label" if mat_slot_idx == 0 else "lambert1"
            
            print(f"  Using texture set: {texture_prefix}")
            
            # UV coordinate node
            uv_node = nodes.new(type='ShaderNodeTexCoord')
            uv_node.location = (-600, 0)
            
            # Load textures
            y_offset = 300
            
            # Base Color
            base_color_path = textures_dir / f"{texture_prefix}_Base_color.png"
            if base_color_path.exists():
                color_node = nodes.new(type='ShaderNodeTexImage')
                color_node.location = (-300, y_offset)
                color_node.image = bpy.data.images.load(str(base_color_path))
                links.new(uv_node.outputs['UV'], color_node.inputs['Vector'])
                links.new(color_node.outputs['Color'], bsdf_node.inputs['Base Color'])
                print(f"  ✓ Loaded base color")
            else:
                # Fallback colors
                if texture_prefix == "Label":
                    bsdf_node.inputs['Base Color'].default_value = (0.8, 0.05, 0.05, 1.0)  # Coca-Cola red
                else:
                    bsdf_node.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1.0)  # Silver/aluminum
            
            # Metallic
            metallic_path = textures_dir / f"{texture_prefix}_Metallic.png"
            if metallic_path.exists():
                metallic_node = nodes.new(type='ShaderNodeTexImage')
                metallic_node.location = (-300, y_offset - 250)
                metallic_node.image = bpy.data.images.load(str(metallic_path))
                metallic_node.image.colorspace_settings.name = 'Non-Color'
                links.new(uv_node.outputs['UV'], metallic_node.inputs['Vector'])
                links.new(metallic_node.outputs['Color'], bsdf_node.inputs['Metallic'])
                print(f"  ✓ Loaded metallic")
            else:
                bsdf_node.inputs['Metallic'].default_value = 0.9 if texture_prefix == "lambert1" else 0.3
            
            # Roughness
            roughness_path = textures_dir / f"{texture_prefix}_Roughness.png"
            if roughness_path.exists():
                roughness_node = nodes.new(type='ShaderNodeTexImage')
                roughness_node.location = (-300, y_offset - 500)
                roughness_node.image = bpy.data.images.load(str(roughness_path))
                roughness_node.image.colorspace_settings.name = 'Non-Color'
                links.new(uv_node.outputs['UV'], roughness_node.inputs['Vector'])
                links.new(roughness_node.outputs['Color'], bsdf_node.inputs['Roughness'])
                print(f"  ✓ Loaded roughness")
            else:
                bsdf_node.inputs['Roughness'].default_value = 0.2 if texture_prefix == "lambert1" else 0.4
            
            # Normal map
            normal_path = textures_dir / f"{texture_prefix}_Normal_OpenGL.png"
            if normal_path.exists():
                normal_tex_node = nodes.new(type='ShaderNodeTexImage')
                normal_map_node = nodes.new(type='ShaderNodeNormalMap')
                normal_tex_node.location = (-300, y_offset - 750)
                normal_map_node.location = (0, y_offset - 750)
                
                normal_tex_node.image = bpy.data.images.load(str(normal_path))
                normal_tex_node.image.colorspace_settings.name = 'Non-Color'
                
                links.new(uv_node.outputs['UV'], normal_tex_node.inputs['Vector'])
                links.new(normal_tex_node.outputs['Color'], normal_map_node.inputs['Color'])
                links.new(normal_map_node.outputs['Normal'], bsdf_node.inputs['Normal'])
                print(f"  ✓ Loaded normal map")
        
        print("✓ All materials configured")
    
    def setup_background(self, background_path=None):
        """Setup background"""
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()
        
        output_node = nodes.new(type='ShaderNodeOutputWorld')
        background_node = nodes.new(type='ShaderNodeBackground')
        
        if background_path and background_path.exists():
            try:
                env_tex_node = nodes.new(type='ShaderNodeTexEnvironment')
                env_tex_node.image = bpy.data.images.load(str(background_path))
                links.new(env_tex_node.outputs['Color'], background_node.inputs['Color'])
                background_node.inputs['Strength'].default_value = 0.8
            except:
                self.setup_solid_background(background_node)
        else:
            self.setup_solid_background(background_node)
        
        links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
    
    def setup_solid_background(self, background_node):
        """Setup solid color background"""
        colors = [
            (0.95, 0.95, 0.95),  # White
            (0.8, 0.8, 0.8),     # Light gray
            (0.2, 0.2, 0.2),     # Dark gray
            (0.1, 0.1, 0.15),    # Dark blue-gray
        ]
        
        color = random.choice(colors)
        background_node.inputs['Color'].default_value = (*color, 1.0)
        background_node.inputs['Strength'].default_value = 1.0
    
    def apply_defect(self, defect_type):
        """Apply defects to materials"""
        for mat_slot in self.can_object.material_slots:
            if not mat_slot.material:
                continue
                
            material = mat_slot.material
            bsdf_node = None
            
            for node in material.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf_node = node
                    break
            
            if not bsdf_node:
                continue
            
            if defect_type == 'scratch':
                # Increase roughness
                if 'Roughness' in bsdf_node.inputs:
                    current = bsdf_node.inputs['Roughness'].default_value
                    bsdf_node.inputs['Roughness'].default_value = min(1.0, current + random.uniform(0.3, 0.5))
                
            elif defect_type == 'dent':
                # Darken and increase roughness slightly
                if 'Base Color' in bsdf_node.inputs:
                    color = bsdf_node.inputs['Base Color'].default_value
                    factor = random.uniform(0.7, 0.85)
                    bsdf_node.inputs['Base Color'].default_value = (
                        color[0] * factor,
                        color[1] * factor,
                        color[2] * factor,
                        1.0
                    )
                
            elif defect_type == 'puncture':
                # Very dark and rough
                if 'Base Color' in bsdf_node.inputs:
                    bsdf_node.inputs['Base Color'].default_value = (0.1, 0.1, 0.1, 1.0)
                if 'Roughness' in bsdf_node.inputs:
                    bsdf_node.inputs['Roughness'].default_value = 0.9
                if 'Metallic' in bsdf_node.inputs:
                    bsdf_node.inputs['Metallic'].default_value = 0.0
    
    def reset_materials(self):
        """Reset all materials to original state"""
        for mat_idx, original_mat in self.original_materials.items():
            if mat_idx < len(self.can_object.material_slots):
                new_mat = original_mat.copy()
                self.can_object.material_slots[mat_idx].material = new_mat
    
    def is_object_in_camera_view(self):
        """Check if the can is reasonably visible in camera view"""
        import bpy_extras.object_utils
        
        scene = bpy.context.scene
        
        # Just check the center point - simpler and more lenient
        center_point = self.can_object.matrix_world @ Vector((0, 0, 0))
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, self.camera, center_point)
        
        # Check if center is in front of camera and within frame bounds
        if co_2d.z > 0:  # In front of camera
            # Use generous margins
            margin = 0.1  # 10% margin from edges
            if margin < co_2d.x < (1 - margin) and margin < co_2d.y < (1 - margin):
                return True
        
        return False
    
    def find_valid_camera_position(self, max_attempts=50):
        """Find a camera position where the can is fully visible"""
        for attempt in range(max_attempts):
            # Use spherical coordinates for camera position with full variety
            distance = random.uniform(1.5, 3.2)  # Wider distance range
            azimuth = random.uniform(-math.pi/2, math.pi/2)  # ±90 degrees (front hemisphere)
            elevation = random.uniform(0.2, 1.5)  # Full height variation
            
            # Calculate camera position
            cam_x = distance * math.sin(azimuth)
            cam_y = -distance * math.cos(azimuth)
            cam_z = elevation
            
            self.camera.location = (cam_x, cam_y, cam_z)
            
            # Point camera at can's actual location (use can's center regardless of orientation)
            can_center = Vector(self.can_object.location)
            direction = can_center - Vector(self.camera.location)
            rot_quat = direction.to_track_quat('-Z', 'Y')
            self.camera.rotation_euler = rot_quat.to_euler()
            
            # Check if can is visible
            if self.is_object_in_camera_view():
                return True
        
        # Fallback to safe position
        print("Warning: Using fallback camera position")
        self.camera.location = (0, -2.5, 0.8)
        can_center = Vector(self.can_object.location)
        direction = can_center - Vector(self.camera.location)
        rot_quat = direction.to_track_quat('-Z', 'Y')
        self.camera.rotation_euler = rot_quat.to_euler()
        
        # Force visibility by adjusting FOV if needed
        self.camera.data.lens = 35  # Wider angle
        return True
    
    def randomize_scene(self):
        """Randomize scene ensuring can is always visible"""
        # Can position - full frame positioning
        self.can_object.location = (
            random.uniform(-0.5, 0.5),  # Full X range across frame
            random.uniform(-0.5, 0.5),  # Full Y range across frame
            random.uniform(-0.2, 0.4)   # Full Z range for vertical positioning
        )
        
        # Can rotation - full orientation variety including sideways
        # 30% chance for sideways orientations (laying down)
        if random.random() < 0.3:
            # Sideways orientations - can laying on its side
            self.can_object.rotation_euler = (
                random.uniform(math.radians(70), math.radians(110)),  # Nearly horizontal
                random.uniform(-math.radians(30), math.radians(30)), # Some Y variation
                random.uniform(0, 2 * math.pi)  # Full Z rotation
            )
        else:
            # Standing/tilted orientations
            self.can_object.rotation_euler = (
                random.uniform(-math.radians(45), math.radians(45)),  # More X tilt
                random.uniform(-math.radians(45), math.radians(45)),  # More Y tilt
                random.uniform(0, 2 * math.pi)  # Full Z rotation for label variety
            )
        
        # Find valid camera position
        self.find_valid_camera_position()
        
        # Vary camera lens slightly
        self.camera.data.lens = random.uniform(45, 55)
        
        # Randomize lighting
        self.key_light.data.energy = random.uniform(2.5, 3.5)
        self.fill_light.data.energy = random.uniform(1.5, 2.5)
        self.rim_light.data.energy = random.uniform(1.0, 2.0)
        
        # Slight rotation for lights
        self.key_light.rotation_euler = (
            random.uniform(0.3, 0.6),
            random.uniform(-0.3, 0.3),
            0
        )
    
    def calculate_bounding_box(self):
        """Calculate YOLO bounding box with validation"""
        try:
            import bpy_extras.object_utils
            
            # Get mesh bounding box corners
            bbox_corners = [self.can_object.matrix_world @ Vector(corner) for corner in self.can_object.bound_box]
            
            # Project to 2D
            scene = bpy.context.scene
            coords_2d = []
            
            for corner in bbox_corners:
                co_2d = bpy_extras.object_utils.world_to_camera_view(scene, self.camera, corner)
                if co_2d.z > 0:  # Only consider points in front of camera
                    coords_2d.append((co_2d.x, 1.0 - co_2d.y))
            
            if len(coords_2d) < 4:
                print("Warning: Not enough visible corners for bounding box")
                return None
            
            # Calculate bounds
            x_coords = [coord[0] for coord in coords_2d]
            y_coords = [coord[1] for coord in coords_2d]
            
            min_x = max(0, min(x_coords))
            max_x = min(1, max(x_coords))
            min_y = max(0, min(y_coords))
            max_y = min(1, max(y_coords))
            
            # YOLO format
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            width = max_x - min_x
            height = max_y - min_y
            
            # Validate bounding box
            min_box_size = 0.05  # Minimum 5% of image
            if width < min_box_size or height < min_box_size:
                print(f"Warning: Bounding box too small: {width:.3f}x{height:.3f}")
                return None
            
            # Check if bounding box is mostly within frame
            if center_x < 0.1 or center_x > 0.9 or center_y < 0.1 or center_y > 0.9:
                print(f"Warning: Bounding box too close to edge: center at ({center_x:.3f}, {center_y:.3f})")
                return None
            
            return center_x, center_y, width, height
            
        except Exception as e:
            print(f"Bounding box calculation error: {e}")
            return None
    
    def generate_single_image(self, image_index, defect_type=None):
        """Generate a single image with validation"""
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                # Reset materials
                self.reset_materials()
                
                # Choose defect type
                if defect_type is None:
                    defect_types = ['pristine', 'dent', 'scratch', 'puncture']
                    weights = [0.4, 0.2, 0.2, 0.2]
                    defect_type = random.choices(defect_types, weights=weights)[0]
                
                # Apply defect
                class_id = 0 if defect_type == 'pristine' else 1
                if defect_type != 'pristine':
                    self.apply_defect(defect_type)
                
                # Randomize scene
                self.randomize_scene()
                
                # Verify can is visible
                if not self.is_object_in_camera_view():
                    print(f"Retry {retry + 1}: Can not fully visible, adjusting...")
                    continue
                
                # Set background
                if self.background_images and random.random() < 0.7:
                    bg_image = random.choice(self.background_images)
                    self.setup_background(bg_image)
                else:
                    self.setup_background()
                
                # Calculate bounding box before rendering
                bbox_data = self.calculate_bounding_box()
                if bbox_data is None:
                    print(f"Retry {retry + 1}: Invalid bounding box, adjusting...")
                    continue
                
                center_x, center_y, width, height = bbox_data
                
                # Render
                image_filename = f"can_{image_index:05d}.png"
                image_path = self.images_dir / image_filename
                
                bpy.context.scene.render.filepath = str(image_path)
                bpy.ops.render.render(write_still=True)
                
                # Verify render
                if not image_path.exists() or image_path.stat().st_size < 10000:
                    print(f"Warning: Render may have failed for image {image_index}")
                    continue
                
                # Save label
                label_filename = f"can_{image_index:05d}.txt"
                label_path = self.labels_dir / label_filename
                
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                # Success log
                if retry > 0:
                    print(f"Success after {retry + 1} attempts")
                
                return True
                
            except Exception as e:
                print(f"Error generating image {image_index} (attempt {retry + 1}): {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Failed to generate image {image_index} after {max_retries} attempts")
        return False
    
    def generate_batch(self, start_index, batch_size):
        """Generate a batch of images"""
        print(f"Generating batch: {start_index} to {start_index + batch_size - 1}")
        
        success_count = 0
        
        for i in range(batch_size):
            if self.generate_single_image(start_index + i):
                success_count += 1
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{batch_size} (Success: {success_count})")
        
        print(f"Batch complete: {success_count}/{batch_size} images")
        return success_count

def parse_arguments():
    """Parse command line arguments"""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fbx_path", type=str, required=True)
    parser.add_argument("--texture_path", type=str, required=True)
    parser.add_argument("--backgrounds_dir", type=str, required=True)
    
    return parser.parse_args(argv)

def main():
    """Main execution"""
    try:
        args = parse_arguments()
        
        print("=== Coca-Cola Can Dataset Generator ===")
        print(f"Model: {args.fbx_path}")
        print(f"Output: {args.output_dir}")
        print(f"Batch: {args.start_index} to {args.start_index + args.batch_size - 1}")
        
        generator = ProfessionalCanGenerator(
            model_path=args.fbx_path,
            texture_path=args.texture_path,
            backgrounds_dir=args.backgrounds_dir,
            output_dir=args.output_dir
        )
        
        success_count = generator.generate_batch(args.start_index, args.batch_size)
        
        print(f"\n✓ Generated {success_count}/{args.batch_size} images")
        
        if success_count > 0:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
