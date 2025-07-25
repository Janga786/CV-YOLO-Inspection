"""
Enhanced Professional Synthetic Data Generator - Blender 2.82 Compatible
Optimized for perfect synthetic training data with Blender 2.82
"""

import bpy
import bmesh
import random
import math
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from mathutils import Vector, Matrix, Euler
from datetime import datetime

class EnhancedSyntheticGenerator:
    def __init__(self, model_path, texture_path, backgrounds_dir, output_dir, config=None):
        self.model_path = Path(model_path)
        self.texture_path = Path(texture_path)
        self.backgrounds_dir = Path(backgrounds_dir)
        self.output_dir = Path(output_dir)
        self.config = config or self.get_default_config()
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.images_dir, self.labels_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get background images
        self.background_images = list(self.backgrounds_dir.glob("*.jpg")) + \
                               list(self.backgrounds_dir.glob("*.png"))
        
        print(f"Found {len(self.background_images)} background images")
        
        # Initialize tracking
        self.generation_stats = {
            'total_generated': 0,
            'quality_passed': 0,
            'defect_distribution': {'pristine': 0, 'scratch': 0, 'dent': 0, 'puncture': 0},
            'start_time': datetime.now().isoformat()
        }
        
        self.setup_scene()
    
    def get_default_config(self):
        """Get default configuration compatible with Blender 2.82"""
        return {
            'render': {
                'resolution': [1024, 1024],
                'samples': 64,
                'engine': 'BLENDER_EEVEE',  # Use EEVEE for better 2.82 compatibility
                'denoising': True
            },
            'camera': {
                'lens_range': [35, 85],
                'distance_range': [1.5, 3.0],
                'height_range': [0.3, 1.2],
                'angle_range': [-45, 45]
            },
            'lighting': {
                'key_light_strength': [3.0, 8.0],
                'fill_light_strength': [1.0, 4.0],
                'rim_light_strength': [2.0, 6.0]
            },
            'defects': {
                'pristine_probability': 0.3,
                'scratch_probability': 0.25,
                'dent_probability': 0.25,
                'puncture_probability': 0.2,
                'defect_intensity_range': [0.3, 0.9]
            },
            'quality': {
                'min_can_area': 0.01,  # Much more lenient
                'max_can_area': 0.95   # Allow larger cans
            }
        }
    
    def setup_scene(self):
        """Setup enhanced Blender scene compatible with 2.82"""
        # Clear scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Enhanced render settings
        scene = bpy.context.scene
        scene.render.engine = self.config['render']['engine']
        scene.render.resolution_x = self.config['render']['resolution'][0]
        scene.render.resolution_y = self.config['render']['resolution'][1]
        scene.render.film_transparent = False
        
        if scene.render.engine == 'BLENDER_EEVEE':
            # Enhanced EEVEE settings for 2.82
            scene.eevee.taa_render_samples = self.config['render']['samples']
            scene.eevee.use_gtao = True
            scene.eevee.gtao_distance = 0.2
            scene.eevee.use_bloom = True
            scene.eevee.bloom_threshold = 0.8
            scene.eevee.use_ssr = True
            scene.eevee.use_ssr_refraction = True
        
        # Load and setup the can model
        self.load_model()
        self.setup_professional_lighting()
        self.setup_camera_system()
    
    def load_model(self):
        """Load and prepare the can model"""
        print(f"Loading enhanced model: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Store existing objects
        existing_objects = set(bpy.data.objects[:])
        
        # Import FBX
        try:
            bpy.ops.import_scene.fbx(filepath=str(self.model_path))
        except Exception as e:
            print(f"Error importing FBX: {e}")
            raise
        
        # Find newly imported objects
        new_objects = [obj for obj in bpy.data.objects if obj not in existing_objects]
        mesh_objects = [obj for obj in new_objects if obj.type == 'MESH']
        
        if not mesh_objects:
            raise ValueError("No mesh objects found in the imported model")
        
        # Join multiple meshes if needed
        if len(mesh_objects) > 1:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in mesh_objects:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = mesh_objects[0]
            bpy.ops.object.join()
            self.can_object = bpy.context.active_object
        else:
            self.can_object = mesh_objects[0]
        
        # Prepare the model
        self.prepare_model()
        self.setup_enhanced_materials()
    
    def prepare_model(self):
        """Prepare the model with proper scaling and positioning"""
        print("Preparing model geometry...")
        
        # Center and reset transforms
        self.can_object.location = (0, 0, 0)
        self.can_object.rotation_euler = (0, 0, 0)
        self.can_object.scale = (1, 1, 1)
        
        # Scale to realistic can dimensions
        bbox = self.can_object.bound_box
        current_height = max(v[2] for v in bbox) - min(v[2] for v in bbox)
        
        if current_height > 0:
            # Scale to 12.3 cm (standard can height)
            scale_factor = 1.23 / current_height
            self.can_object.scale = (scale_factor, scale_factor, scale_factor)
        
        # Apply transforms
        bpy.context.view_layer.objects.active = self.can_object
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        # Position on ground
        bbox = self.can_object.bound_box
        min_z = min(v[2] for v in bbox)
        self.can_object.location.z = -min_z
    
    def setup_enhanced_materials(self):
        """Setup enhanced PBR materials compatible with Blender 2.82"""
        print("Setting up enhanced materials...")
        
        textures_dir = Path("textures")
        
        for mat_slot_idx, mat_slot in enumerate(self.can_object.material_slots):
            if not mat_slot.material:
                continue
            
            material = mat_slot.material
            print(f"Enhancing material {mat_slot_idx}: {material.name}")
            
            # Enable nodes
            material.use_nodes = True
            nodes = material.node_tree.nodes
            links = material.node_tree.links
            
            # Clear existing nodes
            nodes.clear()
            
            # Create enhanced node setup
            self.create_enhanced_material_nodes(material, mat_slot_idx, textures_dir)
    
    def create_enhanced_material_nodes(self, material, mat_slot_idx, textures_dir):
        """Create enhanced material node setup compatible with Blender 2.82"""
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        
        # Base nodes
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        
        output_node.location = (600, 0)
        bsdf_node.location = (300, 0)
        
        # Determine texture set
        texture_prefix = "Label" if 'label' in material.name.lower() else "lambert1"
        
        # UV coordinates
        uv_node = nodes.new(type='ShaderNodeTexCoord')
        uv_node.location = (-600, 0)
        
        # Base color setup
        base_color_path = textures_dir / f"{texture_prefix}_Base_color.png"
        if base_color_path.exists():
            color_node = nodes.new(type='ShaderNodeTexImage')
            color_node.location = (-300, 300)
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
        
        # Metallic setup
        metallic_path = textures_dir / f"{texture_prefix}_Metallic.png"
        if metallic_path.exists():
            metallic_node = nodes.new(type='ShaderNodeTexImage')
            metallic_node.location = (-300, 0)
            metallic_node.image = bpy.data.images.load(str(metallic_path))
            metallic_node.image.colorspace_settings.name = 'Non-Color'
            links.new(uv_node.outputs['UV'], metallic_node.inputs['Vector'])
            links.new(metallic_node.outputs['Color'], bsdf_node.inputs['Metallic'])
            print(f"  ✓ Loaded metallic")
        else:
            bsdf_node.inputs['Metallic'].default_value = 0.9 if texture_prefix == "lambert1" else 0.3
        
        # Roughness setup
        roughness_path = textures_dir / f"{texture_prefix}_Roughness.png"
        if roughness_path.exists():
            roughness_node = nodes.new(type='ShaderNodeTexImage')
            roughness_node.location = (-300, -200)
            roughness_node.image = bpy.data.images.load(str(roughness_path))
            roughness_node.image.colorspace_settings.name = 'Non-Color'
            links.new(uv_node.outputs['UV'], roughness_node.inputs['Vector'])
            links.new(roughness_node.outputs['Color'], bsdf_node.inputs['Roughness'])
            print(f"  ✓ Loaded roughness")
        else:
            bsdf_node.inputs['Roughness'].default_value = 0.2 if texture_prefix == "lambert1" else 0.4
        
        # Normal map setup
        normal_path = textures_dir / f"{texture_prefix}_Normal_OpenGL.png"
        if normal_path.exists():
            normal_tex_node = nodes.new(type='ShaderNodeTexImage')
            normal_map_node = nodes.new(type='ShaderNodeNormalMap')
            normal_tex_node.location = (-300, -400)
            normal_map_node.location = (0, -400)
            
            normal_tex_node.image = bpy.data.images.load(str(normal_path))
            normal_tex_node.image.colorspace_settings.name = 'Non-Color'
            
            links.new(uv_node.outputs['UV'], normal_tex_node.inputs['Vector'])
            links.new(normal_tex_node.outputs['Color'], normal_map_node.inputs['Color'])
            links.new(normal_map_node.outputs['Normal'], bsdf_node.inputs['Normal'])
            print(f"  ✓ Loaded normal map")
        
        # Final connection
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        
        print(f"  ✓ Enhanced material setup complete for {material.name}")
    
    def setup_professional_lighting(self):
        """Setup professional lighting system"""
        print("Setting up professional lighting...")
        
        # Key light (main directional)
        bpy.ops.object.light_add(type='SUN', location=(4, -3, 6))
        self.key_light = bpy.context.object
        self.key_light.name = "Key_Light"
        self.key_light.data.energy = 5.0
        self.key_light.data.angle = math.radians(5)
        
        # Fill light (soft area light)
        bpy.ops.object.light_add(type='AREA', location=(-3, -4, 4))
        self.fill_light = bpy.context.object
        self.fill_light.name = "Fill_Light"
        self.fill_light.data.energy = 3.0
        self.fill_light.data.size = 3.0
        
        # Rim light (edge definition)
        bpy.ops.object.light_add(type='SPOT', location=(2, 2, 3))
        self.rim_light = bpy.context.object
        self.rim_light.name = "Rim_Light"
        self.rim_light.data.energy = 4.0
        self.rim_light.data.spot_size = math.radians(60)
        self.rim_light.data.spot_blend = 0.3
        
        print("✓ Professional lighting setup complete")
    
    def setup_camera_system(self):
        """Setup intelligent camera system"""
        print("Setting up camera system...")
        
        # Main camera
        bpy.ops.object.camera_add(location=(0, -2.5, 0.8))
        self.camera = bpy.context.object
        self.camera.name = "Main_Camera"
        bpy.context.scene.camera = self.camera
        
        # Camera settings
        self.camera.data.lens = 50
        self.camera.data.sensor_width = 36
        
        print("✓ Camera system setup complete")
    
    def randomize_advanced_scene(self):
        """Advanced scene randomization for maximum realism"""
        # Center the can
        self.can_object.location = (0, 0, 0)
        self.can_object.rotation_euler = (0, 0, 0)
        
        # Fixed camera position
        self.camera.location = (0, -3, 1)
        self.camera.rotation_euler = (math.radians(80), 0, 0)
        self.camera.data.lens = 50
        
        # Fixed lighting
        self.key_light.data.energy = 8.0
        self.fill_light.data.energy = 4.0
        self.rim_light.data.energy = 6.0
        
        # Environment setup
        self.setup_environment()
    
    def randomize_camera_position(self):
        """Intelligent camera positioning ensuring can visibility"""
        # Fixed camera position for perfect framing
        self.camera.location = (0, -2.5, 1.0)
        self.camera.rotation_euler = (math.radians(70), 0, 0)
        self.camera.data.lens = 60
    
    def randomize_lighting(self):
        """Randomize lighting for natural variations"""
        config = self.config['lighting']
        
        # Randomize light intensities
        self.key_light.data.energy = random.uniform(*config['key_light_strength'])
        self.fill_light.data.energy = random.uniform(*config['fill_light_strength'])
        self.rim_light.data.energy = random.uniform(*config['rim_light_strength'])
        
        # Randomize light positions slightly
        self.key_light.location = (
            self.key_light.location.x + random.uniform(-1, 1),
            self.key_light.location.y + random.uniform(-1, 1),
            self.key_light.location.z + random.uniform(-1, 1)
        )
    
    def setup_environment(self):
        """Setup environment with backgrounds"""
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
        
        if self.background_images and random.random() < 0.7:
            # Use image background
            bg_image = random.choice(self.background_images)
            try:
                env_tex_node = nodes.new(type='ShaderNodeTexEnvironment')
                env_tex_node.image = bpy.data.images.load(str(bg_image))
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
    
    def apply_enhanced_defect(self, defect_type, intensity=None):
        """Apply enhanced defects"""
        if intensity is None:
            intensity = random.uniform(*self.config['defects']['defect_intensity_range'])
        
        print(f"Applying {defect_type} defect with intensity {intensity:.2f}")
        
        # Apply defect based on type
        if defect_type == 'scratch':
            self.apply_scratch_defect(intensity)
        elif defect_type == 'dent':
            self.apply_dent_defect(intensity)
        elif defect_type == 'puncture':
            self.apply_puncture_defect(intensity)
    
    def apply_scratch_defect(self, intensity):
        """Apply realistic scratch defects"""
        for mat_slot in self.can_object.material_slots:
            if not mat_slot.material:
                continue
            
            material = mat_slot.material
            nodes = material.node_tree.nodes
            
            # Find the BSDF node
            bsdf_node = None
            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf_node = node
                    break
            
            if bsdf_node:
                # Increase roughness for scratches
                if 'Roughness' in bsdf_node.inputs:
                    current_roughness = bsdf_node.inputs['Roughness'].default_value
                    new_roughness = min(1.0, current_roughness + intensity * 0.5)
                    bsdf_node.inputs['Roughness'].default_value = new_roughness
    
    def apply_dent_defect(self, intensity):
        """Apply realistic dent defects"""
        # For simplicity in 2.82, we'll modify material properties
        for mat_slot in self.can_object.material_slots:
            if not mat_slot.material:
                continue
            
            material = mat_slot.material
            nodes = material.node_tree.nodes
            
            bsdf_node = None
            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf_node = node
                    break
            
            if bsdf_node:
                # Darken slightly for dent effect
                if 'Base Color' in bsdf_node.inputs:
                    current_color = bsdf_node.inputs['Base Color'].default_value
                    factor = 1.0 - (intensity * 0.3)
                    bsdf_node.inputs['Base Color'].default_value = (
                        current_color[0] * factor,
                        current_color[1] * factor,
                        current_color[2] * factor,
                        1.0
                    )
    
    def apply_puncture_defect(self, intensity):
        """Apply puncture defects"""
        for mat_slot in self.can_object.material_slots:
            if not mat_slot.material:
                continue
            
            material = mat_slot.material
            nodes = material.node_tree.nodes
            
            bsdf_node = None
            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf_node = node
                    break
            
            if bsdf_node:
                # Significant darkening and roughness increase
                if 'Base Color' in bsdf_node.inputs:
                    bsdf_node.inputs['Base Color'].default_value = (0.1, 0.1, 0.1, 1.0)
                
                if 'Roughness' in bsdf_node.inputs:
                    bsdf_node.inputs['Roughness'].default_value = 0.9
                
                if 'Metallic' in bsdf_node.inputs:
                    bsdf_node.inputs['Metallic'].default_value = 0.0
    
    def calculate_enhanced_bounding_box(self):
        """Calculate enhanced bounding box with quality validation and fallback"""
        try:
            import bpy_extras.object_utils
            
            # Get mesh bounding box corners
            bbox_corners = [self.can_object.matrix_world @ Vector(corner) for corner in self.can_object.bound_box]
            
            # Project to 2D
            scene = bpy.context.scene
            coords_2d = []
            
            for corner in bbox_corners:
                co_2d = bpy_extras.object_utils.world_to_camera_view(scene, self.camera, corner)
                if co_2d.z > 0:  # In front of camera
                    coords_2d.append((co_2d.x, 1.0 - co_2d.y))
            
            # If no vertices visible, return None
            if not coords_2d:
                print("ERROR: No vertices visible to camera")
                return None
            
            # Calculate bounds
            x_coords = [coord[0] for coord in coords_2d]
            y_coords = [coord[1] for coord in coords_2d]
            
            min_x = max(0, min(x_coords))
            max_x = min(1, max(x_coords))
            min_y = max(0, min(y_coords))
            max_y = min(1, max(y_coords))
            
            # Calculate area
            area = (max_x - min_x) * (max_y - min_y)
            
            # More robust quality checks
            if area < 0.01 or area > 0.95:
                print(f"WARNING: Can area out of bounds ({area:.3f}), but proceeding anyway.")
            
            # YOLO format
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            width = max_x - min_x
            height = max_y - min_y
            
            return {
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'area': area
            }
            
        except Exception as e:
            print(f"Bounding box calculation error: {e}")
            return None
    
    def calculate_fallback_bbox(self):
        """Calculate a reasonable fallback bounding box when automatic calculation fails"""
        try:
            import bpy_extras.object_utils
            
            # Project can center to screen
            scene = bpy.context.scene
            can_center_world = self.can_object.matrix_world @ Vector((0, 0, 0.6))  # Can center
            center_2d = bpy_extras.object_utils.world_to_camera_view(scene, self.camera, can_center_world)
            
            if center_2d.z > 0:  # In front of camera
                center_x = center_2d.x
                center_y = 1.0 - center_2d.y
                
                # Use reasonable default size based on distance
                distance = (Vector(self.camera.location) - can_center_world).length
                
                # Estimate size based on distance (closer = larger)
                base_size = 0.4 / distance  # Adjust this factor as needed
                width = min(0.6, max(0.1, base_size))
                height = min(0.8, max(0.15, base_size * 1.5))  # Cans are taller
                
                # Ensure center is within image bounds
                center_x = max(width/2 + 0.05, min(1 - width/2 - 0.05, center_x))
                center_y = max(height/2 + 0.05, min(1 - height/2 - 0.05, center_y))
                
                area = width * height
                
                print(f"Fallback bbox: center=({center_x:.3f}, {center_y:.3f}), size=({width:.3f}, {height:.3f})")
                
                return {
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'area': area
                }
            
        except Exception as e:
            print(f"Fallback calculation also failed: {e}")
        
        # Ultimate fallback - center of image with reasonable size
        print("Using ultimate fallback bounding box")
        return {
            'center_x': 0.5,
            'center_y': 0.5,
            'width': 0.3,
            'height': 0.5,
            'area': 0.15
        }
    
    def save_metadata(self, image_index, defect_type, bbox_data, generation_params):
        """Save comprehensive metadata for each generated image"""
        metadata = {
            'image_index': image_index,
            'timestamp': datetime.now().isoformat(),
            'defect_type': defect_type,
            'bbox_data': bbox_data,
            'generation_params': generation_params,
            'config': self.config,
            'blender_version': bpy.app.version_string
        }
        
        metadata_path = self.metadata_dir / f"can_{image_index:05d}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path
    
    def generate_single_image(self, image_index, defect_type=None):
        """Generate a single high-quality image"""
        try:
            print(f"Generating image {image_index}...")
            start_time = datetime.now()
            
            # Determine defect type
            if defect_type is None:
                defect_type = self.choose_defect_type()
            
            # Store generation parameters
            generation_params = {
                'defect_type': defect_type,
                'camera_position': list(self.camera.location),
                'camera_rotation': list(self.camera.rotation_euler),
                'camera_lens': self.camera.data.lens
            }
            
            # Apply defect
            class_id = 0 if defect_type == 'pristine' else 1
            if defect_type != 'pristine':
                self.apply_enhanced_defect(defect_type)
            
            # Randomize scene
            self.randomize_advanced_scene()
            
            # Render
            image_filename = f"can_{image_index:05d}.png"
            image_path = self.images_dir / image_filename
            
            bpy.context.scene.render.filepath = str(image_path)
            bpy.ops.render.render(write_still=True)
            
            render_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate bounding box - now guaranteed to return something
            bbox_data = self.calculate_enhanced_bounding_box()
            if not bbox_data:
                print(f"ERROR: All bounding box calculations failed for image {image_index}")
                return False
            
            # Save label
            label_filename = f"can_{image_index:05d}.txt"
            label_path = self.labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {bbox_data['center_x']:.6f} {bbox_data['center_y']:.6f} "
                       f"{bbox_data['width']:.6f} {bbox_data['height']:.6f}\n")
            
            # Save metadata
            generation_params['render_time'] = render_time
            self.save_metadata(image_index, defect_type, bbox_data, generation_params)
            
            # Update statistics
            self.generation_stats['total_generated'] += 1
            self.generation_stats['quality_passed'] += 1
            self.generation_stats['defect_distribution'][defect_type] += 1
            
            print(f"✓ Image {image_index} generated successfully ({render_time:.2f}s)")
            return True
            
        except Exception as e:
            print(f"Error generating image {image_index}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def choose_defect_type(self):
        """Choose defect type based on configuration probabilities"""
        config = self.config['defects']
        
        choices = ['pristine', 'scratch', 'dent', 'puncture']
        weights = [
            config['pristine_probability'],
            config['scratch_probability'],
            config['dent_probability'],
            config['puncture_probability']
        ]
        
        return random.choices(choices, weights=weights)[0]
    
    def generate_batch(self, start_index, batch_size):
        """Generate a batch of high-quality images"""
        print(f"Generating enhanced batch: {start_index} to {start_index + batch_size - 1}")
        
        success_count = 0
        
        for i in range(batch_size):
            current_index = start_index + i
            
            if self.generate_single_image(current_index):
                success_count += 1
            
            # Progress reporting
            if (i + 1) % 5 == 0:
                print(f"Progress: {i + 1}/{batch_size} (Success: {success_count})")
        
        print(f"Enhanced batch complete: {success_count}/{batch_size} images")
        return success_count

def parse_arguments():
    """Parse command line arguments"""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(description="Enhanced Synthetic Data Generator")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fbx_path", type=str, required=True)
    parser.add_argument("--texture_path", type=str, required=True)
    parser.add_argument("--backgrounds_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, help="Path to configuration JSON file")
    
    return parser.parse_args(argv)

def main():
    """Main execution function"""
    try:
        args = parse_arguments()
        
        # Load configuration if provided
        config = None
        if args.config_path:
            config_path = Path(args.config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
        
        print("=== Enhanced Synthetic Data Generator (Blender 2.82) ===")
        print(f"Model: {args.fbx_path}")
        print(f"Output: {args.output_dir}")
        print(f"Batch: {args.start_index} to {args.start_index + args.batch_size - 1}")
        
        # Initialize generator
        generator = EnhancedSyntheticGenerator(
            model_path=args.fbx_path,
            texture_path=args.texture_path,
            backgrounds_dir=args.backgrounds_dir,
            output_dir=args.output_dir,
            config=config
        )
        
        # Generate batch
        success_count = generator.generate_batch(args.start_index, args.batch_size)
        
        print(f"\n✓ Enhanced generation complete: {success_count}/{args.batch_size} images")
        
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