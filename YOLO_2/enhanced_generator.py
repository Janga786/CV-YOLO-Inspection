"""
Enhanced Professional Synthetic Data Generator for Perfect Training Data
Includes advanced material system, improved defect simulation, and quality validation
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
        self.quality_dir = self.output_dir / "quality_checks"
        
        for dir_path in [self.images_dir, self.labels_dir, self.metadata_dir, self.quality_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get background images
        self.background_images = list(self.backgrounds_dir.glob("*.jpg")) + \
                               list(self.backgrounds_dir.glob("*.png")) + \
                               list(self.backgrounds_dir.glob("*.hdr")) + \
                               list(self.backgrounds_dir.glob("*.exr"))
        
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
        """Get default configuration for enhanced generation"""
        return {
            'render': {
                'resolution': [1024, 1024],
                'samples': 64,
                'engine': 'CYCLES',  # Use Cycles for photorealistic quality
                'denoising': True,
                'motion_blur': True,
                'depth_of_field': True
            },
            'camera': {
                'lens_range': [35, 85],
                'distance_range': [1.5, 3.0],
                'height_range': [0.3, 1.2],
                'angle_range': [-45, 45],
                'dof_enabled': True,
                'f_stop_range': [1.4, 8.0]
            },
            'lighting': {
                'use_hdri': True,
                'hdri_strength_range': [0.5, 2.0],
                'key_light_strength': [3.0, 8.0],
                'fill_light_strength': [1.0, 4.0],
                'rim_light_strength': [2.0, 6.0],
                'color_temperature_range': [3000, 6500]
            },
            'materials': {
                'label_roughness_variation': 0.3,
                'label_metallic_variation': 0.2,
                'metal_roughness_variation': 0.4,
                'metal_metallic_variation': 0.1,
                'wear_intensity_range': [0.0, 0.8],
                'dirt_intensity_range': [0.0, 0.5]
            },
            'defects': {
                'pristine_probability': 0.3,
                'scratch_probability': 0.25,
                'dent_probability': 0.25,
                'puncture_probability': 0.2,
                'defect_intensity_range': [0.3, 0.9],
                'multiple_defects_probability': 0.1
            },
            'quality': {
                'min_can_area': 0.05,  # Minimum 5% of image
                'max_can_area': 0.8,   # Maximum 80% of image
                'min_bbox_confidence': 0.7,
                'occlusion_threshold': 0.1,
                'blur_threshold': 0.05
            }
        }
    
    def setup_scene(self):
        """Setup enhanced Blender scene with professional settings"""
        # Clear scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Enhanced render settings
        scene = bpy.context.scene
        scene.render.engine = self.config['render']['engine']
        scene.render.resolution_x = self.config['render']['resolution'][0]
        scene.render.resolution_y = self.config['render']['resolution'][1]
        scene.render.film_transparent = False
        
        if scene.render.engine == 'CYCLES':
            # Cycles settings for quality
            scene.cycles.samples = self.config['render']['samples']
            scene.cycles.use_denoising = self.config['render']['denoising']
            scene.cycles.denoiser = 'OPTIX' if bpy.context.preferences.addons.get('cycles') else 'OPENIMAGEDENOISE'
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.adaptive_threshold = 0.01
            
            # Enable motion blur if configured
            if self.config['render']['motion_blur']:
                scene.render.use_motion_blur = True
                scene.render.motion_blur_shutter = 0.5
        
        elif scene.render.engine == 'BLENDER_EEVEE':
            # Enhanced EEVEE settings
            scene.eevee.taa_render_samples = self.config['render']['samples']
            scene.eevee.use_gtao = True
            scene.eevee.gtao_distance = 0.2
            scene.eevee.use_bloom = True
            scene.eevee.bloom_threshold = 0.8
            scene.eevee.use_ssr = True
            scene.eevee.use_ssr_refraction = True
            scene.eevee.ssr_quality = 0.25
            scene.eevee.use_volumetric_lighting = True
        
        # Load and setup the can model
        self.load_model()
        self.setup_professional_lighting()
        self.setup_camera_system()
    
    def load_model(self):
        """Load and prepare the can model with enhanced materials"""
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
        
        # Add subdivision for better defect simulation
        self.add_geometry_detail()
    
    def add_geometry_detail(self):
        """Add subdivision surface for better defect simulation"""
        # Add subdivision surface modifier
        subdiv_mod = self.can_object.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv_mod.levels = 2
        subdiv_mod.render_levels = 3
        
        # Add edge split to maintain sharp edges where needed
        edge_split_mod = self.can_object.modifiers.new(name="EdgeSplit", type='EDGE_SPLIT')
        edge_split_mod.split_angle = math.radians(30)
    
    def setup_enhanced_materials(self):
        """Setup enhanced PBR materials with procedural variations"""
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
        """Create enhanced material node setup with procedural variations"""
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        
        # Base nodes
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        
        output_node.location = (1000, 0)
        bsdf_node.location = (600, 0)
        
        # Determine texture set
        texture_prefix = "Label" if 'label' in material.name.lower() else "lambert1"
        is_label = texture_prefix == "Label"
        
        # UV coordinates
        uv_node = nodes.new(type='ShaderNodeTexCoord')
        uv_node.location = (-1000, 0)
        
        # Add UV distortion for realism
        noise_node = nodes.new(type='ShaderNodeTexNoise')
        noise_node.location = (-800, 200)
        noise_node.inputs['Scale'].default_value = 50.0
        # Note: Blender 2.82 compatibility - using Detail instead of Roughness
        if 'Detail' in noise_node.inputs:
            noise_node.inputs['Detail'].default_value = 2.0
        elif 'Roughness' in noise_node.inputs:
            noise_node.inputs['Roughness'].default_value = 0.5
        
        vector_math_node = nodes.new(type='ShaderNodeVectorMath')
        vector_math_node.location = (-600, 100)
        vector_math_node.operation = 'ADD'
        
        links.new(uv_node.outputs['UV'], vector_math_node.inputs[0])
        links.new(noise_node.outputs['Fac'], vector_math_node.inputs[1])
        
        # Base color setup
        base_color_path = textures_dir / f"{texture_prefix}_Base_color.png"
        if base_color_path.exists():
            color_node = nodes.new(type='ShaderNodeTexImage')
            color_node.location = (-400, 300)
            color_node.image = bpy.data.images.load(str(base_color_path))
            links.new(vector_math_node.outputs['Vector'], color_node.inputs['Vector'])
            
            # Add color variation - Blender 2.82 compatibility
            if bpy.app.version >= (2, 90, 0):
                color_mix_node = nodes.new(type='ShaderNodeMixRGB')
                color_mix_node.blend_type = 'MULTIPLY'
                color_input = 'Color1'
                color_output = 'Color'
            else:
                color_mix_node = nodes.new(type='ShaderNodeMixRGB')
                color_mix_node.blend_type = 'MULTIPLY'
                color_input = 'Color1'
                color_output = 'Color'
            
            color_mix_node.location = (-200, 300)
            
            links.new(color_node.outputs['Color'], color_mix_node.inputs[color_input])
            links.new(color_mix_node.outputs[color_output], bsdf_node.inputs['Base Color'])
        
        # Enhanced metallic setup
        metallic_path = textures_dir / f"{texture_prefix}_Metallic.png"
        if metallic_path.exists():
            metallic_node = nodes.new(type='ShaderNodeTexImage')
            metallic_node.location = (-400, 0)
            metallic_node.image = bpy.data.images.load(str(metallic_path))
            metallic_node.image.colorspace_settings.name = 'Non-Color'
            links.new(vector_math_node.outputs['Vector'], metallic_node.inputs['Vector'])
            
            # Add metallic variation with ColorRamp
            metallic_ramp = nodes.new(type='ShaderNodeValToRGB')
            metallic_ramp.location = (-200, 0)
            links.new(metallic_node.outputs['Color'], metallic_ramp.inputs['Fac'])
            links.new(metallic_ramp.outputs['Color'], bsdf_node.inputs['Metallic'])
        
        # Enhanced roughness setup
        roughness_path = textures_dir / f"{texture_prefix}_Roughness.png"
        if roughness_path.exists():
            roughness_node = nodes.new(type='ShaderNodeTexImage')
            roughness_node.location = (-400, -200)
            roughness_node.image = bpy.data.images.load(str(roughness_path))
            roughness_node.image.colorspace_settings.name = 'Non-Color'
            links.new(vector_math_node.outputs['Vector'], roughness_node.inputs['Vector'])
            
            # Add procedural wear and dirt
            wear_noise = nodes.new(type='ShaderNodeTexNoise')
            wear_noise.location = (-600, -300)
            wear_noise.inputs['Scale'].default_value = 20.0
            
            roughness_mix = nodes.new(type='ShaderNodeMixRGB')
            roughness_mix.location = (-200, -200)
            roughness_mix.blend_type = 'ADD'
            
            links.new(roughness_node.outputs['Color'], roughness_mix.inputs['Color1'])
            links.new(wear_noise.outputs['Fac'], roughness_mix.inputs['Color2'])
            links.new(roughness_mix.outputs['Color'], bsdf_node.inputs['Roughness'])
        
        # Normal map setup
        normal_path = textures_dir / f"{texture_prefix}_Normal_OpenGL.png"
        if normal_path.exists():
            normal_tex_node = nodes.new(type='ShaderNodeTexImage')
            normal_map_node = nodes.new(type='ShaderNodeNormalMap')
            normal_tex_node.location = (-400, -400)
            normal_map_node.location = (0, -400)
            
            normal_tex_node.image = bpy.data.images.load(str(normal_path))
            normal_tex_node.image.colorspace_settings.name = 'Non-Color'
            
            links.new(vector_math_node.outputs['Vector'], normal_tex_node.inputs['Vector'])
            links.new(normal_tex_node.outputs['Color'], normal_map_node.inputs['Color'])
            links.new(normal_map_node.outputs['Normal'], bsdf_node.inputs['Normal'])
        
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
        self.fill_light.data.shape = 'RECTANGLE'
        self.fill_light.data.size_y = 2.0
        
        # Rim light (edge definition)
        bpy.ops.object.light_add(type='SPOT', location=(2, 2, 3))
        self.rim_light = bpy.context.object
        self.rim_light.name = "Rim_Light"
        self.rim_light.data.energy = 4.0
        self.rim_light.data.spot_size = math.radians(60)
        self.rim_light.data.spot_blend = 0.3
        
        # Bounce light (simulated bounce from ground)
        bpy.ops.object.light_add(type='AREA', location=(0, -1, -0.5))
        self.bounce_light = bpy.context.object
        self.bounce_light.name = "Bounce_Light"
        self.bounce_light.data.energy = 1.5
        self.bounce_light.data.size = 4.0
        self.bounce_light.rotation_euler = (math.radians(90), 0, 0)
        
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
        self.camera.data.sensor_height = 24
        
        # Depth of field setup
        if self.config['camera']['dof_enabled']:
            self.camera.data.dof.use_dof = True
            self.camera.data.dof.focus_distance = 2.5
            self.camera.data.dof.aperture_fstop = 2.8
        
        print("✓ Camera system setup complete")
    
    def randomize_advanced_scene(self):
        """Advanced scene randomization for maximum realism"""
        # Randomize can position with subtle variations
        self.can_object.location = (
            random.uniform(-0.05, 0.05),
            random.uniform(-0.05, 0.05),
            max(0, random.uniform(-0.01, 0.02))
        )
        
        # Randomize can rotation
        self.can_object.rotation_euler = (
            random.uniform(-math.radians(5), math.radians(5)),
            random.uniform(-math.radians(5), math.radians(5)),
            random.uniform(0, 2 * math.pi)
        )
        
        # Intelligent camera positioning
        self.randomize_camera_position()
        
        # Dynamic lighting
        self.randomize_lighting()
        
        # Environment setup
        self.setup_environment()
    
    def randomize_camera_position(self):
        """Intelligent camera positioning ensuring can visibility"""
        config = self.config['camera']
        
        # Use spherical coordinates for natural camera movement
        distance = random.uniform(*config['distance_range'])
        azimuth = random.uniform(math.radians(-60), math.radians(60))
        elevation = random.uniform(math.radians(10), math.radians(80))
        
        # Calculate camera position
        cam_x = distance * math.cos(elevation) * math.cos(azimuth)
        cam_y = -distance * math.cos(elevation) * math.sin(azimuth)
        cam_z = distance * math.sin(elevation)
        
        self.camera.location = (cam_x, cam_y, cam_z)
        
        # Point camera at can with slight randomization
        can_center = Vector((0, 0, 0.6))
        can_center.x += random.uniform(-0.1, 0.1)
        can_center.y += random.uniform(-0.1, 0.1)
        can_center.z += random.uniform(-0.1, 0.1)
        
        direction = can_center - Vector(self.camera.location)
        rot_quat = direction.to_track_quat('-Z', 'Y')
        self.camera.rotation_euler = rot_quat.to_euler()
        
        # Randomize camera properties
        self.camera.data.lens = random.uniform(*config['lens_range'])
        
        if config['dof_enabled']:
            self.camera.data.dof.focus_distance = distance + random.uniform(-0.2, 0.2)
            self.camera.data.dof.aperture_fstop = random.uniform(*config['f_stop_range'])
    
    def randomize_lighting(self):
        """Randomize lighting for natural variations"""
        config = self.config['lighting']
        
        # Randomize light intensities
        self.key_light.data.energy = random.uniform(*config['key_light_strength'])
        self.fill_light.data.energy = random.uniform(*config['fill_light_strength'])
        self.rim_light.data.energy = random.uniform(*config['rim_light_strength'])
        
        # Randomize light colors (color temperature)
        temp = random.uniform(*config['color_temperature_range'])
        color = self.temperature_to_rgb(temp)
        
        self.key_light.data.color = color
        self.fill_light.data.color = (color[0] * 0.9, color[1] * 0.95, color[2] * 1.1)
        
        # Randomize light positions slightly
        self.key_light.location = (
            self.key_light.location.x + random.uniform(-1, 1),
            self.key_light.location.y + random.uniform(-1, 1),
            self.key_light.location.z + random.uniform(-1, 1)
        )
    
    def temperature_to_rgb(self, temp):
        """Convert color temperature to RGB"""
        temp = temp / 100.0
        
        if temp <= 66:
            red = 255
            green = temp
            green = 99.4708025861 * math.log(green) - 161.1195681661
            if temp >= 19:
                blue = temp - 10
                blue = 138.5177312231 * math.log(blue) - 305.0447927307
            else:
                blue = 0
        else:
            red = temp - 60
            red = 329.698727446 * math.pow(red, -0.1332047592)
            green = temp - 60
            green = 288.1221695283 * math.pow(green, -0.0755148492)
            blue = 255
        
        return (
            max(0, min(255, red)) / 255.0,
            max(0, min(255, green)) / 255.0,
            max(0, min(255, blue)) / 255.0
        )
    
    def setup_environment(self):
        """Setup environment with HDRI or procedural sky"""
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()
        
        output_node = nodes.new(type='ShaderNodeOutputWorld')
        
        if self.config['lighting']['use_hdri'] and self.background_images:
            # Use HDRI environment
            bg_image = random.choice(self.background_images)
            if bg_image.suffix.lower() in ['.hdr', '.exr']:
                env_tex_node = nodes.new(type='ShaderNodeTexEnvironment')
                env_tex_node.image = bpy.data.images.load(str(bg_image))
                
                background_node = nodes.new(type='ShaderNodeBackground')
                background_node.inputs['Strength'].default_value = random.uniform(
                    *self.config['lighting']['hdri_strength_range']
                )
                
                links.new(env_tex_node.outputs['Color'], background_node.inputs['Color'])
                links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
            else:
                self.setup_procedural_sky(nodes, links, output_node)
        else:
            self.setup_procedural_sky(nodes, links, output_node)
    
    def setup_procedural_sky(self, nodes, links, output_node):
        """Setup procedural sky environment"""
        sky_tex_node = nodes.new(type='ShaderNodeTexSky')
        sky_tex_node.sky_type = 'PREETHAM'
        sky_tex_node.sun_elevation = random.uniform(0.1, 1.4)
        sky_tex_node.sun_rotation = random.uniform(0, 2 * math.pi)
        
        background_node = nodes.new(type='ShaderNodeBackground')
        background_node.inputs['Strength'].default_value = random.uniform(0.8, 1.5)
        
        links.new(sky_tex_node.outputs['Color'], background_node.inputs['Color'])
        links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
    
    def apply_enhanced_defect(self, defect_type, intensity=None):
        """Apply enhanced defects using procedural techniques"""
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
        elif defect_type == 'wear':
            self.apply_wear_defect(intensity)
        elif defect_type == 'dirt':
            self.apply_dirt_defect(intensity)
    
    def apply_scratch_defect(self, intensity):
        """Apply realistic scratch defects"""
        # Modify material properties to simulate scratches
        for mat_slot in self.can_object.material_slots:
            if not mat_slot.material:
                continue
            
            material = mat_slot.material
            nodes = material.node_tree.nodes
            
            # Find the roughness input
            bsdf_node = None
            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf_node = node
                    break
            
            if bsdf_node:
                # Add scratch procedural texture
                scratch_noise = nodes.new(type='ShaderNodeTexNoise')
                scratch_noise.location = (-400, -600)
                scratch_noise.inputs['Scale'].default_value = 100.0 * intensity
                scratch_noise.inputs['Roughness'].default_value = 0.8
                
                # Mix with existing roughness
                if 'Roughness' in bsdf_node.inputs:
                    current_roughness = bsdf_node.inputs['Roughness'].default_value
                    new_roughness = min(1.0, current_roughness + intensity * 0.5)
                    bsdf_node.inputs['Roughness'].default_value = new_roughness
    
    def apply_dent_defect(self, intensity):
        """Apply realistic dent defects using displacement"""
        # Add displacement modifier for dents
        if not any(mod.type == 'DISPLACE' for mod in self.can_object.modifiers):
            displace_mod = self.can_object.modifiers.new(name="Dent_Displace", type='DISPLACE')
            
            # Create displacement texture
            dent_texture = bpy.data.textures.new(name="Dent_Texture", type='CLOUDS')
            dent_texture.noise_scale = 0.5
            dent_texture.noise_depth = 2
            dent_texture.nabla = 0.025
            
            displace_mod.texture = dent_texture
            displace_mod.strength = -0.02 * intensity
            displace_mod.mid_level = 0.5
    
    def apply_puncture_defect(self, intensity):
        """Apply puncture defects"""
        # Similar to dent but more localized and severe
        self.apply_dent_defect(intensity * 1.5)
        
        # Also modify materials for puncture appearance
        for mat_slot in self.can_object.material_slots:
            if not mat_slot.material:
                continue
            
            material = mat_slot.material
            nodes = material.node_tree.nodes
            
            # Find BSDF node
            bsdf_node = None
            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf_node = node
                    break
            
            if bsdf_node:
                # Darken the material
                if 'Base Color' in bsdf_node.inputs:
                    current_color = bsdf_node.inputs['Base Color'].default_value
                    factor = 1.0 - (intensity * 0.6)
                    bsdf_node.inputs['Base Color'].default_value = (
                        current_color[0] * factor,
                        current_color[1] * factor,
                        current_color[2] * factor,
                        1.0
                    )
                
                # Increase roughness significantly
                if 'Roughness' in bsdf_node.inputs:
                    bsdf_node.inputs['Roughness'].default_value = min(1.0, 0.9 * intensity)
                
                # Reduce metallic property
                if 'Metallic' in bsdf_node.inputs:
                    current_metallic = bsdf_node.inputs['Metallic'].default_value
                    bsdf_node.inputs['Metallic'].default_value = current_metallic * (1.0 - intensity * 0.8)
    
    def apply_wear_defect(self, intensity):
        """Apply general wear defects"""
        # Combine multiple subtle defects
        self.apply_scratch_defect(intensity * 0.3)
        
        # Modify material properties for wear
        for mat_slot in self.can_object.material_slots:
            if not mat_slot.material:
                continue
            
            material = mat_slot.material
            nodes = material.node_tree.nodes
            
            # Find BSDF node
            bsdf_node = None
            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf_node = node
                    break
            
            if bsdf_node:
                # Subtle color change
                if 'Base Color' in bsdf_node.inputs:
                    current_color = bsdf_node.inputs['Base Color'].default_value
                    factor = 1.0 - (intensity * 0.2)
                    bsdf_node.inputs['Base Color'].default_value = (
                        current_color[0] * factor,
                        current_color[1] * factor,
                        current_color[2] * factor * 0.95,  # Slight yellowing
                        1.0
                    )
                
                # Increase roughness
                if 'Roughness' in bsdf_node.inputs:
                    current_roughness = bsdf_node.inputs['Roughness'].default_value
                    bsdf_node.inputs['Roughness'].default_value = min(1.0, current_roughness + intensity * 0.3)
    
    def apply_dirt_defect(self, intensity):
        """Apply dirt/grime defects"""
        # Add dirt through material modification
        for mat_slot in self.can_object.material_slots:
            if not mat_slot.material:
                continue
            
            material = mat_slot.material
            nodes = material.node_tree.nodes
            
            # Find BSDF node
            bsdf_node = None
            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf_node = node
                    break
            
            if bsdf_node:
                # Darken and desaturate
                if 'Base Color' in bsdf_node.inputs:
                    current_color = bsdf_node.inputs['Base Color'].default_value
                    dirt_factor = 1.0 - (intensity * 0.4)
                    bsdf_node.inputs['Base Color'].default_value = (
                        current_color[0] * dirt_factor + 0.1 * intensity,
                        current_color[1] * dirt_factor + 0.08 * intensity,
                        current_color[2] * dirt_factor + 0.06 * intensity,
                        1.0
                    )
                
                # Increase roughness
                if 'Roughness' in bsdf_node.inputs:
                    current_roughness = bsdf_node.inputs['Roughness'].default_value
                    bsdf_node.inputs['Roughness'].default_value = min(1.0, current_roughness + intensity * 0.4)
    
    def calculate_enhanced_bounding_box(self):
        """Calculate enhanced bounding box with quality validation"""
        try:
            import bpy_extras.object_utils
            
            # Get all mesh vertices in world space
            mesh = self.can_object.data
            world_vertices = [self.can_object.matrix_world @ v.co for v in mesh.vertices]
            
            # Project to camera view
            scene = bpy.context.scene
            coords_2d = []
            z_values = []
            
            for vertex in world_vertices:
                co_2d = bpy_extras.object_utils.world_to_camera_view(scene, self.camera, vertex)
                if co_2d.z > 0:  # In front of camera
                    coords_2d.append((co_2d.x, 1.0 - co_2d.y))
                    z_values.append(co_2d.z)
            
            if len(coords_2d) < 4:
                print("Warning: Too few vertices visible")
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
            
            # Quality checks
            if area < self.config['quality']['min_can_area']:
                print(f"Warning: Can too small (area: {area:.3f})")
                return None
            
            if area > self.config['quality']['max_can_area']:
                print(f"Warning: Can too large (area: {area:.3f})")
                return None
            
            # YOLO format
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            width = max_x - min_x
            height = max_y - min_y
            
            # Ensure reasonable bounds
            center_x = max(0.05, min(0.95, center_x))
            center_y = max(0.05, min(0.95, center_y))
            width = max(0.1, min(0.9, width))
            height = max(0.1, min(0.9, height))
            
            return {
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'area': area,
                'depth_mean': np.mean(z_values),
                'depth_std': np.std(z_values),
                'vertex_count': len(coords_2d)
            }
            
        except Exception as e:
            print(f"Bounding box calculation error: {e}")
            return None
    
    def validate_image_quality(self, image_path, bbox_data):
        """Validate generated image quality"""
        try:
            # Basic file validation
            if not image_path.exists() or image_path.stat().st_size < 10000:
                return False, "File too small or missing"
            
            # Bounding box validation
            if not bbox_data:
                return False, "Invalid bounding box"
            
            if bbox_data['area'] < self.config['quality']['min_can_area']:
                return False, f"Can too small: {bbox_data['area']:.3f}"
            
            # Additional quality checks could be added here
            # - Image sharpness detection
            # - Occlusion detection
            # - Lighting validation
            
            return True, "Quality check passed"
            
        except Exception as e:
            return False, f"Quality check error: {e}"
    
    def save_metadata(self, image_index, defect_type, bbox_data, generation_params):
        """Save comprehensive metadata for each generated image"""
        metadata = {
            'image_index': image_index,
            'timestamp': datetime.now().isoformat(),
            'defect_type': defect_type,
            'bbox_data': bbox_data,
            'generation_params': generation_params,
            'config': self.config,
            'blender_version': bpy.app.version_string,
            'render_engine': bpy.context.scene.render.engine,
            'render_time': None,  # Will be filled by generate_single_image
            'file_size': None     # Will be filled by generate_single_image
        }
        
        metadata_path = self.metadata_dir / f"can_{image_index:05d}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path
    
    def generate_single_image(self, image_index, defect_type=None):
        """Generate a single high-quality image with comprehensive validation"""
        try:
            print(f"Generating image {image_index}...")
            start_time = datetime.now()
            
            # Reset all modifiers and materials
            self.reset_all_modifications()
            
            # Determine defect type
            if defect_type is None:
                defect_type = self.choose_defect_type()
            
            # Store generation parameters
            generation_params = {
                'defect_type': defect_type,
                'camera_position': list(self.camera.location),
                'camera_rotation': list(self.camera.rotation_euler),
                'camera_lens': self.camera.data.lens,
                'lighting_energy': {
                    'key': self.key_light.data.energy,
                    'fill': self.fill_light.data.energy,
                    'rim': self.rim_light.data.energy
                }
            }
            
            # Apply defect
            class_id = 0 if defect_type == 'pristine' else 1
            if defect_type != 'pristine':
                self.apply_enhanced_defect(defect_type)
            
            # Randomize scene
            self.randomize_advanced_scene()
            
            # Update generation parameters with final values
            generation_params.update({
                'final_camera_position': list(self.camera.location),
                'final_camera_rotation': list(self.camera.rotation_euler),
                'final_camera_lens': self.camera.data.lens,
                'final_lighting_energy': {
                    'key': self.key_light.data.energy,
                    'fill': self.fill_light.data.energy,
                    'rim': self.rim_light.data.energy
                }
            })
            
            # Render
            image_filename = f"can_{image_index:05d}.png"
            image_path = self.images_dir / image_filename
            
            bpy.context.scene.render.filepath = str(image_path)
            bpy.ops.render.render(write_still=True)
            
            render_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate bounding box
            bbox_data = self.calculate_enhanced_bounding_box()
            if not bbox_data:
                print(f"Failed to calculate bounding box for image {image_index}")
                return False
            
            # Validate quality
            quality_passed, quality_message = self.validate_image_quality(image_path, bbox_data)
            if not quality_passed:
                print(f"Quality validation failed for image {image_index}: {quality_message}")
                return False
            
            # Save label
            label_filename = f"can_{image_index:05d}.txt"
            label_path = self.labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {bbox_data['center_x']:.6f} {bbox_data['center_y']:.6f} "
                       f"{bbox_data['width']:.6f} {bbox_data['height']:.6f}\n")
            
            # Save metadata
            generation_params['render_time'] = render_time
            generation_params['file_size'] = image_path.stat().st_size
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
        
        # Check for multiple defects
        if random.random() < config['multiple_defects_probability']:
            # Choose multiple defect types
            available_defects = ['scratch', 'dent', 'wear', 'dirt']
            num_defects = random.randint(2, 3)
            return random.sample(available_defects, num_defects)
        
        # Single defect or pristine
        choices = ['pristine', 'scratch', 'dent', 'puncture']
        weights = [
            config['pristine_probability'],
            config['scratch_probability'],
            config['dent_probability'],
            config['puncture_probability']
        ]
        
        return random.choices(choices, weights=weights)[0]
    
    def reset_all_modifications(self):
        """Reset all modifiers and materials to original state"""
        # Clear all modifiers
        for modifier in self.can_object.modifiers:
            self.can_object.modifiers.remove(modifier)
        
        # Re-add base subdivision
        self.add_geometry_detail()
        
        # Reset materials (already handled in setup_enhanced_materials)
        # This would be more complex in a full implementation
    
    def generate_batch(self, start_index, batch_size):
        """Generate a batch of high-quality images"""
        print(f"Generating enhanced batch: {start_index} to {start_index + batch_size - 1}")
        
        success_count = 0
        
        for i in range(batch_size):
            current_index = start_index + i
            
            # Generate with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                if self.generate_single_image(current_index):
                    success_count += 1
                    break
                else:
                    print(f"Retry {attempt + 1}/{max_retries} for image {current_index}")
            
            # Progress reporting
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{batch_size} (Success: {success_count})")
        
        # Save batch statistics
        self.save_batch_statistics(start_index, batch_size, success_count)
        
        print(f"Enhanced batch complete: {success_count}/{batch_size} images")
        return success_count
    
    def save_batch_statistics(self, start_index, batch_size, success_count):
        """Save batch generation statistics"""
        stats_path = self.output_dir / "generation_stats.json"
        
        batch_stats = {
            'batch_info': {
                'start_index': start_index,
                'batch_size': batch_size,
                'success_count': success_count,
                'success_rate': success_count / batch_size,
                'timestamp': datetime.now().isoformat()
            },
            'overall_stats': self.generation_stats
        }
        
        with open(stats_path, 'w') as f:
            json.dump(batch_stats, f, indent=2)

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
        
        print("=== Enhanced Synthetic Data Generator ===")
        print(f"Model: {args.fbx_path}")
        print(f"Output: {args.output_dir}")
        print(f"Batch: {args.start_index} to {args.start_index + args.batch_size - 1}")
        print(f"Configuration: {'Custom' if config else 'Default'}")
        
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