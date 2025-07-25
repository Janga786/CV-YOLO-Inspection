"""
Professional Soda Can Defect Detection Dataset Generator - Realistic Version
Generates coke cans with realistic defects using geometry modification and texture overlays
Classes: 0=can, 1=scratch, 2=dent, 3=puncture
"""

import bpy
import random
import math
import os
import sys
import argparse
from pathlib import Path
from mathutils import Vector
import bmesh
import numpy as np

class ProfessionalCanDefectGenerator:
    def __init__(self, model_path, texture_path, backgrounds_dir, output_dir, defect_textures_dir=None):
        self.model_path = Path(model_path)
        self.texture_path = Path(texture_path)
        self.backgrounds_dir = Path(backgrounds_dir)
        self.output_dir = Path(output_dir)
        self.defect_textures_dir = Path(defect_textures_dir) if defect_textures_dir else None
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get background images
        self.background_images = list(self.backgrounds_dir.glob("*.jpg")) + \
                               list(self.backgrounds_dir.glob("*.png"))
        
        print(f"Found {len(self.background_images)} background images")
        
        # Store original materials and mesh for reset
        self.original_materials = {}
        self.original_mesh = None
        
        # Camera frustum parameters for visibility check
        self.camera_fov_margin = 0.70
        
        # Defect tracking
        self.defect_regions = []  # List of (defect_type, center, radius) for bounding boxes
        
        # Class mapping
        self.class_map = {
            'can': 0,
            'scratch': 1,
            'dent': 2,
            'puncture': 3
        }
        
        self.setup_scene()
    
    def setup_scene(self):
        """Setup Blender scene with optimal settings"""
        # Clear scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Render settings - use CYCLES for better quality
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 64  # Lower samples for faster rendering
        scene.render.resolution_x = 640
        scene.render.resolution_y = 640
        scene.render.film_transparent = False
        
        # Enable GPU rendering if available
        scene.cycles.device = 'GPU'
        
        # Camera setup
        bpy.ops.object.camera_add(location=(0, -2.5, 0.6))
        self.camera = bpy.context.object
        self.camera.rotation_euler = (math.radians(80), 0, 0)
        scene.camera = self.camera
        self.camera.data.lens = 50
        self.camera.data.clip_start = 0.1
        self.camera.data.clip_end = 100.0
        
        # Professional lighting setup
        # Key light (sun) - stronger for better defect visibility
        bpy.ops.object.light_add(type='SUN', location=(3, -2, 5))
        self.key_light = bpy.context.object
        self.key_light.data.energy = 4.0
        self.key_light.data.angle = math.radians(15)
        
        # Fill light (area)
        bpy.ops.object.light_add(type='AREA', location=(-2, -3, 3))
        self.fill_light = bpy.context.object
        self.fill_light.data.energy = 2.5
        self.fill_light.data.size = 2.0
        
        # Rim light for edge definition
        bpy.ops.object.light_add(type='SPOT', location=(1, 1, 2))
        self.rim_light = bpy.context.object
        self.rim_light.data.energy = 2.0
        self.rim_light.data.spot_size = math.radians(45)
        
        # Additional top light for defect visibility
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 3))
        self.top_light = bpy.context.object
        self.top_light.data.energy = 1.5
        self.top_light.data.size = 1.0
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load the Sketchfab model and properly set up materials"""
        print(f"Loading model: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Import FBX
        try:
            existing_objects = set(bpy.data.objects[:])
            bpy.ops.import_scene.fbx(filepath=str(self.model_path))
            
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
            bpy.ops.object.select_all(action='DESELECT')
            for obj in mesh_objects:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = mesh_objects[0]
            bpy.ops.object.join()
            self.can_object = bpy.context.active_object
        else:
            self.can_object = mesh_objects[0]
        
        print(f"Using can object: {self.can_object.name}")
        
        # Apply transformations
        bpy.context.view_layer.objects.active = self.can_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # Center and scale the can
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
        self.can_object.location = (0, 0, 0)
        self.can_object.rotation_euler = (0, 0, 0)
        
        # Scale to standard can size
        bbox = self.can_object.bound_box
        x_size = max(v[0] for v in bbox) - min(v[0] for v in bbox)
        y_size = max(v[1] for v in bbox) - min(v[1] for v in bbox)
        z_size = max(v[2] for v in bbox) - min(v[2] for v in bbox)
        
        # Check if rotation needed
        if abs(x_size - z_size) < 0.1 and y_size > max(x_size, z_size):
            self.can_object.rotation_euler = (math.radians(90), 0, 0)
            bpy.context.view_layer.objects.active = self.can_object
            bpy.ops.object.transform_apply(rotation=True)
            bbox = self.can_object.bound_box
            current_height = max(v[2] for v in bbox) - min(v[2] for v in bbox)
        else:
            current_height = z_size
        
        if current_height > 0:
            scale_factor = 1.23 / current_height
            self.can_object.scale = (scale_factor, scale_factor, scale_factor)
        
        bpy.context.view_layer.objects.active = self.can_object
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        # Position on ground
        bbox = self.can_object.bound_box
        min_z = min(v[2] for v in bbox)
        self.can_object.location.z = -min_z
        
        # Store dimensions
        self.can_dimensions = self.get_object_dimensions()
        print(f"Can dimensions: {self.can_dimensions}")
        
        # Add subdivision surface for smoother deformations
        subdiv = self.can_object.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv.levels = 2
        subdiv.render_levels = 2
        
        # Store original mesh data
        self.store_original_mesh()
        
        # Setup materials
        self.setup_all_materials()
    
    def store_original_mesh(self):
        """Store a copy of the original mesh for resetting"""
        # Create a copy of the mesh data
        self.original_mesh = self.can_object.data.copy()
    
    def reset_mesh(self):
        """Reset the mesh to its original state"""
        # Replace current mesh with a copy of the original
        new_mesh = self.original_mesh.copy()
        old_mesh = self.can_object.data
        self.can_object.data = new_mesh
        # Remove the old mesh data
        bpy.data.meshes.remove(old_mesh)
    
    def get_object_dimensions(self):
        """Get the dimensions of the can object"""
        bbox = self.can_object.bound_box
        width = max(v[0] for v in bbox) - min(v[0] for v in bbox)
        depth = max(v[1] for v in bbox) - min(v[1] for v in bbox)
        height = max(v[2] for v in bbox) - min(v[2] for v in bbox)
        
        bbox_world = [self.can_object.matrix_world @ Vector(corner) for corner in self.can_object.bound_box]
        world_center = Vector((
            sum(v[0] for v in bbox_world) / 8,
            sum(v[1] for v in bbox_world) / 8,
            sum(v[2] for v in bbox_world) / 8
        ))
        
        return {
            'width': width,
            'depth': depth,
            'height': height,
            'center': world_center,
            'radius': max(width, depth) / 2
        }
    
    def setup_all_materials(self):
        """Setup all materials with proper texture assignments"""
        print("Setting up materials with all texture maps...")
        
        textures_dir = Path("textures")
        
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
            
            nodes.clear()
            
            # Create base nodes
            output_node = nodes.new(type='ShaderNodeOutputMaterial')
            bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
            
            output_node.location = (600, 0)
            bsdf_node.location = (300, 0)
            
            links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
            
            # Determine texture prefix
            texture_prefix = None
            if 'label' in material.name.lower():
                texture_prefix = "Label"
            elif 'lambert' in material.name.lower() or 'body' in material.name.lower():
                texture_prefix = "lambert1"
            else:
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
                color_node.name = "BaseColor"
                links.new(uv_node.outputs['UV'], color_node.inputs['Vector'])
                links.new(color_node.outputs['Color'], bsdf_node.inputs['Base Color'])
            else:
                if texture_prefix == "Label":
                    bsdf_node.inputs['Base Color'].default_value = (0.8, 0.05, 0.05, 1.0)
                else:
                    bsdf_node.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1.0)
            
            # Metallic
            metallic_path = textures_dir / f"{texture_prefix}_Metallic.png"
            if metallic_path.exists():
                metallic_node = nodes.new(type='ShaderNodeTexImage')
                metallic_node.location = (-300, y_offset - 250)
                metallic_node.image = bpy.data.images.load(str(metallic_path))
                metallic_node.image.colorspace_settings.name = 'Non-Color'
                metallic_node.name = "Metallic"
                links.new(uv_node.outputs['UV'], metallic_node.inputs['Vector'])
                links.new(metallic_node.outputs['Color'], bsdf_node.inputs['Metallic'])
            else:
                bsdf_node.inputs['Metallic'].default_value = 0.9 if texture_prefix == "lambert1" else 0.3
            
            # Roughness
            roughness_path = textures_dir / f"{texture_prefix}_Roughness.png"
            if roughness_path.exists():
                roughness_node = nodes.new(type='ShaderNodeTexImage')
                roughness_node.location = (-300, y_offset - 500)
                roughness_node.image = bpy.data.images.load(str(roughness_path))
                roughness_node.image.colorspace_settings.name = 'Non-Color'
                roughness_node.name = "Roughness"
                links.new(uv_node.outputs['UV'], roughness_node.inputs['Vector'])
                links.new(roughness_node.outputs['Color'], bsdf_node.inputs['Roughness'])
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
                normal_tex_node.name = "Normal"
                
                links.new(uv_node.outputs['UV'], normal_tex_node.inputs['Vector'])
                links.new(normal_tex_node.outputs['Color'], normal_map_node.inputs['Color'])
                links.new(normal_map_node.outputs['Normal'], bsdf_node.inputs['Normal'])
        
        print("✓ All materials configured")
    
    def create_scratch_defect(self):
        """Create a visible scratch using a separate mesh object with dark material"""
        # Select random position on can
        angle = random.uniform(0, 2 * math.pi)
        height = random.uniform(-self.can_dimensions['height'] * 0.3, 
                               self.can_dimensions['height'] * 0.3)
        
        # Calculate position on can surface
        x = self.can_dimensions['radius'] * 0.99 * math.cos(angle)
        y = self.can_dimensions['radius'] * 0.99 * math.sin(angle)
        z = self.can_object.location.z + height
        
        center = Vector((x, y, z))
        
        # Scratch parameters - make them bigger and more visible
        length = random.uniform(0.2, 0.4)
        width = random.uniform(0.02, 0.05)
        depth = 0.01  # Slight offset from surface
        scratch_angle = random.uniform(0, math.pi)
        
        # Create a plane for the scratch
        bpy.ops.mesh.primitive_plane_add()
        scratch = bpy.context.active_object
        scratch.name = "Scratch"
        
        # Scale and position the scratch
        scratch.scale = (width, length, 1)
        scratch.location = (x * 1.01, y * 1.01, z)  # Slightly outside can surface
        
        # Rotate to align with can surface and add scratch angle
        scratch.rotation_euler = (0, math.pi/2, angle + scratch_angle)
        
        # Create scratch material - very dark with some metallic shine
        scratch_mat = bpy.data.materials.new(name="ScratchMaterial")
        scratch_mat.use_nodes = True
        nodes = scratch_mat.node_tree.nodes
        nodes.clear()
        
        output = nodes.new(type='ShaderNodeOutputMaterial')
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        
        # Dark gray/black color for scratch
        bsdf.inputs['Base Color'].default_value = (0.05, 0.05, 0.05, 1.0)
        bsdf.inputs['Metallic'].default_value = 0.8
        bsdf.inputs['Roughness'].default_value = 0.7
        
        scratch_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        # Apply material
        scratch.data.materials.clear()
        scratch.data.materials.append(scratch_mat)
        
        # Store defect info for bounding box
        self.defect_regions.append(('scratch', center, max(length, width)/2))
        
        # Also deform the mesh slightly for realism
        self.deform_mesh_for_scratch(center, length, width, scratch_angle)
        
        return center
    
    def deform_mesh_for_scratch(self, center, length, width, angle):
        """Deform the mesh to create a visible groove for the scratch"""
        # Enter edit mode
        bpy.context.view_layer.objects.active = self.can_object
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Get bmesh
        bm = bmesh.from_edit_mesh(self.can_object.data)
        bm.verts.ensure_lookup_table()
        
        # Deform vertices
        for vert in bm.verts:
            vert_world = self.can_object.matrix_world @ vert.co
            
            # Calculate distance along scratch line
            to_vert = vert_world - center
            along_scratch = to_vert.dot(Vector((math.cos(angle), math.sin(angle), 0)))
            across_scratch = to_vert.dot(Vector((-math.sin(angle), math.cos(angle), 0)))
            
            # Check if vertex is within scratch bounds
            if abs(along_scratch) < length/2 and abs(across_scratch) < width/2:
                # Create scratch profile (deeper in center)
                profile = 1.0 - (abs(across_scratch) / (width/2))
                displacement = -0.005 * profile  # Visible depth
                
                # Displace vertex inward
                normal = vert.normal
                vert.co += normal * displacement
        
        # Update mesh
        bmesh.update_edit_mesh(self.can_object.data)
        bpy.ops.object.mode_set(mode='OBJECT')
    
    def create_dent_defect(self):
        """Create a visible dent with dark overlay"""
        # Select random position
        angle = random.uniform(0, 2 * math.pi)
        height = random.uniform(-self.can_dimensions['height'] * 0.3, 
                               self.can_dimensions['height'] * 0.3)
        
        x = self.can_dimensions['radius'] * 0.95 * math.cos(angle)
        y = self.can_dimensions['radius'] * 0.95 * math.sin(angle)
        z = self.can_object.location.z + height
        
        center = Vector((x, y, z))
        
        # Dent parameters - make them bigger
        radius = random.uniform(0.1, 0.2)
        depth = random.uniform(0.02, 0.05)
        
        # Create a flattened sphere for the dent overlay
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8)
        dent = bpy.context.active_object
        dent.name = "Dent"
        
        # Scale and position
        dent.scale = (radius, radius, radius * 0.3)
        dent.location = (x * 0.99, y * 0.99, z)  # Slightly inside can surface
        
        # Rotate to align with can surface
        dent.rotation_euler = (0, math.pi/2, angle)
        
        # Create dent material - darker and less reflective
        dent_mat = bpy.data.materials.new(name="DentMaterial")
        dent_mat.use_nodes = True
        nodes = dent_mat.node_tree.nodes
        nodes.clear()
        
        output = nodes.new(type='ShaderNodeOutputMaterial')
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        
        # Dark color with reduced metallic
        bsdf.inputs['Base Color'].default_value = (0.15, 0.15, 0.15, 1.0)
        bsdf.inputs['Metallic'].default_value = 0.3
        bsdf.inputs['Roughness'].default_value = 0.6
        
        dent_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        # Apply material
        dent.data.materials.clear()
        dent.data.materials.append(dent_mat)
        
        # Deform the actual mesh
        self.deform_mesh_for_dent(center, radius, depth)
        
        # Store defect info
        self.defect_regions.append(('dent', center, radius))
        
        return center
    
    def deform_mesh_for_dent(self, center, radius, depth):
        """Deform mesh to create visible dent"""
        # Enter edit mode
        bpy.context.view_layer.objects.active = self.can_object
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Get bmesh
        bm = bmesh.from_edit_mesh(self.can_object.data)
        bm.verts.ensure_lookup_table()
        
        # Deform vertices
        for vert in bm.verts:
            vert_world = self.can_object.matrix_world @ vert.co
            distance = (vert_world - center).length
            
            if distance < radius:
                # Smooth falloff for dent
                factor = 1.0 - (distance / radius) ** 2
                displacement = -depth * factor
                
                # Displace vertex inward
                normal = vert.normal
                vert.co += normal * displacement
        
        # Update mesh
        bmesh.update_edit_mesh(self.can_object.data)
        bpy.ops.object.mode_set(mode='OBJECT')
    
    def create_puncture_defect(self):
        """Create a visible puncture/hole with black overlay"""
        # Select random position
        angle = random.uniform(0, 2 * math.pi)
        height = random.uniform(-self.can_dimensions['height'] * 0.3, 
                               self.can_dimensions['height'] * 0.3)
        
        x = self.can_dimensions['radius'] * 0.95 * math.cos(angle)
        y = self.can_dimensions['radius'] * 0.95 * math.sin(angle)
        z = self.can_object.location.z + height
        
        center = Vector((x, y, z))
        
        # Puncture parameters
        radius = random.uniform(0.03, 0.06)
        
        # Create a cylinder for the puncture hole
        bpy.ops.mesh.primitive_cylinder_add(vertices=8)
        puncture = bpy.context.active_object
        puncture.name = "Puncture"
        
        # Scale and position
        puncture.scale = (radius, radius, 0.02)
        puncture.location = (x, y, z)
        
        # Rotate to point into can
        puncture.rotation_euler = (0, math.pi/2, angle)
        
        # Create very dark material for hole effect
        puncture_mat = bpy.data.materials.new(name="PunctureMaterial")
        puncture_mat.use_nodes = True
        nodes = puncture_mat.node_tree.nodes
        nodes.clear()
        
        output = nodes.new(type='ShaderNodeOutputMaterial')
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        
        # Pure black, non-reflective
        bsdf.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 1.0
        
        puncture_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        # Apply material
        puncture.data.materials.clear()
        puncture.data.materials.append(puncture_mat)
        
        # Also create deep deformation
        self.deform_mesh_for_puncture(center, radius)
        
        # Store defect info
        self.defect_regions.append(('puncture', center, radius))
        
        return center
    
    def deform_mesh_for_puncture(self, center, radius):
        """Create deep deformation for puncture"""
        # Enter edit mode
        bpy.context.view_layer.objects.active = self.can_object
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Get bmesh
        bm = bmesh.from_edit_mesh(self.can_object.data)
        bm.verts.ensure_lookup_table()
        
        # Create deep indentation
        for vert in bm.verts:
            vert_world = self.can_object.matrix_world @ vert.co
            distance = (vert_world - center).length
            
            if distance < radius:
                # Sharp falloff for puncture
                factor = 1.0 - (distance / radius)
                displacement = -0.08 * factor  # Very deep
                
                normal = vert.normal
                vert.co += normal * displacement
        
        # Update mesh
        bmesh.update_edit_mesh(self.can_object.data)
        bpy.ops.object.mode_set(mode='OBJECT')
    
    def clear_defects(self):
        """Clear all defects and reset mesh"""
        # Remove defect overlay objects
        for obj in bpy.data.objects:
            if obj.name in ["Scratch", "Dent", "Puncture"]:
                bpy.data.objects.remove(obj, do_unlink=True)
        
        self.defect_regions = []
        self.reset_mesh()
        self.reset_materials()
    
    def apply_scratch_material_effect(self, center, length, width, angle):
        """Modify material to show scratch effect"""
        # Since we're using overlay objects, this is now optional
        # We can still modify the base material slightly for added realism
        pass
    
    def apply_dent_material_effect(self, center, radius):
        """Modify material for dented area"""
        # Since we're using overlay objects, this is now optional
        pass
    
    def apply_puncture_material_effect(self, center, radius):
        """Apply very dark material for puncture"""
        # Since we're using overlay objects, this is now optional
        pass
    
    def clear_defects(self):
        """Clear all defects and reset mesh"""
        self.defect_regions = []
        self.reset_mesh()
        self.reset_materials()
    
    def reset_materials(self):
        """Reset all materials to original state"""
        for mat_idx, original_mat in self.original_materials.items():
            if mat_idx < len(self.can_object.material_slots):
                new_mat = original_mat.copy()
                self.can_object.material_slots[mat_idx].material = new_mat
    
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
    
    def is_object_in_camera_view(self):
        """Check if can is visible in camera view"""
        import bpy_extras.object_utils
        
        scene = bpy.context.scene
        
        # Check can center
        center_point = self.can_object.matrix_world @ Vector((0, 0, 0))
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, self.camera, center_point)
        
        if co_2d.z > 0:
            margin = 0.1
            if margin < co_2d.x < (1 - margin) and margin < co_2d.y < (1 - margin):
                return True
        
        return False
    
    def find_valid_camera_position(self, max_attempts=50):
        """Find a camera position where can and defects are visible"""
        for attempt in range(max_attempts):
            distance = random.uniform(1.5, 3.2)
            azimuth = random.uniform(-math.pi/2, math.pi/2)
            elevation = random.uniform(0.2, 1.5)
            
            cam_x = distance * math.sin(azimuth)
            cam_y = -distance * math.cos(azimuth)
            cam_z = elevation
            
            self.camera.location = (cam_x, cam_y, cam_z)
            
            # Point at can
            can_center = Vector(self.can_object.location)
            direction = can_center - Vector(self.camera.location)
            rot_quat = direction.to_track_quat('-Z', 'Y')
            self.camera.rotation_euler = rot_quat.to_euler()
            
            if self.is_object_in_camera_view():
                return True
        
        # Fallback
        self.camera.location = (0, -2.5, 0.8)
        can_center = Vector(self.can_object.location)
        direction = can_center - Vector(self.camera.location)
        rot_quat = direction.to_track_quat('-Z', 'Y')
        self.camera.rotation_euler = rot_quat.to_euler()
        self.camera.data.lens = 35
        return True
    
    def randomize_scene(self):
        """Randomize scene"""
        # Can position
        self.can_object.location = (
            random.uniform(-0.3, 0.3),
            random.uniform(-0.3, 0.3),
            random.uniform(-0.1, 0.2)
        )
        
        # Can rotation
        self.can_object.rotation_euler = (
            random.uniform(-math.radians(30), math.radians(30)),
            random.uniform(-math.radians(30), math.radians(30)),
            random.uniform(0, 2 * math.pi)
        )
        
        self.find_valid_camera_position()
        
        self.camera.data.lens = random.uniform(45, 55)
        
        # Lighting variation
        self.key_light.data.energy = random.uniform(3.5, 4.5)
        self.fill_light.data.energy = random.uniform(2.0, 3.0)
        self.rim_light.data.energy = random.uniform(1.5, 2.5)
        self.top_light.data.energy = random.uniform(1.0, 2.0)
        
        # Slight rotation for lights
        self.key_light.rotation_euler = (
            random.uniform(0.3, 0.6),
            random.uniform(-0.3, 0.3),
            0
        )
    
    def calculate_bounding_box_for_object(self, obj=None):
        """Calculate YOLO bounding box for the entire object"""
        try:
            import bpy_extras.object_utils
            
            if obj is None:
                obj = self.can_object
            
            # Get mesh bounding box corners
            bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            
            # Project to 2D
            scene = bpy.context.scene
            coords_2d = []
            
            for corner in bbox_corners:
                co_2d = bpy_extras.object_utils.world_to_camera_view(scene, self.camera, corner)
                if co_2d.z > 0:  # Only consider points in front of camera
                    coords_2d.append((co_2d.x, 1.0 - co_2d.y))
            
            if len(coords_2d) < 4:
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
            
            # Validate
            min_box_size = 0.05
            if width < min_box_size or height < min_box_size:
                return None
            
            if center_x < 0.05 or center_x > 0.95 or center_y < 0.05 or center_y > 0.95:
                return None
            
            return center_x, center_y, width, height
            
        except Exception as e:
            print(f"Bounding box calculation error: {e}")
            return None
    
    def calculate_defect_bounding_box(self, defect_type, center_3d, radius):
        """Calculate bounding box for a defect given its 3D center and radius"""
        try:
            import bpy_extras.object_utils
            
            scene = bpy.context.scene
            
            # Create points around the defect area
            points_3d = []
            
            # Add center point
            points_3d.append(center_3d)
            
            # Add points in a circle around the defect
            num_points = 8
            for i in range(num_points):
                angle = (2 * math.pi * i) / num_points
                
                # For scratches, create an ellipse
                if defect_type == 'scratch':
                    x_offset = radius * 2 * math.cos(angle)  # Longer in one direction
                    y_offset = radius * 0.5 * math.sin(angle)  # Narrower
                else:
                    x_offset = radius * math.cos(angle)
                    y_offset = radius * math.sin(angle)
                
                point = center_3d + Vector((x_offset, y_offset, 0))
                points_3d.append(point)
            
            # Project all points to 2D
            coords_2d = []
            for point in points_3d:
                co_2d = bpy_extras.object_utils.world_to_camera_view(scene, self.camera, point)
                if co_2d.z > 0:  # In front of camera
                    coords_2d.append((co_2d.x, 1.0 - co_2d.y))
            
            if len(coords_2d) < 4:
                return None
            
            # Calculate bounds
            x_coords = [coord[0] for coord in coords_2d]
            y_coords = [coord[1] for coord in coords_2d]
            
            min_x = max(0, min(x_coords))
            max_x = min(1, max(x_coords))
            min_y = max(0, min(y_coords))
            max_y = min(1, max(y_coords))
            
            # Add some padding to ensure defect is fully contained
            padding = 0.01
            min_x = max(0, min_x - padding)
            max_x = min(1, max_x + padding)
            min_y = max(0, min_y - padding)
            max_y = min(1, max_y + padding)
            
            # YOLO format
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            width = max_x - min_x
            height = max_y - min_y
            
            # Minimum size for defects
            min_defect_size = 0.02
            if width < min_defect_size or height < min_defect_size:
                return None
            
            return center_x, center_y, width, height
            
        except Exception as e:
            print(f"Defect bbox calculation error: {e}")
            return None
    
    def generate_defects(self, num_defects):
        """Generate random defects on the can"""
        defect_types = ['scratch', 'dent', 'puncture']
        weights = [0.4, 0.4, 0.2]  # Scratches and dents more common
        
        for _ in range(num_defects):
            defect_type = random.choices(defect_types, weights=weights)[0]
            
            if defect_type == 'scratch':
                self.create_scratch_defect()
            elif defect_type == 'dent':
                self.create_dent_defect()
            elif defect_type == 'puncture':
                self.create_puncture_defect()
    
    def generate_single_image(self, image_index):
        """Generate a single image with defects"""
        max_retries = 5
        
        for retry in range(max_retries):
            try:
                # Clear previous defects
                self.clear_defects()
                
                # Decide on number of defects
                defect_count = random.choices([0, 1, 2, 3], weights=[0.2, 0.4, 0.3, 0.1])[0]
                
                # Generate defects
                if defect_count > 0:
                    self.generate_defects(defect_count)
                
                # Randomize scene
                self.randomize_scene()
                
                # Verify can is visible
                if not self.is_object_in_camera_view():
                    print(f"Retry {retry + 1}: Can not visible, adjusting...")
                    continue
                
                # Set background
                if self.background_images and random.random() < 0.7:
                    bg_image = random.choice(self.background_images)
                    self.setup_background(bg_image)
                else:
                    self.setup_background()
                
                # Calculate can bounding box
                can_bbox = self.calculate_bounding_box_for_object()
                if can_bbox is None:
                    print(f"Retry {retry + 1}: Invalid can bounding box, adjusting...")
                    continue
                
                # Calculate defect bounding boxes
                defect_bboxes = []
                for defect_type, center, radius in self.defect_regions:
                    bbox = self.calculate_defect_bounding_box(defect_type, center, radius)
                    if bbox:
                        defect_bboxes.append((defect_type, bbox))
                
                # Render
                image_filename = f"can_{image_index:05d}.png"
                image_path = self.images_dir / image_filename
                
                bpy.context.scene.render.filepath = str(image_path)
                bpy.ops.render.render(write_still=True)
                
                # Verify render
                if not image_path.exists() or image_path.stat().st_size < 10000:
                    print(f"Warning: Render may have failed for image {image_index}")
                    continue
                
                # Save labels
                label_filename = f"can_{image_index:05d}.txt"
                label_path = self.labels_dir / label_filename
                
                with open(label_path, 'w') as f:
                    # Write can bounding box (class 0)
                    can_cx, can_cy, can_w, can_h = can_bbox
                    f.write(f"0 {can_cx:.6f} {can_cy:.6f} {can_w:.6f} {can_h:.6f}\n")
                    
                    # Write defect bounding boxes
                    for defect_type, (cx, cy, w, h) in defect_bboxes:
                        class_id = self.class_map[defect_type]
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                
                # Log success
                defect_summary = f"{len(defect_bboxes)} defects" if defect_bboxes else "pristine"
                if defect_bboxes:
                    types = [d[0] for d in defect_bboxes]
                    defect_summary += f" ({', '.join(types)})"
                
                print(f"Image {image_index}: {defect_summary}")
                
                if retry > 0:
                    print(f"  Success after {retry + 1} attempts")
                
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
        print("Classes: 0=can, 1=scratch, 2=dent, 3=puncture")
        
        success_count = 0
        defect_stats = {
            'pristine': 0,
            'scratch': 0,
            'dent': 0,
            'puncture': 0,
            'total_defects': 0
        }
        
        for i in range(batch_size):
            if self.generate_single_image(start_index + i):
                success_count += 1
                
                # Update stats
                if len(self.defect_regions) == 0:
                    defect_stats['pristine'] += 1
                else:
                    for defect_type, _, _ in self.defect_regions:
                        defect_stats[defect_type] += 1
                        defect_stats['total_defects'] += 1
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{batch_size} (Success: {success_count})")
        
        print(f"\nBatch complete: {success_count}/{batch_size} images")
        print("Defect statistics:")
        print(f"  Pristine cans: {defect_stats['pristine']}")
        print(f"  Total defects: {defect_stats['total_defects']}")
        print(f"  - Scratches: {defect_stats['scratch']}")
        print(f"  - Dents: {defect_stats['dent']}")
        print(f"  - Punctures: {defect_stats['puncture']}")
        
        return success_count
    
    def create_classes_file(self):
        """Create classes.txt file for YOLO training"""
        classes_file = self.output_dir / "classes.txt"
        with open(classes_file, 'w') as f:
            f.write("can\n")
            f.write("scratch\n")
            f.write("dent\n")
            f.write("puncture\n")
        print(f"Created classes file: {classes_file}")
    
    def create_data_yaml(self):
        """Create data.yaml file for YOLO training"""
        data_yaml = self.output_dir / "data.yaml"
        
        # Get absolute paths
        train_path = (self.output_dir / "images").absolute()
        val_path = train_path  # Using same for now, should split in practice
        
        yaml_content = f"""# YOLOv8 Coke Can Defect Detection Dataset

# Train/val/test sets
train: {train_path}
val: {val_path}

# Classes
nc: 4  # number of classes
names: ['can', 'scratch', 'dent', 'puncture']

# Dataset information
description: Synthetic Coca-Cola can defect detection dataset with realistic defects
"""
        
        with open(data_yaml, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created data.yaml file: {data_yaml}")

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
    parser.add_argument("--defect_textures_dir", type=str, default=None,
                        help="Directory containing defect texture images (optional)")
    
    return parser.parse_args(argv)

def main():
    """Main execution"""
    try:
        args = parse_arguments()
        
        print("=== Coca-Cola Can Defect Detection Dataset Generator ===")
        print(f"Model: {args.fbx_path}")
        print(f"Output: {args.output_dir}")
        print(f"Batch: {args.start_index} to {args.start_index + args.batch_size - 1}")
        
        generator = ProfessionalCanDefectGenerator(
            model_path=args.fbx_path,
            texture_path=args.texture_path,
            backgrounds_dir=args.backgrounds_dir,
            output_dir=args.output_dir,
            defect_textures_dir=args.defect_textures_dir
        )
        
        # Create dataset files
        generator.create_classes_file()
        generator.create_data_yaml()
        
        success_count = generator.generate_batch(args.start_index, args.batch_size)
        
        print(f"\n✓ Generated {success_count}/{args.batch_size} images")
        print("Classes: 0=can, 1=scratch, 2=dent, 3=puncture")
        print(f"\nDataset ready for YOLO training in: {args.output_dir}")
        print("Use the data.yaml file for training configuration")
        
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
