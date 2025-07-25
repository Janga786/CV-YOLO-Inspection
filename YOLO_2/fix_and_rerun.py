import os
import json

# Restore the original generator script
os.system('cp enhanced_generator_v282.py.bak enhanced_generator_v282.py')

# Create a known-good configuration
config = {
    'render': {
        'resolution': [1024, 1024],
        'samples': 64,
        'engine': 'BLENDER_EEVEE',
        'denoising': True
    },
    'camera': {
        'lens_range': [50, 50],
        'distance_range': [2.5, 2.5],
        'height_range': [1.0, 1.0],
        'angle_range': [0, 0]
    },
    'lighting': {
        'key_light_strength': [8.0, 8.0],
        'fill_light_strength': [4.0, 4.0],
        'rim_light_strength': [6.0, 6.0]
    },
    'defects': {
        'pristine_probability': 1.0,
        'scratch_probability': 0.0,
        'dent_probability': 0.0,
        'puncture_probability': 0.0,
        'defect_intensity_range': [0.0, 0.0]
    },
    'quality': {
        'min_can_area': 0.01,
        'max_can_area': 0.95
    }
}

with open('known_good_config.json', 'w') as f:
    json.dump(config, f, indent=4)

# Run the test runner with the known-good configuration
os.system('python3 enhanced_test_runner.py --config known_good_config.json')
