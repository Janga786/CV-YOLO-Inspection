#!/usr/bin/env python3
"""
Comprehensive Dataset Analyzer for Synthetic Training Data
Validates quality, analyzes distribution, and provides optimization recommendations
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import argparse
from datetime import datetime
import pandas as pd

class DatasetAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"
        self.metadata_dir = self.dataset_path / "metadata"
        self.analysis_dir = self.dataset_path / "analysis"
        
        # Create analysis directory
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Initialize analysis data
        self.image_stats = []
        self.label_stats = []
        self.quality_issues = []
        self.metadata_stats = []
        
    def analyze_complete_dataset(self):
        """Perform comprehensive dataset analysis"""
        print("🔍 Starting comprehensive dataset analysis...")
        
        # Basic file validation
        self.validate_file_structure()
        
        # Image quality analysis
        self.analyze_image_quality()
        
        # Label analysis
        self.analyze_labels()
        
        # Metadata analysis (if available)
        if self.metadata_dir.exists():
            self.analyze_metadata()
        
        # Distribution analysis
        self.analyze_distributions()
        
        # Quality validation
        self.validate_quality_standards()
        
        # Generate comprehensive report
        self.generate_analysis_report()
        
        print("✅ Dataset analysis complete!")
    
    def validate_file_structure(self):
        """Validate basic file structure and pairing"""
        print("📁 Validating file structure...")
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
        
        # Get image and label files
        image_files = list(self.images_dir.glob("*.png")) + list(self.images_dir.glob("*.jpg"))
        label_files = list(self.labels_dir.glob("*.txt"))
        
        print(f"Found {len(image_files)} images and {len(label_files)} labels")
        
        # Check pairing
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}
        
        unpaired_images = image_stems - label_stems
        unpaired_labels = label_stems - image_stems
        
        if unpaired_images:
            print(f"⚠️  {len(unpaired_images)} images without labels")
            self.quality_issues.extend([
                {'type': 'unpaired_image', 'file': f"{stem}.png"} 
                for stem in unpaired_images
            ])
        
        if unpaired_labels:
            print(f"⚠️  {len(unpaired_labels)} labels without images")
            self.quality_issues.extend([
                {'type': 'unpaired_label', 'file': f"{stem}.txt"} 
                for stem in unpaired_labels
            ])
        
        self.paired_files = image_stems & label_stems
        print(f"✅ {len(self.paired_files)} properly paired files")
    
    def analyze_image_quality(self):
        """Analyze image quality metrics"""
        print("🖼️  Analyzing image quality...")
        
        for stem in self.paired_files:
            image_path = self.images_dir / f"{stem}.png"
            if not image_path.exists():
                image_path = self.images_dir / f"{stem}.jpg"
            
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                self.quality_issues.append({
                    'type': 'corrupted_image',
                    'file': image_path.name,
                    'issue': 'Cannot load image'
                })
                continue
            
            # Calculate quality metrics
            stats = self.calculate_image_metrics(img, image_path)
            stats['filename'] = image_path.name
            self.image_stats.append(stats)
        
        print(f"✅ Analyzed {len(self.image_stats)} images")
    
    def calculate_image_metrics(self, img, image_path):
        """Calculate comprehensive image quality metrics"""
        # Basic metrics
        height, width, channels = img.shape
        file_size = image_path.stat().st_size
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness statistics
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Contrast (standard deviation)
        contrast = brightness_std
        
        # Noise estimation (using high-frequency components)
        noise_estimate = self.estimate_noise(gray)
        
        # Color distribution
        color_stats = self.analyze_color_distribution(img)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'file_size': file_size,
            'sharpness': sharpness,
            'brightness_mean': brightness_mean,
            'brightness_std': brightness_std,
            'contrast': contrast,
            'noise_estimate': noise_estimate,
            'edge_density': edge_density,
            **color_stats
        }
    
    def estimate_noise(self, gray_img):
        """Estimate noise level in image"""
        # Use wavelet-based noise estimation
        kernel = np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ])
        
        convolved = cv2.filter2D(gray_img.astype(np.float64), -1, kernel)
        sigma = np.sqrt(np.pi/2) * np.mean(np.abs(convolved))
        return sigma
    
    def analyze_color_distribution(self, img):
        """Analyze color distribution in image"""
        # Calculate histogram statistics for each channel
        stats = {}
        for i, channel in enumerate(['blue', 'green', 'red']):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            stats[f'{channel}_mean'] = np.mean(img[:, :, i])
            stats[f'{channel}_std'] = np.std(img[:, :, i])
            stats[f'{channel}_entropy'] = self.calculate_entropy(hist)
        
        # Overall color diversity
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])
        stats['hue_entropy'] = self.calculate_entropy(hue_hist)
        
        return stats
    
    def calculate_entropy(self, hist):
        """Calculate entropy of histogram"""
        hist = hist.flatten()
        hist = hist[hist > 0]  # Remove zeros
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        return entropy
    
    def analyze_labels(self):
        """Analyze YOLO label files"""
        print("🏷️  Analyzing labels...")
        
        for stem in self.paired_files:
            label_path = self.labels_dir / f"{stem}.txt"
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Parse labels
                labels = []
                for line_num, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        self.quality_issues.append({
                            'type': 'malformed_label',
                            'file': label_path.name,
                            'line': line_num + 1,
                            'issue': f'Expected 5 values, got {len(parts)}'
                        })
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate ranges
                        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 
                               0 < width <= 1 and 0 < height <= 1):
                            self.quality_issues.append({
                                'type': 'invalid_bbox',
                                'file': label_path.name,
                                'line': line_num + 1,
                                'issue': 'Bounding box values out of range'
                            })
                            continue
                        
                        labels.append({
                            'class_id': class_id,
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': width,
                            'height': height,
                            'area': width * height
                        })
                        
                    except ValueError as e:
                        self.quality_issues.append({
                            'type': 'invalid_label_format',
                            'file': label_path.name,
                            'line': line_num + 1,
                            'issue': str(e)
                        })
                
                self.label_stats.append({
                    'filename': label_path.name,
                    'num_objects': len(labels),
                    'labels': labels
                })
                
            except Exception as e:
                self.quality_issues.append({
                    'type': 'label_read_error',
                    'file': label_path.name,
                    'issue': str(e)
                })
        
        print(f"✅ Analyzed {len(self.label_stats)} label files")
    
    def analyze_metadata(self):
        """Analyze metadata files if available"""
        print("📊 Analyzing metadata...")
        
        metadata_files = list(self.metadata_dir.glob("*.json"))
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.metadata_stats.append(metadata)
                
            except Exception as e:
                self.quality_issues.append({
                    'type': 'metadata_read_error',
                    'file': metadata_file.name,
                    'issue': str(e)
                })
        
        print(f"✅ Analyzed {len(self.metadata_stats)} metadata files")
    
    def analyze_distributions(self):
        """Analyze data distributions"""
        print("📈 Analyzing distributions...")
        
        # Class distribution
        self.class_distribution = Counter()
        self.bbox_sizes = []
        self.bbox_positions = []
        self.bbox_aspect_ratios = []
        
        for label_stat in self.label_stats:
            for label in label_stat['labels']:
                self.class_distribution[label['class_id']] += 1
                self.bbox_sizes.append(label['area'])
                self.bbox_positions.append((label['center_x'], label['center_y']))
                self.bbox_aspect_ratios.append(label['width'] / label['height'])
        
        # Image quality distributions
        if self.image_stats:
            self.quality_distributions = {
                'sharpness': [stat['sharpness'] for stat in self.image_stats],
                'brightness': [stat['brightness_mean'] for stat in self.image_stats],
                'contrast': [stat['contrast'] for stat in self.image_stats],
                'noise': [stat['noise_estimate'] for stat in self.image_stats],
                'edge_density': [stat['edge_density'] for stat in self.image_stats]
            }
    
    def validate_quality_standards(self):
        """Validate against quality standards"""
        print("✅ Validating quality standards...")
        
        # Define quality thresholds
        quality_thresholds = {
            'min_sharpness': 50.0,
            'max_noise': 10.0,
            'min_contrast': 20.0,
            'min_edge_density': 0.01,
            'min_bbox_area': 0.01,
            'max_bbox_area': 0.8
        }
        
        quality_failures = []
        
        # Check image quality
        for stat in self.image_stats:
            filename = stat['filename']
            
            if stat['sharpness'] < quality_thresholds['min_sharpness']:
                quality_failures.append({
                    'type': 'low_sharpness',
                    'file': filename,
                    'value': stat['sharpness'],
                    'threshold': quality_thresholds['min_sharpness']
                })
            
            if stat['noise_estimate'] > quality_thresholds['max_noise']:
                quality_failures.append({
                    'type': 'high_noise',
                    'file': filename,
                    'value': stat['noise_estimate'],
                    'threshold': quality_thresholds['max_noise']
                })
            
            if stat['contrast'] < quality_thresholds['min_contrast']:
                quality_failures.append({
                    'type': 'low_contrast',
                    'file': filename,
                    'value': stat['contrast'],
                    'threshold': quality_thresholds['min_contrast']
                })
        
        # Check bounding box quality
        for label_stat in self.label_stats:
            filename = label_stat['filename']
            
            for i, label in enumerate(label_stat['labels']):
                if label['area'] < quality_thresholds['min_bbox_area']:
                    quality_failures.append({
                        'type': 'bbox_too_small',
                        'file': filename,
                        'object': i,
                        'value': label['area'],
                        'threshold': quality_thresholds['min_bbox_area']
                    })
                
                if label['area'] > quality_thresholds['max_bbox_area']:
                    quality_failures.append({
                        'type': 'bbox_too_large',
                        'file': filename,
                        'object': i,
                        'value': label['area'],
                        'threshold': quality_thresholds['max_bbox_area']
                    })
        
        self.quality_failures = quality_failures
        print(f"Found {len(quality_failures)} quality standard violations")
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        print("📊 Generating analysis report...")
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate text report
        self.create_text_report()
        
        # Generate JSON summary
        self.create_json_summary()
        
        print(f"📁 Reports saved to: {self.analysis_dir}")
    
    def create_visualizations(self):
        """Create analysis visualizations"""
        plt.style.use('seaborn-v0_8')
        
        # Class distribution
        if self.class_distribution:
            plt.figure(figsize=(10, 6))
            classes = list(self.class_distribution.keys())
            counts = list(self.class_distribution.values())
            
            plt.bar(classes, counts, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
            plt.title('Class Distribution in Dataset')
            plt.xlabel('Class ID')
            plt.ylabel('Number of Objects')
            plt.grid(True, alpha=0.3)
            
            # Add percentage labels
            total = sum(counts)
            for i, (cls, count) in enumerate(zip(classes, counts)):
                percentage = (count / total) * 100
                plt.text(cls, count + max(counts) * 0.01, f'{percentage:.1f}%', 
                        ha='center', va='bottom')
            
            plt.savefig(self.analysis_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Bounding box size distribution
        if self.bbox_sizes:
            plt.figure(figsize=(12, 8))
            
            # Size histogram
            plt.subplot(2, 2, 1)
            plt.hist(self.bbox_sizes, bins=50, alpha=0.7, color='#2E86AB')
            plt.title('Bounding Box Area Distribution')
            plt.xlabel('Area (normalized)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Aspect ratio distribution
            plt.subplot(2, 2, 2)
            plt.hist(self.bbox_aspect_ratios, bins=50, alpha=0.7, color='#A23B72')
            plt.title('Bounding Box Aspect Ratio Distribution')
            plt.xlabel('Width/Height Ratio')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Position heatmap
            plt.subplot(2, 2, 3)
            if len(self.bbox_positions) > 10:
                x_coords = [pos[0] for pos in self.bbox_positions]
                y_coords = [pos[1] for pos in self.bbox_positions]
                plt.hexbin(x_coords, y_coords, gridsize=20, cmap='Blues')
                plt.colorbar(label='Object Count')
            plt.title('Object Position Heatmap')
            plt.xlabel('Center X (normalized)')
            plt.ylabel('Center Y (normalized)')
            
            # Box plot of sizes by class
            plt.subplot(2, 2, 4)
            if len(set(label['class_id'] for label_stat in self.label_stats for label in label_stat['labels'])) > 1:
                size_by_class = defaultdict(list)
                for label_stat in self.label_stats:
                    for label in label_stat['labels']:
                        size_by_class[label['class_id']].append(label['area'])
                
                classes = list(size_by_class.keys())
                sizes = [size_by_class[cls] for cls in classes]
                plt.boxplot(sizes, labels=classes)
                plt.title('Bounding Box Size by Class')
                plt.xlabel('Class ID')
                plt.ylabel('Area (normalized)')
            
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'bbox_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Image quality metrics
        if self.image_stats:
            plt.figure(figsize=(15, 10))
            
            metrics = ['sharpness', 'brightness_mean', 'contrast', 'noise_estimate', 'edge_density']
            for i, metric in enumerate(metrics):
                plt.subplot(2, 3, i + 1)
                values = [stat[metric] for stat in self.image_stats]
                plt.hist(values, bins=30, alpha=0.7, color=plt.cm.Set1(i))
                plt.title(f'{metric.replace("_", " ").title()} Distribution')
                plt.xlabel(metric.replace('_', ' ').title())
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
            
            # Quality correlation matrix
            plt.subplot(2, 3, 6)
            quality_df = pd.DataFrame(self.image_stats)[metrics]
            correlation_matrix = quality_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Quality Metrics Correlation')
            
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'quality_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_text_report(self):
        """Create comprehensive text report"""
        report_path = self.analysis_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE DATASET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Path: {self.dataset_path}\n\n")
            
            # File Structure Summary
            f.write("FILE STRUCTURE SUMMARY\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total paired files: {len(self.paired_files)}\n")
            f.write(f"Quality issues found: {len(self.quality_issues)}\n\n")
            
            # Class Distribution
            if self.class_distribution:
                f.write("CLASS DISTRIBUTION\n")
                f.write("-" * 18 + "\n")
                total_objects = sum(self.class_distribution.values())
                for class_id, count in sorted(self.class_distribution.items()):
                    percentage = (count / total_objects) * 100
                    f.write(f"Class {class_id}: {count} objects ({percentage:.1f}%)\n")
                f.write(f"Total objects: {total_objects}\n\n")
            
            # Image Quality Summary
            if self.image_stats:
                f.write("IMAGE QUALITY SUMMARY\n")
                f.write("-" * 21 + "\n")
                
                quality_metrics = ['sharpness', 'brightness_mean', 'contrast', 'noise_estimate']
                for metric in quality_metrics:
                    values = [stat[metric] for stat in self.image_stats]
                    f.write(f"{metric.replace('_', ' ').title()}:\n")
                    f.write(f"  Mean: {np.mean(values):.2f}\n")
                    f.write(f"  Std:  {np.std(values):.2f}\n")
                    f.write(f"  Min:  {np.min(values):.2f}\n")
                    f.write(f"  Max:  {np.max(values):.2f}\n\n")
            
            # Bounding Box Analysis
            if self.bbox_sizes:
                f.write("BOUNDING BOX ANALYSIS\n")
                f.write("-" * 21 + "\n")
                f.write(f"Area statistics:\n")
                f.write(f"  Mean: {np.mean(self.bbox_sizes):.4f}\n")
                f.write(f"  Std:  {np.std(self.bbox_sizes):.4f}\n")
                f.write(f"  Min:  {np.min(self.bbox_sizes):.4f}\n")
                f.write(f"  Max:  {np.max(self.bbox_sizes):.4f}\n\n")
                
                f.write(f"Aspect ratio statistics:\n")
                f.write(f"  Mean: {np.mean(self.bbox_aspect_ratios):.2f}\n")
                f.write(f"  Std:  {np.std(self.bbox_aspect_ratios):.2f}\n")
                f.write(f"  Min:  {np.min(self.bbox_aspect_ratios):.2f}\n")
                f.write(f"  Max:  {np.max(self.bbox_aspect_ratios):.2f}\n\n")
            
            # Quality Issues
            if self.quality_issues:
                f.write("QUALITY ISSUES\n")
                f.write("-" * 14 + "\n")
                issue_types = Counter(issue['type'] for issue in self.quality_issues)
                for issue_type, count in issue_types.most_common():
                    f.write(f"{issue_type}: {count} issues\n")
                f.write("\n")
            
            # Quality Standard Violations
            if hasattr(self, 'quality_failures') and self.quality_failures:
                f.write("QUALITY STANDARD VIOLATIONS\n")
                f.write("-" * 28 + "\n")
                violation_types = Counter(failure['type'] for failure in self.quality_failures)
                for violation_type, count in violation_types.most_common():
                    f.write(f"{violation_type}: {count} violations\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            # Class balance recommendations
            if self.class_distribution:
                class_counts = list(self.class_distribution.values())
                if max(class_counts) / min(class_counts) > 2:
                    f.write("• Dataset is imbalanced. Consider generating more samples of underrepresented classes.\n")
            
            # Quality recommendations
            if hasattr(self, 'quality_failures'):
                if any(f['type'] == 'low_sharpness' for f in self.quality_failures):
                    f.write("• Some images have low sharpness. Consider improving rendering quality or camera focus.\n")
                
                if any(f['type'] == 'high_noise' for f in self.quality_failures):
                    f.write("• Some images have high noise. Consider enabling denoising or using more render samples.\n")
                
                if any(f['type'].startswith('bbox_') for f in self.quality_failures):
                    f.write("• Some bounding boxes are outside acceptable size ranges. Review camera positioning.\n")
            
            # General recommendations
            if len(self.paired_files) < 1000:
                f.write("• Consider generating more data for better model training.\n")
            
            f.write("\nAnalysis complete. Review visualizations for detailed insights.\n")
    
    def create_json_summary(self):
        """Create JSON summary for programmatic access"""
        summary = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'dataset_path': str(self.dataset_path),
                'analyzer_version': '1.0'
            },
            'dataset_summary': {
                'total_paired_files': len(self.paired_files),
                'total_quality_issues': len(self.quality_issues),
                'class_distribution': dict(self.class_distribution) if self.class_distribution else {},
                'total_objects': sum(self.class_distribution.values()) if self.class_distribution else 0
            },
            'quality_metrics': {},
            'recommendations': []
        }
        
        # Add quality metrics summary
        if self.image_stats:
            quality_metrics = ['sharpness', 'brightness_mean', 'contrast', 'noise_estimate']
            for metric in quality_metrics:
                values = [stat[metric] for stat in self.image_stats]
                summary['quality_metrics'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # Add bounding box summary
        if self.bbox_sizes:
            summary['bbox_analysis'] = {
                'area_stats': {
                    'mean': float(np.mean(self.bbox_sizes)),
                    'std': float(np.std(self.bbox_sizes)),
                    'min': float(np.min(self.bbox_sizes)),
                    'max': float(np.max(self.bbox_sizes))
                },
                'aspect_ratio_stats': {
                    'mean': float(np.mean(self.bbox_aspect_ratios)),
                    'std': float(np.std(self.bbox_aspect_ratios)),
                    'min': float(np.min(self.bbox_aspect_ratios)),
                    'max': float(np.max(self.bbox_aspect_ratios))
                }
            }
        
        # Save summary
        summary_path = self.analysis_dir / 'analysis_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Comprehensive Dataset Analyzer")
    parser.add_argument("dataset_path", help="Path to dataset directory")
    parser.add_argument("--output-dir", help="Output directory for analysis (default: dataset_path/analysis)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(args.dataset_path)
    
    # Override output directory if specified
    if args.output_dir:
        analyzer.analysis_dir = Path(args.output_dir)
        analyzer.analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    try:
        analyzer.analyze_complete_dataset()
        print(f"\n✅ Analysis complete! Check {analyzer.analysis_dir} for results.")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())