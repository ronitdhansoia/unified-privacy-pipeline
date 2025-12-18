#!/usr/bin/env python3
"""
Dataset verification script.

Checks if datasets are downloaded correctly and provides statistics.
"""

import os
import sys
import pandas as pd
from pathlib import Path

def check_face_dataset():
    """Check LFW face dataset."""
    print("\n" + "="*60)
    print("Face Recognition Dataset (LFW)")
    print("="*60)

    lfw_path = Path("datasets/face_recognition/lfw")

    if not lfw_path.exists():
        print("[FAIL] LFW dataset not found")
        print("   Run: ./scripts/download_datasets.sh")
        return False

    # Count images and people
    image_files = list(lfw_path.rglob("*.jpg"))
    people_dirs = [d for d in lfw_path.iterdir() if d.is_dir()]

    print(f"[PASS] LFW dataset found")
    print(f"   Location: {lfw_path}")
    print(f"   People: {len(people_dirs)} individuals")
    print(f"   Images: {len(image_files)} face images")

    # Sample some people
    if people_dirs:
        print(f"\n   Sample people:")
        for person in sorted(people_dirs)[:5]:
            num_images = len(list(person.glob("*.jpg")))
            print(f"   - {person.name}: {num_images} images")

    return True


def check_health_datasets():
    """Check health prediction datasets."""
    print("\n" + "="*60)
    print("Health Prediction Datasets")
    print("="*60)

    health_path = Path("datasets/health_prediction")
    success = True

    # Check Heart Disease dataset
    heart_file = health_path / "cleveland_heart.csv"
    if heart_file.exists():
        print(f"\n[PASS] Heart Disease Dataset")
        print(f"   Location: {heart_file}")

        # Try to load and show info
        try:
            df = pd.read_csv(heart_file, header=None)
            print(f"   Samples: {len(df)} patients")
            print(f"   Features: {len(df.columns)} clinical features")
            print(f"   Size: {heart_file.stat().st_size / 1024:.1f} KB")
        except Exception as e:
            print(f"   Warning: Could not parse file: {e}")
    else:
        print(f"[FAIL] Heart Disease Dataset not found")
        success = False

    # Check Diabetes dataset
    diabetes_file = health_path / "diabetes.csv"
    if diabetes_file.exists():
        print(f"\n[PASS] Diabetes Dataset")
        print(f"   Location: {diabetes_file}")

        try:
            df = pd.read_csv(diabetes_file, header=None)
            print(f"   Samples: {len(df)} patients")
            print(f"   Features: {len(df.columns)} clinical features")
            print(f"   Size: {diabetes_file.stat().st_size / 1024:.1f} KB")
        except Exception as e:
            print(f"   Warning: Could not parse file: {e}")
    else:
        print(f"[FAIL] Diabetes Dataset not found")
        success = False

    return success


def calculate_total_size():
    """Calculate total dataset size."""
    print("\n" + "="*60)
    print("Storage Summary")
    print("="*60)

    datasets_path = Path("datasets")

    if not datasets_path.exists():
        print("[FAIL] Datasets directory not found")
        return

    # Calculate sizes
    face_size = sum(f.stat().st_size for f in Path("datasets/face_recognition").rglob("*") if f.is_file())
    health_size = sum(f.stat().st_size for f in Path("datasets/health_prediction").rglob("*") if f.is_file())
    total_size = face_size + health_size

    print(f"\nFace Recognition: {face_size / (1024**2):.1f} MB")
    print(f"Health Prediction: {health_size / (1024**2):.1f} MB")
    print(f"Total: {total_size / (1024**2):.1f} MB")


def main():
    print("="*60)
    print("Dataset Verification")
    print("Unified Privacy Pipeline")
    print("="*60)

    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    print(f"\nProject root: {project_root}")

    # Check datasets
    face_ok = check_face_dataset()
    health_ok = check_health_datasets()

    # Calculate sizes
    calculate_total_size()

    # Final summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)

    if face_ok and health_ok:
        print("\n[PASS] All datasets verified successfully!")
        print("\nYou can now:")
        print("  1. Run demos with real data")
        print("  2. Train privacy-preserving models")
        print("  3. Test unlearning on real datasets")
        return 0
    else:
        print("\n[WARNING] Some datasets are missing")
        print("\nTo download missing datasets:")
        print("  ./scripts/download_datasets.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
