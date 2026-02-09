"""
Performance Comparison: Local vs Docker Training

This script runs training both locally and in Docker to compare performance.
"""

import subprocess
import time
import json
from pathlib import Path


def run_local_training():
    """Run training locally and measure time."""
    print("=" * 60)
    print("RUNNING LOCAL TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', 'src/train.py'],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        elapsed_time = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return {
            'success': result.returncode == 0,
            'time_seconds': elapsed_time,
            'time_minutes': elapsed_time / 60
        }
    
    except subprocess.TimeoutExpired:
        print("Training timed out after 1 hour")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def run_docker_training():
    """Run training in Docker and measure time."""
    print("\n" + "=" * 60)
    print("RUNNING DOCKER TRAINING")
    print("=" * 60)
    
    # Build image first
    print("Building Docker image...")
    subprocess.run(['docker-compose', 'build', 'training'], check=True)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['docker-compose', 'up', 'training'],
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        elapsed_time = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        # Clean up
        subprocess.run(['docker-compose', 'down'], capture_output=True)
        
        return {
            'success': result.returncode == 0,
            'time_seconds': elapsed_time,
            'time_minutes': elapsed_time / 60
        }
    
    except subprocess.TimeoutExpired:
        print("Training timed out after 1 hour")
        subprocess.run(['docker-compose', 'down'], capture_output=True)
        return None
    except Exception as e:
        print(f"Error: {e}")
        subprocess.run(['docker-compose', 'down'], capture_output=True)
        return None


def compare_results(local_result, docker_result):
    """Compare and display results."""
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    if local_result:
        print(f"\nLocal Training:")
        print(f"  Time: {local_result['time_minutes']:.2f} minutes")
        print(f"  Status: {'✓ Success' if local_result['success'] else '✗ Failed'}")
    
    if docker_result:
        print(f"\nDocker Training:")
        print(f"  Time: {docker_result['time_minutes']:.2f} minutes")
        print(f"  Status: {'✓ Success' if docker_result['success'] else '✗ Failed'}")
    
    if local_result and docker_result:
        time_diff = abs(local_result['time_minutes'] - docker_result['time_minutes'])
        faster = "Local" if local_result['time_minutes'] < docker_result['time_minutes'] else "Docker"
        
        print(f"\nTime Difference: {time_diff:.2f} minutes")
        print(f"Faster: {faster}")
        
        # Load metrics if available
        metrics_path = Path('results/metrics.json')
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            print(f"\nModel Performance:")
            print(f"  Best Val Accuracy: {metrics['best_val_accuracy']:.2f}%")
            print(f"  Test Accuracy: {metrics['test_accuracy']:.2f}%")
            print(f"  Total Parameters: {metrics['total_parameters']:,}")
    
    # Save comparison
    comparison = {
        'local': local_result,
        'docker': docker_result,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results/comparison.json', 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print("\nComparison saved to results/comparison.json")


def main():
    print("=" * 60)
    print("TRAINING COMPARISON: LOCAL vs DOCKER")
    print("=" * 60)
    print()
    print("This script will:")
    print("1. Run training locally")
    print("2. Run training in Docker")
    print("3. Compare performance")
    print()
    print("WARNING: This will take ~70-80 minutes total")
    print("=" * 60)
    
    response = input("\nProceed? (y/n): ").lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Check if data exists
    if not Path('data/train').exists():
        print("\nError: Dataset not found!")
        print("Please run setup_dataset.py first")
        return
    
    # Run comparisons
    local_result = run_local_training()
    docker_result = run_docker_training()
    
    # Compare
    compare_results(local_result, docker_result)
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
