"""
Quick setup and run script for the Recommendation System
"""
import os
import sys
import subprocess


def setup_and_run():
    """Setup virtual environment and run recommendation system"""
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists('venv'):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
    
    # Determine the pip command based on OS
    if sys.platform == 'win32':
        pip_cmd = 'venv\\Scripts\\pip'
        python_cmd = 'venv\\Scripts\\python'
    else:
        pip_cmd = 'venv/bin/pip'
        python_cmd = 'venv/bin/python'
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([pip_cmd, 'install', '-r', 'requirements.txt'], check=True)
    
    # Get user input
    print("\n" + "="*60)
    print("Recommendation System Setup Complete!")
    print("="*60)
    
    csv_file = input("\nEnter your CSV file name (e.g., train_v3.csv): ").strip()
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found!")
        return
    
    model = input("Choose model [MF/BPR/NMF, default: MF]: ").strip().upper() or "MF"
    num_items = input("Number of recommendations per user [default: 50]: ").strip()
    num_items = int(num_items) if num_items.isdigit() else 50
    
    # Run the recommendation system
    print(f"\nRunning recommendation system...")
    print(f"  • CSV: {csv_file}")
    print(f"  • Model: {model}")
    print(f"  • Top-N: {num_items}")
    print("="*60 + "\n")
    
    subprocess.run([python_cmd, 'main.py', csv_file, model, str(num_items)], check=True)
    
    # Output file info
    output_file = csv_file.replace('.csv', '_recommendations.txt').replace('.txt', '_recommendations.txt')
    print(f"\n✓ Recommendations saved to: {output_file}")


if __name__ == '__main__':
    try:
        setup_and_run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
