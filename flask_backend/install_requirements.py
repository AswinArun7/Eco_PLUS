import subprocess
import sys
import os

def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f"Successfully installed {package}")

def main():
    print("Installing required packages for EcoSense Flask Backend...")
    
    # List of required packages
    required_packages = [
        "flask",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "requests"
    ]
    
    # Install each package
    for package in required_packages:
        try:
            install_package(package)
        except Exception as e:
            print(f"Error installing {package}: {e}")
    
    print("\nAll packages installed successfully!")
    print("\nNow you can run the Flask app with:")
    print("python app.py")

if __name__ == "__main__":
    main() 