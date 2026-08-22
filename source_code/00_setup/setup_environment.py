"""
Environment setup verification script.
Run this first to confirm all libraries are correctly installed.
"""

import sys
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable."""
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
        import importlib.metadata
        try:
            version = importlib.metadata.version(package_name)
            print(f"  [OK] {package_name} == {version}")
        except Exception:
            print(f"  [OK] {package_name} (version unknown)")
        return True
    except ImportError:
        print(f"  [MISSING] {package_name} — install with: pip install {package_name}")
        return False

def main():
    print("=" * 60)
    print("ECOSYSTEM COLLAPSE EWS — Environment Check")
    print("=" * 60)

    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("xarray", "xarray"),
        ("netCDF4", "netCDF4"),
        ("geopandas", "geopandas"),
        ("shapely", "shapely"),
        ("rasterio", "rasterio"),
        ("pyproj", "pyproj"),
        ("regionmask", "regionmask"),
        ("cartopy", "cartopy"),
        ("rioxarray", "rioxarray"),
        ("earthaccess", "earthaccess"),
        ("cdsapi", "cdsapi"),
        ("tqdm", "tqdm"),
        ("ewstools", "ewstools"),
        ("tigramite", "tigramite"),
        ("scikit-learn", "sklearn"),
        ("statsmodels", "statsmodels"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]

    print("\nChecking required packages:")
    missing = []
    for pkg, imp in packages:
        if not check_package(pkg, imp):
            missing.append(pkg)

    print("\n" + "=" * 60)
    if not missing:
        print("All packages installed. Ready to proceed.")
    else:
        print(f"Missing {len(missing)} package(s). Run:")
        print(f"  pip install {' '.join(missing)}")
    print("=" * 60)

    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 9):
        print("WARNING: Python 3.9+ recommended.")
    else:
        print("Python version OK.")

if __name__ == "__main__":
    main()
