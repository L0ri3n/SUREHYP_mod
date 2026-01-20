"""
Prepare for reprocessing by deleting old reflectance files.
This forces the script to rerun atmospheric correction with the new fixes.

Run: python prepare_for_reprocessing.py
"""

import os
import glob
from pathlib import Path

OUTDIR = Path(r"C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT")
FNAME = "EO1H2020342013284110KF"

print("=" * 80)
print("PREPARING FOR REPROCESSING")
print("=" * 80)
print()

# Check if files exist
reflectance_files = list(OUTDIR.glob(f"{FNAME}_reflectance.*"))
derived_files = [
    OUTDIR / f"{FNAME}_NDVI.npy",
    OUTDIR / f"{FNAME}_statistics.txt",
    OUTDIR / f"{FNAME}_valid_pixels_mask.npy",
    OUTDIR / f"{FNAME}_reflectance_good_bands_mask.npy",
]
quicklook_files = list((OUTDIR / "quicklooks").glob(f"{FNAME}_*.*"))

all_files = reflectance_files + [f for f in derived_files if f.exists()] + quicklook_files

if not all_files:
    print("✅ No old files found - ready for fresh processing!")
    print()
    print("You can now run:")
    print("  conda activate hyperion_roger")
    print("  python process_hyperion.py")
    print()
else:
    print(f"Found {len(all_files)} files to delete:")
    print()

    for f in all_files[:10]:
        print(f"  - {f.name}")
    if len(all_files) > 10:
        print(f"  ... and {len(all_files)-10} more")

    print()
    response = input("Delete these files? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        deleted = 0
        for f in all_files:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                print(f"  ⚠️  Could not delete {f.name}: {e}")

        print()
        print(f"✅ Deleted {deleted} files successfully!")
        print()
        print("=" * 80)
        print("NOW RUN REPROCESSING:")
        print("=" * 80)
        print()
        print("  conda activate hyperion_roger")
        print("  python process_hyperion.py")
        print()
        print("LOOK FOR THIS MESSAGE during processing:")
        print('  "⚠️  WARNING: Found X bands with extreme max values"')
        print()
        print("If you see it, the fix is working!")
        print()
    else:
        print()
        print("Cancelled - no files deleted.")
        print()
        print("⚠️  WARNING: If you run process_hyperion.py now,")
        print("   it will skip atmospheric correction and use the old files!")
        print()

print("=" * 80)
