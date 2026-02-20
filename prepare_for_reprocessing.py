"""
Prepare for full reprocessing by deleting all intermediate and output files.
This forces process_hyperion.py to rerun both Step 1 (radiance preprocessing)
and Step 2 (atmospheric correction) from scratch.

Run: python prepare_for_reprocessing.py
"""

from pathlib import Path

OUTDIR = Path(r"C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT")
FNAME = "EO1H2020342013284110KF"

print("=" * 80)
print("PREPARING FOR FULL REPROCESSING")
print("=" * 80)
print()

# Preprocessed radiance files (Step 1 output)
radiance_files = list(OUTDIR.glob(f"{FNAME}_preprocessed.*"))
radiance_files += list(OUTDIR.glob(f"{FNAME}_preprocessed.hdr.backup"))

# Reflectance image files (Step 2 output)
reflectance_files = list(OUTDIR.glob(f"{FNAME}_reflectance.*"))
reflectance_files += list(OUTDIR.glob(f"{FNAME}_reflectance.hdr.backup"))

# Mask files created during atmospheric correction
mask_files = [
    OUTDIR / f"{FNAME}_reflectance_clearview_mask.npy",
    OUTDIR / f"{FNAME}_reflectance_cloud_mask.npy",
    OUTDIR / f"{FNAME}_reflectance_shadows_mask.npy",
    OUTDIR / f"{FNAME}_reflectance_cirrus_mask.npy",
    OUTDIR / f"{FNAME}_reflectance_good_bands_mask.npy",
]

# Spectral info files (wavelength backups created by fix_envi_hdr_for_snap)
spectral_info_files = list(OUTDIR.glob(f"{FNAME}*_spectral_info.txt"))

# Derived products
derived_files = [
    OUTDIR / f"{FNAME}_NDVI.npy",
    OUTDIR / f"{FNAME}_statistics.txt",
    OUTDIR / f"{FNAME}_valid_pixels_mask.npy",
]

# Quicklooks
quicklook_dir = OUTDIR / "quicklooks"
quicklook_files = list(quicklook_dir.glob(f"{FNAME}_*.*")) if quicklook_dir.exists() else []

all_files = (
    radiance_files +
    reflectance_files +
    [f for f in mask_files if f.exists()] +
    spectral_info_files +
    [f for f in derived_files if f.exists()] +
    quicklook_files
)

if not all_files:
    print("No old files found - ready for fresh processing!")
    print()
    print("You can now run:")
    print("  conda activate hyperion_roger")
    print("  python process_hyperion.py")
    print()
else:
    print(f"Found {len(all_files)} files to delete:")
    print()

    # Show counts by category
    if radiance_files:
        print(f"  Preprocessed radiance files: {len(radiance_files)}")
    if reflectance_files:
        print(f"  Reflectance files:           {len(reflectance_files)}")
    mask_count = len([f for f in mask_files if f.exists()])
    if mask_count:
        print(f"  Mask files:                  {mask_count}")
    if spectral_info_files:
        print(f"  Spectral info files:         {len(spectral_info_files)}")
    derived_count = len([f for f in derived_files if f.exists()])
    if derived_count:
        print(f"  Derived products:            {derived_count}")
    if quicklook_files:
        print(f"  Quicklook files:             {len(quicklook_files)}")

    print()
    print("Files to delete:")
    for f in all_files[:20]:
        print(f"  - {f.name}")
    if len(all_files) > 20:
        print(f"  ... and {len(all_files)-20} more")

    print()
    response = input("Delete these files? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        deleted = 0
        for f in all_files:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                print(f"  [!] Could not delete {f.name}: {e}")

        print()
        print(f"Deleted {deleted} files successfully!")
        print()
        print("=" * 80)
        print("NOW RUN REPROCESSING:")
        print("=" * 80)
        print()
        print("  conda activate hyperion_roger")
        print("  python process_hyperion.py")
        print()
    else:
        print()
        print("Cancelled - no files deleted.")
        print()
        print("WARNING: If you run process_hyperion.py now,")
        print("  existing files will cause steps to be skipped!")
        print()

print("=" * 80)
