@echo off
echo ============================================================
echo DELETING OLD REFLECTANCE FILES TO FORCE REPROCESSING
echo ============================================================
echo.

set OUTDIR=C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT
set FNAME=EO1H2020342013284110KF

echo Deleting reflectance files...
del "%OUTDIR%\%FNAME%_reflectance.*" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   ✓ Deleted reflectance files
) else (
    echo   - No reflectance files found or already deleted
)

echo Deleting derived products...
del "%OUTDIR%\%FNAME%_NDVI.npy" 2>nul
del "%OUTDIR%\%FNAME%_statistics.txt" 2>nul
del "%OUTDIR%\%FNAME%_valid_pixels_mask.npy" 2>nul
del "%OUTDIR%\%FNAME%_reflectance_good_bands_mask.npy" 2>nul
echo   ✓ Deleted derived products

echo Deleting quicklooks...
del "%OUTDIR%\quicklooks\%FNAME%_*.*" 2>nul
echo   ✓ Deleted quicklooks

echo.
echo ============================================================
echo NOW RUNNING REPROCESSING WITH FIXES
echo ============================================================
echo.

cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"

echo Activating conda environment and running process_hyperion.py...
call conda activate hyperion_roger && python process_hyperion.py

echo.
echo ============================================================
echo REPROCESSING COMPLETE
echo ============================================================
echo.
echo Next steps:
echo   1. Check the console output above for the bad band masking messages
echo   2. Run: python check_reflectance_values.py
echo   3. Look at new spectra: OUT\quicklooks\%FNAME%_spectra.png
echo.
pause
