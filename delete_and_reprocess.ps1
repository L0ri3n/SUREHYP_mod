# PowerShell script to delete old reflectance and reprocess

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "DELETING OLD REFLECTANCE FILES TO FORCE REPROCESSING" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$OUTDIR = "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT"
$FNAME = "EO1H2020342013284110KF"

Write-Host "Deleting reflectance files..." -ForegroundColor Yellow
Get-ChildItem "$OUTDIR\$FNAME`_reflectance.*" -ErrorAction SilentlyContinue | Remove-Item -Force
Write-Host "  ✓ Deleted reflectance files" -ForegroundColor Green

Write-Host "Deleting derived products..." -ForegroundColor Yellow
Remove-Item "$OUTDIR\$FNAME`_NDVI.npy" -ErrorAction SilentlyContinue
Remove-Item "$OUTDIR\$FNAME`_statistics.txt" -ErrorAction SilentlyContinue
Remove-Item "$OUTDIR\$FNAME`_valid_pixels_mask.npy" -ErrorAction SilentlyContinue
Remove-Item "$OUTDIR\$FNAME`_reflectance_good_bands_mask.npy" -ErrorAction SilentlyContinue
Write-Host "  ✓ Deleted derived products" -ForegroundColor Green

Write-Host "Deleting quicklooks..." -ForegroundColor Yellow
Get-ChildItem "$OUTDIR\quicklooks\$FNAME`_*.*" -ErrorAction SilentlyContinue | Remove-Item -Force
Write-Host "  ✓ Deleted quicklooks" -ForegroundColor Green

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "NOW RUNNING REPROCESSING WITH FIXES" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"

Write-Host "Activating conda environment and running process_hyperion.py..." -ForegroundColor Yellow
Write-Host ""

# Run Python script with conda
& conda run -n hyperion_roger python process_hyperion.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "REPROCESSING COMPLETE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "  1. Check the console output above for the bad band masking messages"
Write-Host "  2. Run: python check_reflectance_values.py"
Write-Host "  3. Look at new spectra: OUT\quicklooks\$FNAME`_spectra.png"
Write-Host ""
Read-Host "Press Enter to exit"
