"""Test script to verify config.py is working correctly"""

import config

print("\n" + "=" * 60)
print("CONFIG.PY VERIFICATION")
print("=" * 60)

print("\n[PATHS]")
print("Base path: " + config.basePath)
print("Output path: " + config.pathOut)
print("GEE Project: " + config.GEE_PROJECT_ID)

print("\n[SINGLE IMAGE PROCESSING]")
print("Current image: " + config.CURRENT_IMAGE)
img_config = config.get_current_image_config()
print("Radiance output: " + img_config["nameOut_radiance"])
print("Reflectance output: " + img_config["nameOut_reflectance"])

print("\n[INUNDATION MAPPING]")
print("Number of image pairs: " + str(len(config.INUNDATION_IMAGE_PAIRS)))
inund_config = config.get_inundation_config(0)
print("Pair 1 Early: " + inund_config["image1_id"])
print("Pair 1 Late:  " + inund_config["image2_id"])
print("Threshold: " + str(inund_config["threshold"]))

print("\n[PROCESSING OPTIONS]")
print("Destriping method: " + config.destripingMethod)
print("Local destriping: " + str(config.localDestriping))
print("Topographic correction: " + str(config.use_topo))
print("DEM ID: " + config.demID)
print("Post-processing: " + str(config.run_postprocessing))

print("\n" + "=" * 60)
print("CONFIG VERIFICATION COMPLETE!")
print("=" * 60)
