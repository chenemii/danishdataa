#!/usr/bin/env python3
import zipfile
import os
from pathlib import Path

DOWNLOADS_DIR = "./downloads"   # folder with your .zip files
OUTPUT_DIR = "./unzipped"       # where extracted files will go

# Make sure output folder exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Get list of .zip files
zip_files = [f for f in os.listdir(DOWNLOADS_DIR) if f.lower().endswith(".zip")]

if not zip_files:
    print("No .zip files found in", DOWNLOADS_DIR)
else:
    print(f"Found {len(zip_files)} zip files.")

    for zfile in zip_files:
        zip_path = os.path.join(DOWNLOADS_DIR, zfile)
        print(f"Extracting {zfile} ...")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract into subfolder with the zip's name (without .zip)
                extract_folder = os.path.join(OUTPUT_DIR, Path(zfile).stem)
                Path(extract_folder).mkdir(parents=True, exist_ok=True)
                zip_ref.extractall(extract_folder)
        except zipfile.BadZipFile:
            print(f"❌ Skipping {zfile} — not a valid zip file.")

    print("✅ All done!")

