# Danish Job Data Translation Pipeline

This repository contains tools to download, unzip, and translate Danish job posting data to English using AI models.

## Overview

The pipeline consists of three main steps:
1. **Download** - Fetch the Danish job data from the source
2. **Unzip** - Extract the compressed data files
3. **Translate** - Convert Danish text to English using AI translation

## Prerequisites

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster translation)
- At least 8GB RAM (16GB+ recommended)
- 10GB+ free disk space

### Python Dependencies
Install the required packages:

```bash
# Create and activate virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 1: Download Danish Job Data

### Option A: Using the Download Script
```bash
# Download the latest Danish job data
python download_danish_jobs.py

# Or specify a custom output directory
python download_danish_jobs.py --output /path/to/downloads
```

### Option B: Manual Download
1. Visit the Danish job data source
2. Download the compressed file (usually `.zip` or `.tar.gz`)
3. Place it in your working directory

## Step 2: Unzip the Data

### Option A: Using the Unzip Script
```bash
# Unzip the downloaded file
python unzip_danish_jobs.py

# Or specify custom input/output directories
python unzip_danish_jobs.py --input downloaded_file.zip --output /c/denmark/unzipped
```

### Option B: Manual Extraction
```bash
# For ZIP files
unzip downloaded_file.zip -d /c/denmark/unzipped

# For TAR.GZ files
tar -xzf downloaded_file.tar.gz -C /c/denmark/unzipped
```

## Step 3: Translate Danish to English

### Basic Usage
```bash
# Translate using GPU (recommended)
python translate_unzipped_eurollm.py --root /c/denmark/unzipped --fields BODY TITLE_RAW --device-map cuda:0

# Translate using CPU (slower but works on any machine)
python translate_unzipped_eurollm.py --root /c/denmark/unzipped --fields BODY TITLE_RAW --device-map cpu
```

### Advanced Options

```bash
# Custom model and settings
python translate_unzipped_eurollm.py \
    --root /c/denmark/unzipped \
    --fields BODY TITLE_RAW DESCRIPTION \
    --model "Helsinki-NLP/opus-mt-da-en" \
    --device-map cuda:0 \
    --torch-dtype bfloat16 \
    --rate-limit 0.1

# Process only specific files
python translate_unzipped_eurollm.py \
    --root /c/denmark/unzipped/Denmark_2018_10_postings \
    --fields BODY TITLE_RAW \
    --device-map cuda:0
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--root` | Root directory to scan for CSV files | `/c/denmark/unzipped` |
| `--fields` | CSV columns to translate | `["BODY", "TITLE_RAW"]` |
| `--out-suffix` | Suffix for output files | `.en` |
| `--model` | Translation model to use | `Helsinki-NLP/opus-mt-da-en` |
| `--device-map` | Device to use (cuda:0, cpu, auto) | `auto` |
| `--torch-dtype` | Data type (bfloat16, float16, float32) | `auto` |
| `--rate-limit` | Delay between translations (seconds) | `0.0` |
| `--local-files-only` | Use only local model files | `False` |

## Output

The translation script will:
- Scan for CSV files recursively in the specified directory
- Detect Danish text using character and word-based heuristics
- Translate only Danish segments to English
- Preserve English text, URLs, and formatting
- Create new files with `.en` suffix

### Example Output Structure
```
/c/denmark/unzipped/
├── Denmark_2018_10_postings/
│   ├── Denmark_2018_10_postings.csv          # Original file
│   └── Denmark_2018_10_postings.en.csv       # Translated file
├── Denmark_2018_11_postings/
│   ├── Denmark_2018_11_postings.csv
│   └── Denmark_2018_11_postings.en.csv
└── ...
```

## Performance Tips

### GPU Optimization
- Use `--device-map cuda:0` for GPU acceleration
- Use `--torch-dtype bfloat16` for memory efficiency
- The script automatically uses batch processing for better GPU utilization

### Memory Management
- For large files, consider processing in smaller batches
- Use `--rate-limit 0.1` to add delays between translations
- Monitor GPU memory usage with `nvidia-smi`

### Speed Optimization
- Use GPU instead of CPU (10-50x faster)
- Process multiple files in parallel if you have multiple GPUs
- Use the batch processing feature (enabled by default)

## Troubleshooting

### Common Issues

**GPU Memory Errors**
```bash
# Reduce batch size or use CPU
python translate_unzipped_eurollm.py --device-map cpu
```

**Model Download Issues**
```bash
# Use local files only
python translate_unzipped_eurollm.py --local-files-only
```

**Translation Quality Issues**
```bash
# Try a different model
python translate_unzipped_eurollm.py --model "microsoft/DialoGPT-medium"
```

### Error Messages

- **"CUDA out of memory"**: Use CPU or reduce batch size
- **"Model not found"**: Check internet connection or use `--local-files-only`
- **"Translation error"**: Text might be too long, check chunking

## Model Information

### Default Model: Helsinki-NLP/opus-mt-da-en
- **Type**: Marian MT (Machine Translation)
- **Size**: ~300MB
- **Languages**: Danish → English
- **Quality**: High quality for job descriptions
- **Speed**: Fast inference

### Alternative Models
- `microsoft/DialoGPT-medium`: General purpose, larger model
- `utter-project/EuroLLM-9B-Instruct`: Large language model (requires more memory)

## File Structure

```
/c/denmark/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── download_danish_jobs.py            # Download script
├── unzip_danish_jobs.py               # Unzip script
├── translate_unzipped_eurollm.py      # Translation script
├── unzipped/                          # Extracted data
│   └── [job posting files]
└── .venv/                             # Virtual environment
```

## Example Workflow

```bash
# 1. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download data
python download_danish_jobs.py

# 3. Unzip data
python unzip_danish_jobs.py

# 4. Translate to English
python translate_unzipped_eurollm.py \
    --root /c/denmark/unzipped \
    --fields BODY TITLE_RAW \
    --device-map cuda:0

# 5. Check results
ls -la /c/denmark/unzipped/*.en.csv
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your Python environment and dependencies
3. Check GPU drivers and CUDA installation
4. Review the error messages for specific guidance

## License

This project is for educational and research purposes. Please respect the terms of use for the Danish job data source.


