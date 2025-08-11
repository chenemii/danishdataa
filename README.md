# Danish Translation Pipeline

A comprehensive pipeline for translating Danish job descriptions to English using transformer models with intelligent model type detection, GPU optimization, and resume functionality.

## Overview

This project provides a robust solution for translating Danish text to English, specifically designed for job description datasets. It intelligently handles different model types (causal LMs like EuroLLM and seq2seq models like Helsinki-NLP) and automatically optimizes processing for maximum efficiency.

## Key Features

- **🤖 Intelligent Model Type Detection**: Automatically detects causal LMs (EuroLLM, Llama, etc.) vs seq2seq models
- **⚡ GPU-Optimized Batch Processing**: Uses dataset-based batching for maximum GPU efficiency
- **🔄 Resume Functionality**: Skip already processed files when restarting interrupted jobs
- **📏 Automatic Text Length Detection**: Uses tokenizer to estimate text length and determine optimal processing strategy
- **🔀 Model Separation**: Routes short texts (≤512 tokens) to small model, long texts (>512 tokens) to large model
- **🏢 Two-Phase Distributed Processing**: Support for processing across multiple machines
- **⚙️ Flexible Model Configuration**: Can use different models for small and large texts
- **🚀 Dynamic Batch Sizing**: Automatically adjusts batch sizes based on available GPU memory
- **📁 Smart File Filtering**: Automatically skips output files (.en, .large, .small) from previous runs

## Processing Modes

### 1. Single Machine Processing (Default)
Process all files on one machine with both models loaded simultaneously.

### 2. Two-Phase Distributed Processing
Split processing across two machines for better resource utilization:
- **Phase 1**: Local machine processes short texts, saves long texts to separate CSV
- **Phase 2**: Remote machine processes long texts with large model
- **Merge**: Combine results back into single CSV

## Quick Start

### Single Machine Processing

```bash
# Basic usage with default models
python3 translate_unzipped_eurollm.py --root /path/to/csv/files

# With resume functionality (skip already processed files)
python3 translate_unzipped_eurollm.py --root /path/to/csv/files --resume

# Advanced usage with custom models and GPU optimization
python3 translate_unzipped_eurollm.py \
    --root /path/to/csv/files \
    --model-small "Helsinki-NLP/opus-mt-da-en" \
    --model-large "eurollm/eurollm-7b" \
    --max-tokens 512 \
    --device-map auto \
    --torch-dtype bfloat16 \
    --resume

# CPU-only processing (no GPU required)
python3 translate_unzipped_eurollm.py \
    --root /path/to/csv/files \
    --device-map cpu \
    --resume
```

### Two-Phase Distributed Processing

#### Phase 1: Local Processing (Small Model)
```bash
python3 translate_unzipped_eurollm.py \
    --root /path/to/csv/files \
    --phase 1 \
    --model-small "Helsinki-NLP/opus-mt-da-en" \
    --max-tokens 512 \
    --large-csv-suffix ".large" \
    --resume
```

#### Phase 2: Remote Processing (Large Model)
```bash
python3 translate_unzipped_eurollm.py \
    --phase 2 \
    --remote-input /path/to/file.large.csv \
    --model-large "eurollm/eurollm-7b" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --resume
```

#### Merge Results
```bash
python merge_phases.py \
    --phase1 /path/to/file.en.csv \
    --phase2 /path/to/file.large.en.csv \
    --output /path/to/file.final.csv \
    --original /path/to/original/file.csv
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers library
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Install required packages
pip install transformers torch accelerate safetensors datasets

# For GPU acceleration with quantization (NVIDIA GPUs only)
pip install bitsandbytes

# Clone the repository
git clone <repository-url>
cd danish-translation-pipeline
```

## Configuration

### Environment Variables

- `HF_MODEL_SMALL`: Default small model (Helsinki-NLP/opus-mt-da-en)
- `HF_MODEL_LARGE`: Default large model (eurollm/eurollm-7b)

### Command Line Options

#### General Options
- `--root`: Root folder to scan for CSV files (default: /c/denmark/unzipped)
- `--fields`: CSV columns to translate (default: ["BODY", "TITLE_RAW"])
- `--out-suffix`: Suffix for output files (default: .en)
- `--rate-limit`: Sleep seconds after each field translation (default: 0.0)
- `--max-tokens`: Maximum tokens for small model (default: 512)

#### Model Options
- `--model-small`: Model ID for texts ≤512 tokens
- `--model-large`: Model ID for texts >512 tokens
- `--device-map`: Device for model loading (auto, cpu, cuda)
- `--load-in-4bit`: Load model in 4-bit quantization
- `--torch-dtype`: Torch dtype (auto, bfloat16, float16, float32)
- `--local-files-only`: Use only local model files

#### Two-Phase Options
- `--phase`: Processing phase (1=local only, 2=remote only, both=single machine)
- `--large-csv-suffix`: Suffix for CSV containing large rows (default: .large)
- `--remote-input`: Input CSV for Phase 2 (remote processing)

#### Resume and Optimization Options
- `--resume`: Skip processing files that already have complete output (resume mode)

#### Supported Model Types
The script automatically detects and supports:
- **Seq2Seq Models**: Helsinki-NLP/opus-mt-da-en, facebook/nllb-*, etc.
- **Causal Language Models**: eurollm/eurollm-7b, meta-llama/*, mistralai/*, etc.

## Model Recommendations

### Small Model (≤512 tokens)
- **Helsinki-NLP/opus-mt-da-en**: Fast, reliable for short texts
- **Helsinki-NLP/opus-mt-da-en**: Good balance of speed and quality

### Large Model (>512 tokens)
- **eurollm/eurollm-7b**: EuroLLM causal LM for better handling of long texts and context
- **facebook/nllb-200-distilled-600M**: Alternative seq2seq model for long texts
- **meta-llama/Llama-2-7b-chat-hf**: Llama model with instruction following
- **Custom fine-tuned models**: For domain-specific translations

### GPU Memory Requirements
- **4-8GB VRAM**: Small models (Helsinki-NLP, smaller NLLB)
- **12-16GB VRAM**: Medium models (eurollm-7b with 4-bit quantization)
- **24GB+ VRAM**: Large models (eurollm-7b full precision)
- **CPU only**: All models supported with longer processing time

## File Structure

```
danish-translation-pipeline/
├── translate_unzipped_eurollm.py  # Main translation script
├── merge_phases.py                # Merge script for two-phase results
├── test_model_separation.py       # Test script for functionality
├── README.md                      # This file
├── README_model_separation.md     # Detailed documentation
└── requirements.txt               # Python dependencies
```

## Example Workflow

### Input CSV Structure
```csv
ID,BODY,TITLE_RAW
1,"Dette er en kort dansk tekst.","Kort titel"
2,"Dette er en meget længere dansk tekst...","Lang titel..."
3,"En anden kort tekst på dansk.","Anden titel"
```

### Output CSV Structure
```csv
ID,BODY,TITLE_RAW
1,"This is a short Danish text.","Short title"
2,"This is a much longer Danish text...","Long title..."
3,"Another short text in Danish.","Another title"
```

## Performance Considerations

### Single Machine
- **Memory Usage**: Loading two models requires more RAM/VRAM
- **Speed**: Small texts process faster with the small model
- **Quality**: Large model may provide better translations for complex texts
- **GPU Optimization**: Uses dataset-based batching for maximum GPU efficiency
- **Dynamic Batch Sizing**: Automatically adjusts batch sizes based on available GPU memory
- **Resume Capability**: Skip already processed files when restarting

### Two-Phase Distributed
- **Memory Efficiency**: Each machine only loads one model
- **Scalability**: Can process larger datasets across multiple machines
- **Network Transfer**: Only large texts need to be transferred between machines
- **Fault Tolerance**: If one phase fails, you can restart just that phase with `--resume`

### File Processing Optimization
- **Smart Filtering**: Automatically skips output files (.en, .large, .small) from previous runs
- **Resume Support**: Use `--resume` to continue interrupted processing sessions
- **Progress Tracking**: Clear progress indicators for large dataset processing

## Testing

Run the test script to verify functionality:

```bash
python test_model_separation.py
```

This creates a test CSV with both short and long Danish texts and demonstrates the separation process.

## Troubleshooting

### Common Issues

#### Out of Memory Errors
- Use `--load-in-4bit` for quantization (NVIDIA GPUs only)
- Set `--device-map cpu` to use CPU only
- Script automatically adjusts batch sizes based on available GPU memory
- Use two-phase processing to split memory requirements

#### Model Loading Issues
- Check internet connection for model downloads
- Use `--local-files-only` if models are cached locally
- Verify model IDs are correct
- Script automatically detects model type (causal LM vs seq2seq)

#### Token Count Estimation
- The script uses the tokenizer's `encode()` method for accurate counts
- Falls back to character-based estimation if tokenizer fails
- Adjust `--max-tokens` if needed for your specific use case

#### Two-Phase Issues
- Ensure Phase 1 completes before starting Phase 2
- Verify the large CSV file exists before Phase 2
- Use `--resume` to skip already completed files
- Check that merge script can access all required files

#### Resume and File Processing
- Use `--resume` to continue interrupted processing
- Script automatically skips files with `.en`, `.large`, or `.small` suffixes
- Output files are checked for completeness (same row count as input)
- Delete incomplete output files if you want to reprocess them

#### GPU Efficiency Warnings
- Script uses optimized dataset-based batching to maximize GPU efficiency
- Batch sizes are automatically adjusted based on GPU memory
- Causal LMs and seq2seq models use different optimization strategies

### Debug Mode

For debugging, you can enable verbose output by setting environment variables:

```bash
export CUDA_LAUNCH_BLOCKING=1
python3 translate_unzipped_eurollm.py --root /path/to/csv/files --resume
```

## Complete Usage Examples

### Example 1: First-Time Processing
```bash
# Process all CSV files in a directory for the first time
python3 translate_unzipped_eurollm.py \
    --root ./unzipped \
    --model-small "Helsinki-NLP/opus-mt-da-en" \
    --model-large "eurollm/eurollm-7b" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --fields BODY TITLE_RAW

# Output:
# Found 168 CSV files to process:
# ✓ GPU detected: NVIDIA RTX 4090
# 🔧 Optimal batch size for 24.0GB GPU: 16
# 🔄 [1/168] Processing: Denmark_2018_10_postings.csv
# 📊 Found 1500 rows to process
# 🔍 Separating rows by token count...
# 📊 Row separation results:
#   - Small rows (≤512 tokens): 1200
#   - Large rows (>512 tokens): 300
# 🚀 Processing small rows with small model...
# 🚀 Processing large rows with large model...
# ✅ Successfully processed: Denmark_2018_10_postings.csv
```

### Example 2: Resume After Interruption
```bash
# Continue processing after interruption - skips already completed files
python3 translate_unzipped_eurollm.py \
    --root ./unzipped \
    --model-large "eurollm/eurollm-7b" \
    --resume

# Output:
# Found 168 CSV files to process:
# 🔄 [1/168] Processing: Denmark_2018_10_postings.csv
# ⏭️ Skipping (already complete): Denmark_2018_10_postings.csv
# 🔄 [2/168] Processing: Denmark_2018_11_postings.csv
# ⏭️ Skipping (already complete): Denmark_2018_11_postings.csv
# 🔄 [3/168] Processing: Denmark_2018_12_postings.csv
# 📊 Found 2000 rows to process
# ... continues with unprocessed files only
```

### Example 3: Two-Phase Processing
```bash
# Phase 1: Local machine (processes short texts, saves long texts)
python3 translate_unzipped_eurollm.py \
    --root ./unzipped \
    --phase 1 \
    --model-small "Helsinki-NLP/opus-mt-da-en" \
    --resume

# Transfer *.large.csv files to remote machine

# Phase 2: Remote machine with powerful GPU (processes long texts)
python3 translate_unzipped_eurollm.py \
    --phase 2 \
    --remote-input ./Denmark_2018_10_postings.large.csv \
    --model-large "eurollm/eurollm-7b" \
    --device-map auto \
    --load-in-4bit \
    --resume
```

### Example 4: CPU-Only Processing
```bash
# Process on CPU-only machine (slower but works without GPU)
python3 translate_unzipped_eurollm.py \
    --root ./unzipped \
    --device-map cpu \
    --model-small "Helsinki-NLP/opus-mt-da-en" \
    --model-large "facebook/nllb-200-distilled-600M" \
    --resume
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Helsinki-NLP for the Danish-English translation models
- EuroLLM team for the large language model
- Hugging Face for the Transformers library

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the detailed documentation in `README_model_separation.md`
3. Open an issue on the repository


