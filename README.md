# Danish Translation Pipeline

A comprehensive pipeline for translating Danish job descriptions to English using transformer models with support for both single-machine and distributed processing.

## Overview

This project provides a robust solution for translating Danish text to English, specifically designed for job description datasets. It handles the common challenge of max sequence length limitations by automatically separating texts into appropriate categories and using the most suitable model for each.

## Key Features

- **Automatic Text Length Detection**: Uses tokenizer to estimate text length and determine optimal processing strategy
- **Model Separation**: Routes short texts (≤512 tokens) to small model, long texts (>512 tokens) to large model
- **Two-Phase Distributed Processing**: Support for processing across multiple machines
- **Flexible Model Configuration**: Can use different models for small and large texts
- **Batch Processing**: Efficient batch translation for improved performance
- **Backward Compatibility**: Works with original single-model approach

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
python translate_unzipped_eurollm.py --root /path/to/csv/files

# Advanced usage with custom models
python translate_unzipped_eurollm.py \
    --root /path/to/csv/files \
    --model-small "Helsinki-NLP/opus-mt-da-en" \
    --model-large "eurollm/eurollm-7b" \
    --max-tokens 512
```

### Two-Phase Distributed Processing

#### Phase 1: Local Processing (Small Model)
```bash
python translate_unzipped_eurollm.py \
    --root /path/to/csv/files \
    --phase 1 \
    --model-small "Helsinki-NLP/opus-mt-da-en" \
    --max-tokens 512 \
    --large-csv-suffix ".large"
```

#### Phase 2: Remote Processing (Large Model)
```bash
python translate_unzipped_eurollm.py \
    --phase 2 \
    --remote-input /path/to/file.large.csv \
    --model-large "eurollm/eurollm-7b"
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
pip install transformers torch accelerate safetensors

# For GPU acceleration with quantization
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

## Model Recommendations

### Small Model (≤512 tokens)
- **Helsinki-NLP/opus-mt-da-en**: Fast, reliable for short texts
- **Helsinki-NLP/opus-mt-da-en**: Good balance of speed and quality

### Large Model (>512 tokens)
- **eurollm/eurollm-7b**: EuroLLM model for better handling of long texts
- **facebook/nllb-200-distilled-600M**: Alternative for long texts
- **Custom fine-tuned models**: For domain-specific translations

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
- **Batch Processing**: Both models use efficient batch processing

### Two-Phase Distributed
- **Memory Efficiency**: Each machine only loads one model
- **Scalability**: Can process larger datasets across multiple machines
- **Network Transfer**: Only large texts need to be transferred between machines
- **Fault Tolerance**: If one phase fails, you can restart just that phase

## Testing

Run the test script to verify functionality:

```bash
python test_model_separation.py
```

This creates a test CSV with both short and long Danish texts and demonstrates the separation process.

## Troubleshooting

### Common Issues

#### Out of Memory Errors
- Use `--load-in-4bit` for quantization
- Set `--device-map cpu` to use CPU only
- Reduce batch size in the code if needed
- Use two-phase processing to split memory requirements

#### Model Loading Issues
- Check internet connection for model downloads
- Use `--local-files-only` if models are cached locally
- Verify model IDs are correct

#### Token Count Estimation
- The script uses the tokenizer's `encode()` method for accurate counts
- Falls back to character-based estimation if tokenizer fails
- Adjust `--max-tokens` if needed for your specific use case

#### Two-Phase Issues
- Ensure Phase 1 completes before starting Phase 2
- Verify the large CSV file exists before Phase 2
- Check that merge script can access all required files
- Use `--original` parameter in merge to maintain row order

### Debug Mode

For debugging, you can enable verbose output by setting environment variables:

```bash
export CUDA_LAUNCH_BLOCKING=1
python translate_unzipped_eurollm.py --root /path/to/csv/files
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


