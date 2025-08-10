#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
import csv
from typing import List, Tuple, Iterable
from dataclasses import dataclass

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:  # optional offline deps
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModelForSeq2SeqLM = None
    pipeline = None
    BitsAndBytesConfig = None


HF_MODEL_SMALL = os.environ.get("HF_MODEL_SMALL", "Helsinki-NLP/opus-mt-da-en")
HF_MODEL_LARGE = os.environ.get("HF_MODEL_LARGE", "eurollm/eurollm-7b")


def is_probably_danish(text: str) -> bool:
    """
    Lightweight heuristic to detect Danish sentences/segments.
    - Contains Danish-specific characters: æ, ø, å (case-insensitive)
    - OR at least two common Danish words
    """
    if not text:
        return False

    lowered = text.lower()
    if any(ch in lowered for ch in ("æ", "ø", "å")):
        return True

    danish_markers = {
        "og", "ikke", "til", "med", "for", "på", "af", "er", "vi", "du", "kan", "skal",
        "job", "jobbet", "stilling", "ansøgning", "opgaver", "løn", "arbejdsplads", "uddannelse",
        "arbejde", "erfaring", "krav", "ansvar", "tiltrædelse", "vejledning", "kolleger",
    }
    # Use a simpler regex that works across Python versions
    words = re.findall(r"\b[a-zA-ZæøåÆØÅ']+\b", lowered)
    hits = sum(1 for w in words if w in danish_markers)
    return hits >= 2


def split_into_sentences_preserve_delimiters(text: str) -> List[str]:
    """
    Split text into sentences while keeping delimiters. Falls back to splitting on newlines and punctuation.
    """
    if not text:
        return []
    # First, normalize CRLF
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split by newline boundaries preserving them
    parts: List[str] = []
    for block in normalized.split("\n"):
        if not block:
            parts.append("\n")
            continue
        # Split on sentence terminators while keeping them
        segments = re.split(r"([.!?])\s+", block)
        # Re-stitch keeping delimiters in place
        rebuilt: List[str] = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            if i + 1 < len(segments) and segments[i + 1] in {".", "!", "?"}:
                rebuilt.append(seg + segments[i + 1] + " ")
                i += 2
            else:
                rebuilt.append(seg)
                i += 1
        parts.extend(rebuilt)
        parts.append("\n")
    # Remove a trailing newline artifact if present
    if parts and parts[-1] == "\n":
        parts.pop()
    return parts


def group_consecutive(predicate_list: List[bool]) -> List[Tuple[int, int, bool]]:
    """
    Given a list of booleans, returns list of (start_idx, end_idx_exclusive, value) for consecutive runs.
    """
    if not predicate_list:
        return []
    groups: List[Tuple[int, int, bool]] = []
    start = 0
    current = predicate_list[0]
    for i in range(1, len(predicate_list)):
        if predicate_list[i] != current:
            groups.append((start, i, current))
            start = i
            current = predicate_list[i]
    groups.append((start, len(predicate_list), current))
    return groups


def _build_instruction(text: str) -> str:
    return (
        "Translate the following text to English, but only translate Danish parts. "
        "Keep any English text unchanged. Preserve the original structure, line breaks, URLs, and punctuation. "
        "Return only the transformed text with no extra commentary.\n\n"
        "<TEXT>\n" + text + "\n</TEXT>"
    )


def estimate_token_count(text: str, tokenizer) -> int:
    """
    Estimate the number of tokens in the text using the tokenizer.
    """
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception:
        # Fallback: rough estimate based on character count
        return len(text) // 4  # Rough approximation: 4 chars per token


@dataclass
class LocalGenConfig:
    model_id: str
    device_map: str = "auto"
    load_in_4bit: bool = False
    local_files_only: bool = False
    torch_dtype: str = "auto"  # "auto", "bfloat16", "float16", "float32"
    
    def __post_init__(self):
        # Auto-detect GPU and set optimal defaults
        try:
            import torch
            if torch.cuda.is_available():
                if self.device_map == "auto":
                    self.device_map = "cuda:0"
                if self.torch_dtype == "auto":
                    self.torch_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
                print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"  Device map: {self.device_map}, dtype: {self.torch_dtype}")
            else:
                if self.device_map == "auto":
                    self.device_map = "cpu"
                print("⚠ No GPU detected, using CPU")
        except ImportError:
            print("⚠ PyTorch not available, using CPU")
            self.device_map = "cpu"


class LocalTranslator:
    def __init__(self, cfg: LocalGenConfig) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None or pipeline is None:
            raise RuntimeError("Transformers not available. Please install: pip install -U transformers accelerate safetensors")

        quant_config = None
        if cfg.load_in_4bit:
            if BitsAndBytesConfig is None:
                raise RuntimeError("bitsandbytes not available. Install with: pip install bitsandbytes (requires NVIDIA CUDA)")
            quant_config = BitsAndBytesConfig(load_in_4bit=True)

        model_kwargs = {
            "device_map": cfg.device_map,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        if cfg.torch_dtype != "auto":
            import torch  # local import
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            model_kwargs["torch_dtype"] = dtype_map.get(cfg.torch_dtype, torch.float32)

        # Load model with GPU optimization
        # Use Helsinki-NLP/opus-mt-da-en for reliable Danish to English translation
        
        # Set up GPU-optimized loading
        import torch
        if cfg.device_map.startswith("cuda"):
            print(f"Loading model on GPU: {cfg.device_map}")
            # Use GPU-optimized settings
            model_kwargs_gpu = {
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                "low_cpu_mem_usage": True,
            }
        else:
            print("Loading model on CPU")
            model_kwargs_gpu = {}
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_id,
                local_files_only=cfg.local_files_only,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.model_id,
                local_files_only=cfg.local_files_only,
                **model_kwargs_gpu,
            )
        except Exception as load_err:
            if cfg.local_files_only:
                print(
                    "Model/tokenizer not found locally; retrying with online download enabled...",
                    file=sys.stderr,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    cfg.model_id,
                    local_files_only=False,
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    cfg.model_id,
                    local_files_only=False,
                    **model_kwargs_gpu,
                )
            else:
                raise load_err
        
        # Move model to device after loading
        if cfg.device_map.startswith("cuda"):
            import torch
            self.model = self.model.to(cfg.device_map)
            self.tokenizer = self.tokenizer
        
        # Set pad_token_id once to avoid repeated warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Create pipeline with device handling
        try:
            if cfg.device_map.startswith("cuda"):
                self.pipe = pipeline(
                    task="translation_da_to_en",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=cfg.device_map,
                )
            else:
                self.pipe = pipeline(
                    task="translation_da_to_en",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1,  # CPU
                )
        except Exception as e:
            print(f"Pipeline creation error: {e}")
            # Fallback to CPU if GPU fails
            self.pipe = pipeline(
                task="translation_da_to_en",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # CPU
            )
        
        # Set environment variable to help with CUDA debugging
        import os
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    def translate(self, text: str) -> str:
        try:
            # Clean input text
            text = text.replace('\x00', '').replace('\ufffd', '')
            
            # Handle long texts by chunking them
            max_length = 400  # Conservative max length for the model
            if len(text) > max_length:
                # Split into chunks and translate each
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                translated_chunks = []
                for j, chunk in enumerate(chunks):
                    try:
                        outputs = self.pipe(chunk, max_length=max_length)
                        if isinstance(outputs, list) and outputs:
                            translated = outputs[0].get("translation_text", "")
                            translated_chunks.append(str(translated).strip())
                        else:
                            translated_chunks.append(chunk)  # Keep original if translation fails
                    except Exception as chunk_error:
                        print(f"      Chunk {j+1}/{len(chunks)} translation error: {chunk_error}")
                        translated_chunks.append(chunk)  # Keep original if translation fails
                return " ".join(translated_chunks)
            else:
                outputs = self.pipe(text, max_length=max_length)
                if isinstance(outputs, list) and outputs:
                    translated = outputs[0].get("translation_text", "")
                    return str(translated).strip()
                return ""
        except Exception as e:
            print(f"      Translation error: {e}")
            return text  # Return original text on error

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate multiple texts efficiently using batch processing"""
        if not texts:
            return []
        
        try:
            # Process in batches for better GPU efficiency
            batch_size = 4  # Reduced batch size to avoid CUDA errors
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    # Clean and validate input texts
                    cleaned_batch = []
                    for text in batch:
                        # Remove or replace problematic characters
                        cleaned = text.replace('\x00', '').replace('\ufffd', '')
                        # Limit length to avoid CUDA errors
                        if len(cleaned) > 500:
                            cleaned = cleaned[:500]
                        cleaned_batch.append(cleaned)
                    
                    outputs = self.pipe(cleaned_batch, max_length=400, batch_size=len(cleaned_batch))
                    if isinstance(outputs, list):
                        for output in outputs:
                            if isinstance(output, dict) and "translation_text" in output:
                                results.append(str(output["translation_text"]).strip())
                            else:
                                results.append("")  # Fallback for failed translations
                    else:
                        results.extend([""] * len(cleaned_batch))
                except Exception as batch_error:
                    print(f"      Batch translation error: {batch_error}")
                    # Fallback to individual translation for this batch
                    for text in batch:
                        try:
                            cleaned = text.replace('\x00', '').replace('\ufffd', '')
                            if len(cleaned) > 500:
                                cleaned = cleaned[:500]
                            output = self.pipe(cleaned, max_length=400)
                            if isinstance(output, list) and output and isinstance(output[0], dict):
                                results.append(str(output[0].get("translation_text", "")).strip())
                            else:
                                results.append("")
                        except Exception as single_error:
                            print(f"      Single translation error: {single_error}")
                            results.append("")
            
            return results
        except Exception as e:
            print(f"      Batch translation error: {e}")
            return [text for text in texts]  # Return original texts on error


def translate_mixed_text(text: str, *, translator: LocalTranslator) -> str:
    if not text:
        return text

    parts = split_into_sentences_preserve_delimiters(text)
    if not parts:
        return text
    flags = [is_probably_danish(p) for p in parts]
    if not any(flags):
        return text  # likely already English-only

    reconstructed: List[str] = []
    for start, end, is_danish_block in group_consecutive(flags):
        block_text = "".join(parts[start:end])
        if is_danish_block:
            try:
                translated = translator.translate(block_text)
            except Exception:
                translated = block_text  # fail-safe
            reconstructed.append(translated)
        else:
            reconstructed.append(block_text)
    return "".join(reconstructed)


def iter_csv_rows(path: str) -> Tuple[List[str], Iterable[List[str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(
            f,
            delimiter=",",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            strict=True,
        )
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        return header, reader


def separate_rows_by_length(rows: List[List[str]], target_indices: List[int], 
                           small_translator: LocalTranslator, max_tokens: int = 512) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Separate rows into small and large based on estimated token count.
    Returns (small_rows, large_rows) where small_rows can be handled by the small model.
    """
    small_rows = []
    large_rows = []
    
    for row in rows:
        # Make sure row has at least header length (malformed rows will be padded)
        if len(row) < len(target_indices):
            row = row + [""] * (len(target_indices) - len(row))
        
        # Check if any target field exceeds the token limit
        needs_large_model = False
        for idx in target_indices:
            if idx < len(row):
                text = row[idx]
                if text and text.strip():
                    token_count = estimate_token_count(text, small_translator.tokenizer)
                    if token_count > max_tokens:
                        needs_large_model = True
                        break
        
        if needs_large_model:
            large_rows.append(row)
        else:
            small_rows.append(row)
    
    return small_rows, large_rows


def process_csv_phase1_local(in_path: str, out_path: str, large_csv_path: str, target_fields: List[str], 
                           rate_limit_s: float, max_tokens: int, *, translator: LocalTranslator) -> None:
    """
    Phase 1: Process CSV locally with small model.
    - Translate short texts and save to output CSV
    - Save long texts to separate CSV for processing on another machine
    """
    print(f"  Processing CSV (Phase 1 - Local): {os.path.basename(in_path)}")
    
    with open(in_path, "r", encoding="utf-8", newline="") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout, \
         open(large_csv_path, "w", encoding="utf-8", newline="") as flarge:
        
        reader = csv.reader(
            fin,
            delimiter=",",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            strict=True,
        )
        writer = csv.writer(
            fout,
            delimiter=",",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            lineterminator="\n",
        )
        large_writer = csv.writer(
            flarge,
            delimiter=",",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            lineterminator="\n",
        )

        try:
            header = next(reader)
        except StopIteration:
            print(f"  ⚠ Empty CSV file: {in_path}")
            return

        # Write headers to both files
        writer.writerow(header)
        large_writer.writerow(header)
        
        name_to_idx = {name: i for i, name in enumerate(header)}

        missing = [name for name in target_fields if name not in name_to_idx]
        if missing:
            print(f"  ⚠ Warning: {in_path} missing columns: {missing}")

        target_indices = [name_to_idx[name] for name in target_fields if name in name_to_idx]
        
        # Count total rows for progress
        rows = list(reader)
        total_rows = len(rows)
        print(f"  📊 Found {total_rows} rows to process")
        
        if total_rows == 0:
            print(f"  ✅ No data rows found in {in_path}")
            return

        # Separate rows into small and large based on token count
        print(f"  🔍 Separating rows by token count...")
        small_rows, large_rows = separate_rows_by_length(rows, target_indices, translator, max_tokens)
        
        print(f"  📊 Row separation results:")
        print(f"    - Small rows (≤{max_tokens} tokens): {len(small_rows)}")
        print(f"    - Large rows (>{max_tokens} tokens): {len(large_rows)}")
        
        # Process small rows with local model
        if small_rows:
            print(f"  🚀 Processing small rows with local model...")
            process_rows_with_translator(small_rows, target_indices, translator, "local")
            
            # Write small rows to output file
            for row in small_rows:
                writer.writerow(row)
        
        # Write large rows to separate CSV for processing on another machine
        if large_rows:
            print(f"  📁 Saving {len(large_rows)} large rows to: {os.path.basename(large_csv_path)}")
            for row in large_rows:
                large_writer.writerow(row)
        
        print(f"  ✅ Phase 1 completed:")
        print(f"    - Translated {len(small_rows)} rows locally")
        print(f"    - Saved {len(large_rows)} rows to {os.path.basename(large_csv_path)} for remote processing")


def process_csv_phase2_remote(in_path: str, out_path: str, target_fields: List[str], 
                            rate_limit_s: float, *, translator: LocalTranslator) -> None:
    """
    Phase 2: Process CSV with large model (run on machine with large model).
    This processes only the rows that were too long for the small model.
    """
    print(f"  Processing CSV (Phase 2 - Remote): {os.path.basename(in_path)}")
    
    with open(in_path, "r", encoding="utf-8", newline="") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:
        
        reader = csv.reader(
            fin,
            delimiter=",",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            strict=True,
        )
        writer = csv.writer(
            fout,
            delimiter=",",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            lineterminator="\n",
        )

        try:
            header = next(reader)
        except StopIteration:
            print(f"  ⚠ Empty CSV file: {in_path}")
            return

        writer.writerow(header)
        name_to_idx = {name: i for i, name in enumerate(header)}

        missing = [name for name in target_fields if name not in name_to_idx]
        if missing:
            print(f"  ⚠ Warning: {in_path} missing columns: {missing}")

        target_indices = [name_to_idx[name] for name in target_fields if name in name_to_idx]
        
        # Count total rows for progress
        rows = list(reader)
        total_rows = len(rows)
        print(f"  📊 Found {total_rows} large rows to process with remote model")
        
        if total_rows == 0:
            print(f"  ✅ No data rows found in {in_path}")
            return

        # Process all rows with large model (they were already filtered to be large)
        print(f"  🚀 Processing all rows with large model...")
        process_rows_with_translator(rows, target_indices, translator, "remote")
        
        # Write all processed rows
        for i, row in enumerate(rows):
            # Show progress every 10 rows or at the end
            if i % 10 == 0 or i == total_rows - 1:
                print(f"  📝 Writing row {i+1}/{total_rows} ({(i+1)*100//total_rows}%)")
            writer.writerow(row)
            if rate_limit_s > 0:
                time.sleep(rate_limit_s)
        
        print(f"  ✅ Phase 2 completed: processed {total_rows} large rows")


def process_csv_with_model_separation(in_path: str, out_path: str, target_fields: List[str], 
                                    rate_limit_s: float, max_tokens: int, *, small_translator: LocalTranslator, 
                                    large_translator: LocalTranslator = None) -> None:
    """Legacy function for single-machine processing"""
    print(f"  Processing CSV: {os.path.basename(in_path)}")
    
    with open(in_path, "r", encoding="utf-8", newline="") as fin, open(out_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(
            fin,
            delimiter=",",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            strict=True,
        )
        writer = csv.writer(
            fout,
            delimiter=",",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            lineterminator="\n",
        )

        try:
            header = next(reader)
        except StopIteration:
            print(f"  ⚠ Empty CSV file: {in_path}")
            return

        writer.writerow(header)
        name_to_idx = {name: i for i, name in enumerate(header)}

        missing = [name for name in target_fields if name not in name_to_idx]
        if missing:
            print(f"  ⚠ Warning: {in_path} missing columns: {missing}")

        target_indices = [name_to_idx[name] for name in target_fields if name in name_to_idx]
        
        # Count total rows for progress
        rows = list(reader)
        total_rows = len(rows)
        print(f"  📊 Found {total_rows} rows to process")
        
        if total_rows == 0:
            print(f"  ✅ No data rows found in {in_path}")
            return

        # Separate rows into small and large based on token count
        print(f"  🔍 Separating rows by token count...")
        small_rows, large_rows = separate_rows_by_length(rows, target_indices, small_translator, max_tokens)
        
        print(f"  📊 Row separation results:")
        print(f"    - Small rows (≤{max_tokens} tokens): {len(small_rows)}")
        print(f"    - Large rows (>{max_tokens} tokens): {len(large_rows)}")
        
        # Process small rows with small model
        if small_rows:
            print(f"  🚀 Processing small rows with small model...")
            process_rows_with_translator(small_rows, target_indices, small_translator, "small")
        
        # Process large rows with large model (if available)
        if large_rows:
            if large_translator:
                print(f"  🚀 Processing large rows with large model...")
                process_rows_with_translator(large_rows, target_indices, large_translator, "large")
            else:
                print(f"  ⚠ No large model available, skipping {len(large_rows)} large rows")
                # Keep original text for large rows if no large model
                pass
        
        # Combine all rows back in original order
        all_processed_rows = small_rows + large_rows
        
        # Write all rows
        for i, row in enumerate(all_processed_rows):
            # Show progress every 10 rows or at the end
            if i % 10 == 0 or i == total_rows - 1:
                print(f"  📝 Writing row {i+1}/{total_rows} ({(i+1)*100//total_rows}%)")
            writer.writerow(row)
            if rate_limit_s > 0:
                time.sleep(rate_limit_s)
        
        print(f"  ✅ Completed processing {total_rows} rows")


def process_rows_with_translator(rows: List[List[str]], target_indices: List[int], 
                               translator: LocalTranslator, model_type: str) -> None:
    """Process a batch of rows with the specified translator"""
    if not rows:
        return
    
    # Collect all texts that need translation for batch processing
    texts_to_translate = []
    translation_indices = []  # (row_idx, field_idx) pairs
    
    for i, row in enumerate(rows):
        for idx in target_indices:
            if idx < len(row):
                original = row[idx]
                if original and original.strip():  # Only translate non-empty fields
                    texts_to_translate.append(original)
                    translation_indices.append((i, idx))
    
    print(f"    🔄 Found {len(texts_to_translate)} fields to translate with {model_type} model")
    
    # Batch translate all texts
    if texts_to_translate:
        print(f"    🚀 Starting batch translation with {model_type} model...")
        translated_texts = translator.translate_batch(texts_to_translate)
        print(f"    ✅ Batch translation completed with {model_type} model")
        
        # Apply translations back to rows
        for (row_idx, field_idx), translated_text in zip(translation_indices, translated_texts):
            if translated_text:  # Only update if translation succeeded
                rows[row_idx][field_idx] = translated_text
                print(f"      🔄 Translated field (row {row_idx+1}, field {field_idx}) with {model_type} model")


def process_csv(in_path: str, out_path: str, target_fields: List[str], rate_limit_s: float, *, translator: LocalTranslator) -> None:
    """Legacy function for backward compatibility"""
    process_csv_with_model_separation(in_path, out_path, target_fields, rate_limit_s, 512,
                                    small_translator=translator, large_translator=None)


def find_csv_files(root: str) -> List[str]:
    csvs: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".csv"):
                continue
            if fn.startswith("._"):
                continue  # skip macOS metadata files
            csvs.append(os.path.join(dirpath, fn))
    return csvs


def main():
    parser = argparse.ArgumentParser(description="Translate Danish parts of job description fields to English using local Transformers pipelines with two-phase processing support.")
    parser.add_argument("--root", default="/c/denmark/unzipped", help="Root folder to scan for CSV files (recursively)")
    parser.add_argument("--fields", nargs="*", default=["BODY", "TITLE_RAW"], help="CSV columns to process")
    parser.add_argument("--out-suffix", default=".en", help="Suffix to add before .csv for output files")
    parser.add_argument("--rate-limit", type=float, default=0.0, help="Sleep seconds after each field translation (to avoid rate limits)")
    parser.add_argument("--model-small", default=HF_MODEL_SMALL, help="Small model ID for texts ≤512 tokens")
    parser.add_argument("--model-large", default=HF_MODEL_LARGE, help="Large model ID for texts >512 tokens (default: eurollm/eurollm-7b)")
    parser.add_argument("--device-map", default="auto", help='Device map for local model (e.g., "auto", "cpu", "cuda")')
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (requires bitsandbytes + CUDA)")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"], help="Torch dtype for local model")
    parser.add_argument("--local-files-only", action="store_true", help="Transformers should only use local files (no network)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens for small model (default: 512)")
    
    # Two-phase processing options
    parser.add_argument("--phase", choices=["1", "2", "both"], default="both", 
                       help="Processing phase: 1=local only, 2=remote only, both=single machine")
    parser.add_argument("--large-csv-suffix", default=".large", 
                       help="Suffix for CSV containing large rows (Phase 1)")
    parser.add_argument("--remote-input", help="Input CSV for Phase 2 (remote processing)")
    
    args = parser.parse_args()

    csv_files = find_csv_files(args.root)
    if not csv_files:
        print(f"No CSV files found under {args.root}")
        return 0

    print(f"Found {len(csv_files)} CSV file(s) under {args.root}")

    if args.phase == "2":
        # Phase 2: Remote processing only
        if not args.remote_input:
            print("❌ Error: --remote-input is required for Phase 2")
            return 1
        
        print(f"🔄 Loading large model for remote processing: {args.model_large}")
        cfg_large = LocalGenConfig(
            model_id=args.model_large,
            device_map=args.device_map,
            load_in_4bit=args.load_in_4bit,
            local_files_only=args.local_files_only,
            torch_dtype=args.torch_dtype,
        )
        large_translator = LocalTranslator(cfg_large)
        
        base, ext = os.path.splitext(args.remote_input)
        out_path = base + args.out_suffix + ext
        
        print(f"\n🔄 Processing remote CSV: {os.path.basename(args.remote_input)}")
        print(f"   📁 Input:  {args.remote_input}")
        print(f"   📁 Output: {out_path}")
        
        try:
            process_csv_phase2_remote(args.remote_input, out_path, args.fields, args.rate_limit, 
                                   translator=large_translator)
            print(f"   ✅ Successfully processed: {os.path.basename(args.remote_input)}")
        except Exception as e:
            print(f"   ❌ Error processing {os.path.basename(args.remote_input)}: {e}")
            return 1
        
    else:
        # Phase 1 or both: Local processing
        print(f"🔄 Loading small model for local processing: {args.model_small}")
        cfg_small = LocalGenConfig(
            model_id=args.model_small,
            device_map=args.device_map,
            load_in_4bit=args.load_in_4bit,
            local_files_only=args.local_files_only,
            torch_dtype=args.torch_dtype,
        )
        small_translator = LocalTranslator(cfg_small)

        if args.phase == "both":
            # Single machine processing with both models
            large_translator = None
            if args.model_large != args.model_small:
                print(f"🔄 Loading large model: {args.model_large}")
                cfg_large = LocalGenConfig(
                    model_id=args.model_large,
                    device_map=args.device_map,
                    load_in_4bit=args.load_in_4bit,
                    local_files_only=args.local_files_only,
                    torch_dtype=args.torch_dtype,
                )
                large_translator = LocalTranslator(cfg_large)
            else:
                print(f"ℹ️ Using same model for both small and large texts: {args.model_small}")
                large_translator = small_translator

            for i, in_path in enumerate(csv_files):
                base, ext = os.path.splitext(in_path)
                out_path = base + args.out_suffix + ext
                print(f"\n🔄 [{i+1}/{len(csv_files)}] Processing: {os.path.basename(in_path)}")
                print(f"   📁 Input:  {in_path}")
                print(f"   📁 Output: {out_path}")
                try:
                    process_csv_with_model_separation(in_path, out_path, args.fields, args.rate_limit, args.max_tokens,
                                                   small_translator=small_translator, 
                                                   large_translator=large_translator)
                    print(f"   ✅ Successfully processed: {os.path.basename(in_path)}")
                except Exception as e:
                    print(f"   ❌ Error processing {os.path.basename(in_path)}: {e}")
        
        else:
            # Phase 1: Local processing only
            for i, in_path in enumerate(csv_files):
                base, ext = os.path.splitext(in_path)
                out_path = base + args.out_suffix + ext
                large_csv_path = base + args.large_csv_suffix + ext
                
                print(f"\n🔄 [{i+1}/{len(csv_files)}] Processing (Phase 1): {os.path.basename(in_path)}")
                print(f"   📁 Input:  {in_path}")
                print(f"   📁 Output: {out_path}")
                print(f"   📁 Large:  {large_csv_path}")
                
                try:
                    process_csv_phase1_local(in_path, out_path, large_csv_path, args.fields, args.rate_limit, args.max_tokens,
                                          translator=small_translator)
                    print(f"   ✅ Successfully processed: {os.path.basename(in_path)}")
                except Exception as e:
                    print(f"   ❌ Error processing {os.path.basename(in_path)}: {e}")
    
    print(f"\n🎉 All done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())


