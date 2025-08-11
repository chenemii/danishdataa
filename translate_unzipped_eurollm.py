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
    try:
        from datasets import Dataset
    except ImportError:
        Dataset = None
except Exception:  # optional offline deps
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModelForSeq2SeqLM = None
    pipeline = None
    BitsAndBytesConfig = None
    Dataset = None


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
    def _is_causal_lm_model(self, model_id: str) -> bool:
        """Detect if this is a causal LM model based on the model name"""
        causal_lm_patterns = [
            "llama", "mistral", "gpt", "falcon", "bloom", "opt", "eurollm",
            "codellama", "vicuna", "alpaca", "wizard", "orca"
        ]
        model_lower = model_id.lower()
        return any(pattern in model_lower for pattern in causal_lm_patterns)
    
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
        # Detect model type based on model name/config
        
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
        
        # Determine if this is a causal LM (like Llama/EuroLLM) or seq2seq model
        self.is_causal_lm = self._is_causal_lm_model(cfg.model_id)
        print(f"Detected model type: {'Causal LM' if self.is_causal_lm else 'Seq2Seq'}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_id,
                local_files_only=cfg.local_files_only,
            )
            
            if self.is_causal_lm:
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.model_id,
                    local_files_only=cfg.local_files_only,
                    **model_kwargs_gpu,
                )
            else:
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
                
                if self.is_causal_lm:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        cfg.model_id,
                        local_files_only=False,
                        **model_kwargs_gpu,
                    )
                else:
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
            
        # Create pipeline with device handling based on model type
        try:
            if self.is_causal_lm:
                # For causal LMs, use text-generation pipeline
                if cfg.device_map.startswith("cuda"):
                    self.pipe = pipeline(
                        task="text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=cfg.device_map,
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=0.1,
                    )
                else:
                    self.pipe = pipeline(
                        task="text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=-1,  # CPU
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=0.1,
                    )
            else:
                # For seq2seq models, use translation pipeline
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
            if self.is_causal_lm:
                self.pipe = pipeline(
                    task="text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1,  # CPU
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.1,
                )
            else:
                self.pipe = pipeline(
                    task="translation_da_to_en",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1,  # CPU
                )
        
        # Set environment variable to help with CUDA debugging
        import os
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Calculate optimal batch sizes based on GPU memory
        self._calculate_optimal_batch_sizes()

    def _calculate_optimal_batch_sizes(self):
        """Calculate optimal batch sizes based on available GPU memory"""
        try:
            import torch
            if torch.cuda.is_available() and hasattr(self, 'model') and self.model.device.type == 'cuda':
                # Get GPU memory info
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                if self.is_causal_lm:
                    # Causal LMs need more memory per sample
                    if gpu_memory_gb >= 24:  # High-end GPU
                        self.optimal_batch_size = 16
                    elif gpu_memory_gb >= 12:  # Mid-range GPU
                        self.optimal_batch_size = 8
                    elif gpu_memory_gb >= 8:  # Lower-end GPU
                        self.optimal_batch_size = 4
                    else:  # Very limited memory
                        self.optimal_batch_size = 2
                else:
                    # Seq2seq models are more memory efficient
                    if gpu_memory_gb >= 24:  # High-end GPU
                        self.optimal_batch_size = 32
                    elif gpu_memory_gb >= 12:  # Mid-range GPU
                        self.optimal_batch_size = 16
                    elif gpu_memory_gb >= 8:  # Lower-end GPU
                        self.optimal_batch_size = 8
                    else:  # Very limited memory
                        self.optimal_batch_size = 4
                
                print(f"  🔧 Optimal batch size for {gpu_memory_gb:.1f}GB GPU: {self.optimal_batch_size}")
            else:
                # CPU fallback
                self.optimal_batch_size = 4 if self.is_causal_lm else 8
                print(f"  🔧 CPU batch size: {self.optimal_batch_size}")
                
        except Exception as e:
            print(f"  ⚠ Could not calculate optimal batch size: {e}")
            self.optimal_batch_size = 4 if self.is_causal_lm else 8

    def _extract_translation_from_generated(self, generated_text: str, instruction: str) -> str:
        """Extract the translation from the generated text by the causal LM"""
        try:
            # Remove the original instruction from the generated text
            if instruction in generated_text:
                result = generated_text.replace(instruction, "").strip()
            else:
                result = generated_text.strip()
            
            # If the model added extra text, try to find the actual translation
            # Look for the text after </TEXT> or just take everything after the instruction
            if "</TEXT>" in result:
                parts = result.split("</TEXT>", 1)
                if len(parts) > 1:
                    result = parts[1].strip()
            
            # Remove common model artifacts
            result = result.replace("Translation:", "").replace("English translation:", "").strip()
            
            # If the result is empty or very short, return the original text
            if not result or len(result) < 3:
                # Try to extract from the original instruction
                text_start = instruction.find("<TEXT>\n") + 7
                text_end = instruction.find("\n</TEXT>")
                if text_start > 6 and text_end > text_start:
                    return instruction[text_start:text_end]
                else:
                    return generated_text.strip()
            
            return result
        except Exception:
            # Fallback: return the original generated text
            return generated_text.strip()

    def translate(self, text: str) -> str:
        try:
            # Clean input text
            text = text.replace('\x00', '').replace('\ufffd', '')
            
            if self.is_causal_lm:
                # For causal LMs, we need to use instruction-based prompting
                instruction = _build_instruction(text)
                
                # Handle long texts by chunking them
                max_input_length = 1000  # Conservative max length for causal LMs
                if len(instruction) > max_input_length:
                    # For very long texts, chunk the original text, not the instruction
                    chunks = [text[i:i+max_input_length//2] for i in range(0, len(text), max_input_length//2)]
                    translated_chunks = []
                    for j, chunk in enumerate(chunks):
                        try:
                            chunk_instruction = _build_instruction(chunk)
                            outputs = self.pipe(chunk_instruction, max_new_tokens=512, do_sample=False, temperature=0.1)
                            if isinstance(outputs, list) and outputs:
                                generated_text = outputs[0].get("generated_text", "")
                                # Extract the translation from the generated text
                                translated = self._extract_translation_from_generated(generated_text, chunk_instruction)
                                translated_chunks.append(str(translated).strip())
                            else:
                                translated_chunks.append(chunk)  # Keep original if translation fails
                        except Exception as chunk_error:
                            print(f"      Chunk {j+1}/{len(chunks)} translation error: {chunk_error}")
                            translated_chunks.append(chunk)  # Keep original if translation fails
                    return " ".join(translated_chunks)
                else:
                    outputs = self.pipe(instruction, max_new_tokens=512, do_sample=False, temperature=0.1)
                    if isinstance(outputs, list) and outputs:
                        generated_text = outputs[0].get("generated_text", "")
                        # Extract the translation from the generated text
                        translated = self._extract_translation_from_generated(generated_text, instruction)
                        return str(translated).strip()
                    return ""
            else:
                # For seq2seq models, use the original translation logic
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
        """Translate multiple texts efficiently using dataset-based batch processing"""
        if not texts:
            return []
        
        try:
            # Clean and preprocess all texts
            cleaned_texts = []
            for text in texts:
                cleaned = text.replace('\x00', '').replace('\ufffd', '')
                # Limit length to avoid CUDA errors
                if len(cleaned) > 500:
                    cleaned = cleaned[:500]
                cleaned_texts.append(cleaned)
            
            if self.is_causal_lm:
                # For causal LMs, we need dataset-based batching with instructions
                return self._translate_batch_causal_lm(cleaned_texts)
            else:
                # For seq2seq models, use optimized dataset-based batching
                return self._translate_batch_seq2seq(cleaned_texts)
                
        except Exception as e:
            print(f"      Batch translation error: {e}")
            return [text for text in texts]  # Return original texts on error

    def _translate_batch_causal_lm(self, texts: List[str]) -> List[str]:
        """Optimized batch processing for causal language models using datasets"""
        if Dataset is None:
            print("      Dataset not available, falling back to sequential processing")
            return self._translate_batch_sequential(texts)
        
        try:
            # Prepare instructions for all texts
            instructions = [_build_instruction(text) for text in texts]
            
            # Create dataset for efficient batching
            dataset = Dataset.from_dict({"text": instructions, "original": texts})
            
            # Use pipeline with dataset for optimized GPU batching
            # The pipeline will automatically handle batching efficiently
            batch_size = min(self.optimal_batch_size, len(texts))  # Use calculated optimal batch size
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch_instructions = instructions[i:i + batch_size]
                batch_originals = texts[i:i + batch_size]
                
                try:
                    # Process batch with optimized pipeline
                    outputs = self.pipe(
                        batch_instructions, 
                        max_new_tokens=512, 
                        do_sample=False, 
                        temperature=0.1,
                        batch_size=batch_size,
                        return_full_text=False
                    )
                    
                    if isinstance(outputs, list):
                        for j, output in enumerate(outputs):
                            if isinstance(output, dict) and "generated_text" in output:
                                generated_text = output["generated_text"]
                                # Extract translation from generated text
                                translated = self._extract_translation_from_generated(
                                    generated_text, batch_instructions[j]
                                )
                                results.append(str(translated).strip())
                            else:
                                results.append(batch_originals[j])  # Fallback
                    else:
                        results.extend(batch_originals)  # Fallback
                        
                except Exception as batch_error:
                    print(f"      Causal LM batch error: {batch_error}")
                    # Fallback to individual processing for this batch
                    for text in batch_originals:
                        try:
                            translated = self.translate(text)
                            results.append(translated)
                        except Exception:
                            results.append(text)
            
            return results
            
        except Exception as e:
            print(f"      Causal LM batch processing error: {e}")
            return self._translate_batch_sequential(texts)

    def _translate_batch_seq2seq(self, texts: List[str]) -> List[str]:
        """Optimized batch processing for seq2seq models using datasets"""
        if Dataset is None:
            print("      Dataset not available, falling back to sequential processing")
            return self._translate_batch_sequential(texts)
        
        try:
            # Create dataset for efficient batching
            dataset = Dataset.from_dict({"text": texts})
            
            # Use pipeline with dataset for optimized GPU batching
            # This uses the recommended approach for GPU efficiency
            batch_size = min(self.optimal_batch_size, len(texts))  # Use calculated optimal batch size
            
            # Process using dataset with pipeline - this is the recommended approach
            def process_batch(examples):
                try:
                    outputs = self.pipe(
                        examples["text"], 
                        max_length=400,
                        batch_size=batch_size
                    )
                    
                    if isinstance(outputs, list):
                        translations = []
                        for output in outputs:
                            if isinstance(output, dict) and "translation_text" in output:
                                translations.append(str(output["translation_text"]).strip())
                            else:
                                translations.append("")  # Fallback for failed translations
                        return {"translations": translations}
                    else:
                        return {"translations": [""] * len(examples["text"])}
                        
                except Exception as e:
                    print(f"      Seq2seq batch processing error: {e}")
                    return {"translations": examples["text"]}  # Return original texts
            
            # Process dataset in batches
            results = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_dict = {"text": batch_texts}
                batch_results = process_batch(batch_dict)
                results.extend(batch_results["translations"])
            
            return results
            
        except Exception as e:
            print(f"      Seq2seq batch processing error: {e}")
            return self._translate_batch_sequential(texts)

    def _translate_batch_sequential(self, texts: List[str]) -> List[str]:
        """Fallback sequential processing when dataset batching is not available"""
        results = []
        for text in texts:
            try:
                translated = self.translate(text)
                results.append(translated)
            except Exception as single_error:
                print(f"      Single translation error: {single_error}")
                results.append(text)  # Return original text on error
        return results


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
            
            # Skip files that are output or intermediate files from previous processing
            fn_lower = fn.lower()
            name_without_ext = fn_lower.replace('.csv', '')
            
            # Check if the filename contains processing suffixes
            skip_patterns = ['en', 'small', 'large']
            should_skip = False
            
            for pattern in skip_patterns:
                # Check if pattern appears as a suffix (with or without dots)
                if (name_without_ext.endswith('.' + pattern) or 
                    name_without_ext.endswith('_' + pattern) or
                    ('.' + pattern + '.') in fn_lower or
                    ('_' + pattern + '_') in fn_lower or
                    # Also check for direct matches like "test_EN.csv"
                    name_without_ext.endswith(pattern)):
                    should_skip = True
                    break
            
            if should_skip:
                continue  # skip processed output files and intermediate files
            
            csvs.append(os.path.join(dirpath, fn))
    return csvs


def is_output_complete(input_path: str, output_path: str) -> bool:
    """
    Check if the output CSV exists and has the same number of rows as the input CSV.
    Returns True if output is complete, False otherwise.
    """
    if not os.path.exists(output_path):
        return False
    
    try:
        # Count rows in input file
        with open(input_path, "r", encoding="utf-8", newline="") as f:
            input_reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE, escapechar="\\")
            input_rows = sum(1 for _ in input_reader)
        
        # Count rows in output file
        with open(output_path, "r", encoding="utf-8", newline="") as f:
            output_reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONE, escapechar="\\")
            output_rows = sum(1 for _ in output_reader)
        
        # Output should have the same number of rows as input (including header)
        return input_rows == output_rows and output_rows > 0
    
    except Exception as e:
        print(f"  ⚠ Error checking output completeness for {output_path}: {e}")
        return False


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
    parser.add_argument("--resume", action="store_true", 
                       help="Skip processing files that already have complete output (resume mode)")
    
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
        
        # Check if resume mode is enabled and output is already complete
        if args.resume and is_output_complete(args.remote_input, out_path):
            print(f"   ⏭️ Skipping (already complete): {os.path.basename(args.remote_input)}")
        else:
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
                
                # Check if resume mode is enabled and output is already complete
                if args.resume and is_output_complete(in_path, out_path):
                    print(f"   ⏭️ Skipping (already complete): {os.path.basename(in_path)}")
                    continue
                
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
                
                # Check if resume mode is enabled and both outputs are already complete
                if args.resume:
                    output_complete = is_output_complete(in_path, out_path)
                    large_complete = is_output_complete(in_path, large_csv_path)
                    if output_complete and large_complete:
                        print(f"   ⏭️ Skipping (already complete): {os.path.basename(in_path)}")
                        continue
                    elif output_complete:
                        print(f"   ⏭️ Skipping main output (already complete), large CSV may be updated")
                    elif large_complete:
                        print(f"   ⏭️ Large CSV already complete, main output may be updated")
                
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


