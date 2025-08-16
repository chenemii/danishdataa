#!/usr/bin/env python3
# This is a way to run it: 
# python translate_da_en_simple.py --root .\unzipped --year 2024 --batch-size 16 --buffer-rows 64 --out-root en --max-input-tokens 350 --max-length 500
# python translate_da_en_simple.py --root .\unzipped --year 2024 --batch-size 16 --buffer-rows 64 --out-root en --max-input-tokens 350 --max-length 50
# python translate_da_en_simple.py --root .\unzipped --year 2024 --batch-size 16 --buffer-rows 64 --out-root en --max-input-tokens 350 --max-length 500
import argparse
import os
import re
import csv
import json
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from datetime import datetime

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)


def is_probably_danish(text: str) -> bool:
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
    words = re.findall(r"\b[a-zA-ZæøåÆØÅ']+\b", lowered)
    return sum(1 for w in words if w in danish_markers) >= 2


def estimate_token_count(text: str, tokenizer) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return max(1, len(text) // 4)


def chunk_text_by_punctuation_tokens(text: str, tokenizer, max_tokens: int) -> List[str]:
    if not text:
        return []
    
    # Use a more conservative max to ensure we stay under the hard limit
    safe_max_tokens = min(max_tokens, 300)  # Stay well under 512 token model limit
    
    segments: List[str] = []
    for m in re.finditer(r"[^\.,]+[\.,]?\s*", text, flags=re.DOTALL):
        seg = m.group(0)
        if seg:
            segments.append(seg)

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush():
        nonlocal current
        if current:
            chunks.append("".join(current))
            current = []

    for seg in segments:
        seg_tokens = estimate_token_count(seg, tokenizer)
        if seg_tokens > safe_max_tokens:
            flush()
            # For oversized segments, use proper token-aware splitting
            words = seg.split()
            temp_chunk = []
            temp_tokens = 0
            
            for word in words:
                word_tokens = estimate_token_count(word, tokenizer)
                if temp_tokens + word_tokens > safe_max_tokens and temp_chunk:
                    # Flush current chunk and start new one
                    chunks.append(" ".join(temp_chunk))
                    temp_chunk = [word]
                    temp_tokens = word_tokens
                else:
                    temp_chunk.append(word)
                    temp_tokens += word_tokens
            
            if temp_chunk:  # Don't forget the last chunk
                chunks.append(" ".join(temp_chunk))
            current_tokens = 0
            continue

        if current_tokens + seg_tokens <= safe_max_tokens or not current:
            current.append(seg)
            current_tokens += seg_tokens
        else:
            flush()
            current.append(seg)
            current_tokens = seg_tokens

    flush()
    return chunks if chunks else [text]


def find_csv_files(root: str, year: str | None) -> List[str]:
    results: List[str] = []
    year_str = str(year) if year else None
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".csv"):
                continue
            full = os.path.join(dirpath, fn)
            low = full.lower()
            # Skip previously translated outputs
            if low.endswith(".en.csv") or low.endswith("_en.csv") or ".en." in low or "_en_" in low:
                continue
            if year_str and (year_str not in low):
                continue
            # Focus only on postings, ignore skills
            if "skills" in low:
                continue
            if "postings" not in low:
                continue
            results.append(full)
    return results


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def load_completed_set(log_path: str) -> set[str]:
    completed: set[str] = set()
    if not log_path or not os.path.exists(log_path):
        return completed
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Accept formats: "DONE\t<path>" or "DONE <path>"
                if line.startswith("DONE"):
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        completed.add(_normalize_path(parts[1]))
    except Exception:
        pass
    return completed


def append_completed(log_path: str, in_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    except Exception:
        pass
    try:
        ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"DONE\t{_normalize_path(in_path)}\t{ts}\n")
    except Exception:
        # Logging failures should not crash processing
        pass


def build_translation_jobs(
    rows: List[Dict[str, str]],
    header: List[str],
    fields: List[str],
    tokenizer,
    max_tokens: int,
) -> Tuple[List[str], List[Tuple[int, str, int, int]]]:
    # Filter fields to only those that exist in the header
    target_fields = [f for f in fields if f in header]

    flat_inputs: List[str] = []
    # mapping entries: (row_idx, field_name, start, end) in flat_inputs
    mapping: List[Tuple[int, str, int, int]] = []

    # Force-translate designated columns for every row (no Danish detection)
    for r_idx, row in enumerate(rows):
        for field_name in target_fields:
            text = (row.get(field_name) or "")
            if not text.strip():
                continue
            # Always chunk to be safe - never pass unchecked text to the model
            chunks = chunk_text_by_punctuation_tokens(text, tokenizer, max_tokens)
            start = len(flat_inputs)
            flat_inputs.extend(chunks)
            end = len(flat_inputs)
            mapping.append((r_idx, field_name, start, end))

    return flat_inputs, mapping


def translate_texts_with_pipeline(
    texts: List[str],
    pipe,
    batch_size: int,
    max_length: int,
) -> List[str]:
    if not texts:
        return []
    
    # Final safety check: filter out any text that's still too long
    safe_texts = []
    max_safe_tokens = 300  # Conservative limit for the model
    
    for text in texts:
        try:
            # Get the tokenizer from the pipeline
            tokenizer = pipe.tokenizer
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            
            if token_count <= max_safe_tokens:
                safe_texts.append(text)
            else:
                # Emergency truncation - split by sentences and take what fits
                sentences = text.split('. ')
                truncated = ""
                for sent in sentences:
                    test_text = truncated + sent + ". "
                    test_tokens = len(tokenizer.encode(test_text, add_special_tokens=False))
                    if test_tokens <= max_safe_tokens:
                        truncated = test_text
                    else:
                        break
                safe_texts.append(truncated.strip() if truncated.strip() else text[:100])  # Fallback
        except Exception:
            # If anything fails, truncate by character count
            safe_texts.append(text[:400])  # Conservative character limit
    
    # Use pipeline's built-in dataset processing for maximum efficiency
    try:
        import torch
        with torch.inference_mode():
            outputs = pipe(safe_texts, max_length=max_length, batch_size=batch_size)
    except Exception:
        outputs = pipe(safe_texts, max_length=max_length, batch_size=batch_size)
    
    # Handle both single output and batch output formats
    if isinstance(outputs, list):
        return [str(out.get("translation_text", "")).strip() for out in outputs]
    else:
        return [str(outputs.get("translation_text", "")).strip()]


def process_csv(
    in_path: str,
    out_path: str,
    fields: List[str],
    pipe,
    tokenizer,
    max_tokens: int,
    batch_size: int,
    max_length: int,
    buffer_rows: int,
    out_format: str,
) -> None:
    # Count total data rows for progress (excluding header)
    total_rows = 0
    try:
        with open(in_path, "r", encoding="utf-8", newline="") as fcount:
            cnt_reader = csv.DictReader(fcount, escapechar='\\')
            for _ in cnt_reader:
                total_rows += 1
    except Exception:
        total_rows = 0

    with open(in_path, "r", encoding="utf-8", newline="") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin, escapechar='\\')
        writer = None
        if out_format == "csv":
            writer = csv.writer(
                fout,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
                doublequote=True,
                lineterminator="\n",
            )

        # Get header from DictReader fieldnames (reads first row automatically)
        header = reader.fieldnames
        if not header:
            return

        if out_format == "csv":
            writer.writerow(header)
        num_cols = len(header)

        # Stream rows with optional small buffering to improve GPU utilization
        buffer: List[Dict[str, str]] = []
        rows_pbar = tqdm(total=total_rows or None, desc="Rows", unit="row", leave=False)
        def _clean_cell(val: str) -> str:
            s = (val or "")
            # Clear cells that are just a single backslash (common artifact in source CSVs)
            return "" if s.strip() == "\\" else s

        def flush_buffer():
            nonlocal buffer
            if not buffer:
                return
            # Build combined translation jobs for the buffered rows
            flat_inputs: List[str] = []
            combined_mapping: List[Tuple[int, str, int, int]] = []
            for idx, r in enumerate(buffer):
                # Clean each field in the dictionary
                cleaned_r = {k: _clean_cell(v) for k, v in r.items()}
                fi, mp = build_translation_jobs([cleaned_r], header, fields, tokenizer, max_tokens)
                # remap indices for this row relative to global flat_inputs
                offset = len(flat_inputs)
                flat_inputs.extend(fi)
                for (_, field_name, start, end) in mp:
                    combined_mapping.append((idx, field_name, offset + start, offset + end))

            translations: List[str] = []
            if flat_inputs:
                translations = translate_texts_with_pipeline(flat_inputs, pipe, batch_size, max_length)

            # Apply translations per row and write immediately
            for idx, row in enumerate(buffer):
                # Clean each field in the dictionary
                row_dict = {k: _clean_cell(v) for k, v in row.items()}
                # Ensure all header fields are present
                for col in header:
                    if col not in row_dict:
                        row_dict[col] = ""
                
                # apply mapped translations for this row
                for (r_idx, field_name, start, end) in combined_mapping:
                    if r_idx != idx:
                        continue
                    joined = " ".join(translations[start:end]) if (end > start) else ""
                    if joined:
                        row_dict[field_name] = joined
                
                if out_format == "csv":
                    # Convert dict back to list in header order for CSV writing
                    row_list = [row_dict.get(col, "") for col in header]
                    writer.writerow(row_list)  # type: ignore[union-attr]
                else:
                    json.dump(row_dict, fout, ensure_ascii=False)
                    fout.write("\n")
                try:
                    fout.flush()
                except Exception:
                    pass
                try:
                    rows_pbar.update(1)
                except Exception:
                    pass
            buffer = []

        try:
            for row_dict in reader:
                # Ensure all header fields are present in the row
                row = {}
                for col in header:
                    row[col] = row_dict.get(col, "")
                buffer.append(row)
                if len(buffer) >= max(1, buffer_rows):
                    flush_buffer()
            # Flush any remaining
            flush_buffer()
        finally:
            try:
                rows_pbar.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate Danish text in selected CSV columns to English using a small seq2seq model with batching.")
    parser.add_argument("--root", required=True, help="Root directory to recursively scan for CSV files")
    parser.add_argument("--fields", nargs="*", default=["BODY", "TITLE_RAW"], help="CSV columns to translate")
    parser.add_argument("--year", type=str, help="Only process CSV files whose path/name contains this year (e.g., 2024)")
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-da-en", help="Seq2seq translation model id")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for translation")
    parser.add_argument("--max-input-tokens", type=int, default=480, help="Max tokens per chunk before splitting by punctuation")
    parser.add_argument("--max-length", type=int, default=400, help="Generation max_length for the model")
    parser.add_argument("--log-file", type=str, default=None, help="Path to progress log file (default: <root>/da_en_progress.log)")
    parser.add_argument("--out-root", type=str, default=None, help="Directory to save translated files; input tree will be mirrored here")
    parser.add_argument("--buffer-rows", type=int, default=64, help="Number of rows to buffer before a batched translation to improve GPU utilization (each row still written immediately after its batch)")
    parser.add_argument("--out-format", choices=["csv", "jsonl"], default="jsonl", help="Output format: csv or jsonl (default: jsonl)")
    args = parser.parse_args()

    # Load model + pipeline
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # Detect and force GPU usage if available
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True  # best-effort
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

        cuda_available = torch.cuda.is_available()
        print(f"cuda_available={cuda_available}")
        if cuda_available:
            device = 0
            model.to("cuda")
            model.eval()
            try:
                gpu_name = torch.cuda.get_device_name(0)
                cuda_ver = getattr(torch.version, "cuda", None)
                print(f"Using GPU: {gpu_name} | torch={torch.__version__} cuda={cuda_ver}")
            except Exception:
                pass
            # Sanity check the model is on CUDA
            try:
                first_param_device = next(model.parameters()).device.type
                print(f"model_device={first_param_device}")
            except Exception:
                pass
        else:
            device = -1
            model.eval()
            print("CUDA not available; using CPU")
    except Exception as e:
        device = -1
        print(f"PyTorch CUDA check failed; using CPU ({e})")

    pipe = pipeline(task="translation_da_to_en", model=model, tokenizer=tokenizer, device=device)
    try:
        # Verify pipeline device
        pipe_device = getattr(pipe, "device", None)
        print(f"pipeline_device={pipe_device}")
    except Exception:
        pass

    csv_files = find_csv_files(args.root, args.year)
    if not csv_files:
        print("No CSV files found to process.")
        return 0

    # Resolve log file path and load completed set
    log_path = args.log_file or os.path.join(args.root, "da_en_progress.log")
    # Ensure log file exists (touch) so logging is always on by default
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8"):
            pass
    except Exception:
        pass
    completed = load_completed_set(log_path)
    tqdm.write(f"Logging progress to: {log_path}")
    if completed:
        tqdm.write(f"Loaded {len(completed)} completed entries from log")

    print(f"Found {len(csv_files)} CSV file(s) to process.")
    # Resolve output root
    out_root = os.path.abspath(args.out_root) if args.out_root else None
    for i, in_path in enumerate(tqdm(csv_files, desc="Files", unit="file"), 1):
        base, ext = os.path.splitext(in_path)
        if out_root:
            rel_path = os.path.relpath(base, args.root)
            out_base = os.path.join(out_root, rel_path)
            os.makedirs(os.path.dirname(out_base), exist_ok=True)
            suffix = ".en.jsonl" if args.out_format == "jsonl" else f".en{ext}"
            out_path = f"{out_base}{suffix}"
        else:
            suffix = ".en.jsonl" if args.out_format == "jsonl" else f".en{ext}"
            out_path = f"{base}{suffix}"
        norm_in = _normalize_path(in_path)
        if norm_in in completed:
            tqdm.write(f"[{i}/{len(csv_files)}] Skipping (already completed): {os.path.basename(in_path)}")
            continue
        tqdm.write(f"[{i}/{len(csv_files)}] Translating: {os.path.basename(in_path)} → {os.path.basename(out_path)}")
        try:
            process_csv(
                in_path=in_path,
                out_path=out_path,
                fields=args.fields,
                pipe=pipe,
                tokenizer=tokenizer,
                max_tokens=args.max_input_tokens,
                batch_size=args.batch_size,
                max_length=args.max_length,
                buffer_rows=args.buffer_rows,
                out_format=args.out_format,
            )
            append_completed(log_path, in_path)
        except Exception as e:
            print(f"  ⚠ Error processing {in_path}: {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

