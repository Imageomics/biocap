import os
import time
import argparse
import glob
from typing import List
from vllm import LLM, SamplingParams
from caption_gen.data.data_loader import load_wiki_captions, load_class_examples
from caption_gen.caption.generator import update_single_tar_with_captions


def get_input_files(args) -> List[str]:
    """Determine input files to process based on arguments."""
    input_files = []

    if args.input_dir:
        # Directory mode: find all tar files
        tar_pattern = os.path.join(args.input_dir, "*.tar")
        input_files = sorted(glob.glob(tar_pattern))
        if args.output_dir is None:
            args.output_dir = args.input_dir
        os.makedirs(args.output_dir, exist_ok=True)

    elif args.input_tars:
        # File(s) mode: one or more specific files
        input_files = args.input_tars
        if len(input_files) == 1 and args.output_tar is None:
            input_dir = os.path.dirname(input_files[0])
            input_name = os.path.basename(input_files[0])
            name_without_ext = os.path.splitext(input_name)[0]
            args.output_tar = os.path.join(input_dir, f"{name_without_ext}_with_captions.tar")
        else:
            if args.output_dir is None:
                args.output_dir = os.path.dirname(input_files[0]) if input_files else "."
            os.makedirs(args.output_dir, exist_ok=True)

    return input_files


def initialize_model(args):
    print(f"Loading model {args.model_name} with tensor_parallel_size={args.tensor_parallel_size}")
    llm = LLM(
        model=args.model_name,
        quantization=(None if (args.quantization is None or str(args.quantization).lower() in ["", "none"]) else args.quantization),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=False,
        disable_log_stats=True,
    )
    print("Model loaded successfully")
    return llm


def process_files(args, input_files: List[str], llm, sampling_params, class_mapping, wiki_lookup):
    """Process all input files."""
    for i, input_tar in enumerate(input_files):
        if len(input_files) == 1 and args.output_tar:
            output_tar = args.output_tar
        else:
            input_name = os.path.basename(input_tar)
            name_without_ext = os.path.splitext(input_name)[0]
            output_tar = os.path.join(args.output_dir, f"{name_without_ext}_with_captions.tar")


        file_start_time = time.time()
        update_single_tar_with_captions(
            input_tar,
            output_tar,
            llm,
            sampling_params,
            class_mapping,
            batch_size=args.batch_size,
            wiki_lookup=wiki_lookup,
            max_wiki_chars=args.max_wiki_chars
        )
        file_elapsed = time.time() - file_start_time
        print(f"File {i+1}/{len(input_files)} completed in {file_elapsed:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Add captions to WebDataset tar files using direct vLLM inference with taxonomy-based examples.")

    # Input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=str, help="Directory containing tar files to process.")
    input_group.add_argument("--input_tars", type=str, nargs='+', help="One or more tar files to process.")

    parser.add_argument("--output_tar", type=str, help="Path to output tar file (single file mode only).")
    parser.add_argument("--output_dir", type=str, help="Output directory (batch mode, defaults to input_dir).")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3-38B-AWQ", help="Model name for caption generation.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for caption generation.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism.")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Maximum model length.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling parameter.")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum tokens to generate.")
    parser.add_argument("--format_examples", type=str, default="path/to/format_examples.parquet", help="Parquet file containing class-specific format examples.")
    parser.add_argument("--quantization", type=str, default="awq", help="Quantization method for vLLM (e.g., 'awq', 'gptq'). Use empty string or 'none' to disable.")
    parser.add_argument("--wiki_data", type=str, default="path/to/uuid_caption_description.parquet", help="Parquet file for UUID-based wiki lookup.")
    parser.add_argument("--max_wiki_chars", type=int, default=600, help="Max number of characters from the wiki excerpt to include in the prompt.")

    args = parser.parse_args()
    input_files = get_input_files(args)
    class_mapping = load_class_examples(args.format_examples)
    wiki_lookup = load_wiki_captions(args.wiki_data)
    llm = initialize_model(args)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "</s>", "<|endoftext|>"],
    )

    process_files(args, input_files, llm, sampling_params, class_mapping, wiki_lookup)


if __name__ == "__main__":
    main()
