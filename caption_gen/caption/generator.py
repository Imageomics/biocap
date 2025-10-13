import os
import sys
import webdataset as wds
from tqdm import tqdm
from typing import Dict, Optional
from vllm import LLM, SamplingParams

from caption_gen.utils.text_utils import safe_get_text, extract_class_from_taxon
from caption_gen.data.data_loader import find_wiki_caption
from caption_gen.caption.prompts import build_prompt


def update_single_tar_with_captions(
    input_tar: str,
    output_tar: str,
    llm: LLM,
    sampling_params: SamplingParams,
    class_mapping: Dict[str, Dict[str, str]],
    batch_size: int = 16,
    wiki_lookup: Optional[Dict[str, str]] = None,
    max_wiki_chars: int = 1000
):
    dataset = wds.WebDataset(input_tar).decode("pil")

    # Prepare batches for processing
    def create_batches(dataset, batch_size):
        current_batch = []
        for sample in dataset:
            current_batch.append(sample)
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []
        if current_batch:
            yield current_batch

    with wds.TarWriter(output_tar) as sink:
        batch_list = list(create_batches(dataset, batch_size))
        total_samples = 0

        for batch in tqdm(
            batch_list,
            desc=f"Processing {os.path.basename(input_tar)}",
            ncols=100,
            unit="batch",
            total=len(batch_list),
            file=sys.stdout,
            disable=not sys.stdout.isatty(),
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        ):
            # Prepare batch data for vLLM
            batch_prompts = []
            samples_to_process = []
            caption_types = []

            for sample in batch:
                sample = dict(sample)
                image = sample.get("jpg", None)
                if image is None:
                    sink.write(sample)
                    continue

                species_name = safe_get_text(sample, "common_name.txt")
                if not species_name:
                    species_name = safe_get_text(sample, "scientific_name.txt")
                if not species_name:
                    species_name = safe_get_text(sample, "taxonomic_name.txt")

                # Scientific name for wiki lookup
                scientific_name = safe_get_text(sample, "scientific_name.txt") or safe_get_text(sample, "taxonomic_name.txt")

                # Extract class from taxonomy for prompt templates and examples
                taxon_tag = safe_get_text(sample, "taxontag.txt")
                extracted_class = extract_class_from_taxon(taxon_tag)

                class_info = class_mapping.get(extracted_class.lower(), {})

                wiki_excerpt = ""
                if wiki_lookup:
                    wiki_text = find_wiki_caption(scientific_name, wiki_lookup)
                    if wiki_text:
                        wiki_excerpt = wiki_text[:max(0, int(max_wiki_chars))]

                prompt_text, config = build_prompt(
                    species_name=species_name,
                    scientific_name=scientific_name,
                    class_info=class_info,
                    wiki_excerpt=wiki_excerpt,
                    max_wiki_chars=max_wiki_chars
                )

                # Format for vLLM multimodal input
                batch_prompts.append({
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": image}
                })
                samples_to_process.append(sample)
                caption_types.append(config)

            if not batch_prompts:
                continue

            # Generate captions using vLLM batch processing
            outputs = llm.generate(batch_prompts, sampling_params)

            # Write samples with new captions
            for i, output in enumerate(outputs):
                sample = samples_to_process[i]
                sample = dict(sample)
                caption = output.outputs[0].text.strip()
                config = caption_types[i]
                sample[config['output_key']] = caption
                sink.write(sample)
                total_samples += 1

