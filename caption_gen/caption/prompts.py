from typing import Dict


def build_prompt(
    species_name: str,
    scientific_name: str,
    class_info: Dict[str, str],
    wiki_excerpt: str = "",
    max_wiki_chars: int = 1000
) -> str:

    default_examples = (
        "1. Amanita muscaria displays a vibrant red cap with white warts, striated margins, and a thick white stalk ending in a bulbous volva. "
        "2. Athyrium filix-femina features arching, finely divided fronds with serrated edges, a dark midrib, and a slightly hairy surface. "
        "3. Papilio machaon exhibits vivid yellow forewings with black vein patterns, long-tailed hindwings with blue bands and orange eyespots, and a slim black abdomen. "
    )
    
    config = {
        'examples': class_info.get('description', default_examples),
        'word_limit': 20,
        'output_key': 'caption.txt'
    }
    
    
    prompt_parts = [
        "You are a biologist describing organisms based strictly on what is visible in the image.",
        "Your goal is to produce a concise caption that highlights diagnostic, image-based traits.",
        "Focus primarily on anatomical structures (e.g., color, shape, pattern, texture, position).",
        "If clearly visible, you may mention substrate, scale cues, or explicit interactions.",
        "Use precise biological terminology. Avoid vague or generic words.\n\n",
        f"Examples of good captions:\n{config['examples']}\n\n",
    ]

    if wiki_excerpt:
        prompt_parts.append(
            f"Reference excerpt about {scientific_name or species_name} "
            f"(use only to standardize correct terms that match visible traits; "
            f"do not copy text; do not add traits not visible in the image):\n{wiki_excerpt}\n\n"
        )

    prompt_parts.extend([
        f"The caption must not exceed {config['word_limit']} words.",
        f'Include the species name "{species_name}" naturally in the sentence.',
        "Priority order: (1) the most diagnostic visible trait, "
        "(2) a secondary distinctive trait, "
        "(3) a contextual detail only if it strengthens identification.",
        f"For the following image of a {species_name}, write a single, concise sentence describing its visible traits."
    ])
    
    prompt_content = "".join(prompt_parts)
    prompt_text = f"<|im_start|>user\n<image>\n{prompt_content}<|im_end|>\n<|im_start|>assistant\n"
    
    return prompt_text, config