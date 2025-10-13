"""This script runs evals for CLIP models on INQUIRE-Rerank, the reranking task.
The data is automatically loaded from HuggingFace Hub, so you don't need to download
anything yourself to run this evaluation."""
import os
import json
import shutil

# Set HuggingFace cache to project's data folder BEFORE importing datasets
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cache_dir = os.path.join(project_root, 'data', 'inquire')
os.environ['HF_DATASETS_CACHE'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
import argparse
import open_clip

from .metrics import MetricAverage, compute_retrieval_metrics

# Command line argument parser
parser = argparse.ArgumentParser(description='Run retrieval evaluation.')
parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                    help="Dataset split to evaluate on. Options: 'val', 'test'. Default is 'test'.")
parser.add_argument('--model-name', type=str, default='hf-hub:imageomics/biocap',
                    help="Model name or HuggingFace hub path. Default is 'hf-hub:imageomics/biocap'.")
parser.add_argument('--pretrained', type=str, default='',
                    help="Pretrained weights path. Default is empty string.")
args = parser.parse_args()

split = args.split
save_results_path = f'results_rerank_with_clip_{split}.csv'

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load INQUIRE-Rerank from HuggingFace
dataset = load_dataset("evendrow/INQUIRE-Rerank", split=('validation' if split == 'val' else 'test'))
queries = np.unique(dataset['query']).tolist()

batch_size = 256
num_workers = 8

all_models = {
    'biocap': {'model_name': args.model_name, 'pretrained': args.pretrained},
}

results = []
for title, model_config in all_models.items():
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_config['model_name'],
        pretrained=model_config['pretrained']
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_config['model_name'])

    # Efficiently compute image embeddings in batches
    def collate_transform(examples):
        pixel_values = torch.cat([preprocess(ex["image"]).unsqueeze(0) for ex in examples])
        ids = [ex['inat24_image_id'] for ex in examples]
        return pixel_values, ids
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_transform, num_workers=num_workers)

    image_emb_cache = {}
    for images, ids in tqdm(dataloader, total=len(dataset)//batch_size):
        with torch.no_grad(), torch.autocast(device):
            image_embs = model.encode_image(images.to(device)).cpu()
            image_embs /= image_embs.norm(dim=-1, keepdim=True)
        image_emb_cache.update(dict(zip(ids, image_embs)))


    # Score images for each query by embedding similarity
    metrics_avg = MetricAverage()
    for query in queries:
        query_ds = dataset.select(np.argwhere(np.asarray(dataset['query']) == query).squeeze())

        text = tokenizer(query).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_emb = model.encode_text(text).squeeze().cpu()
            text_emb /= text_emb.norm(dim=-1, keepdim=True)

        image_embs = torch.stack([image_emb_cache[image_id] for image_id in query_ds['inat24_image_id']])
        y_pred = (image_embs.float() @ text_emb.float()).numpy()
        y_true = np.asarray(query_ds['relevant'])

        pr, rec, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=sum(y_true))
        metrics_avg.update([ap*100, ndcg*100, mrr])
        results.append(dict(model=title, query=query, ap=ap*100, ndcg=ndcg*100, mrr=mrr))

    ap, ndcg, mrr = metrics_avg.avg
    print(f'{title:30s}\t{ap:.1f}\t{ndcg:.1f}\t{mrr:.2f}')

results_df = pd.DataFrame.from_dict(results)
pd.options.display.float_format = ' {:,.2f}'.format
print(results_df.groupby('model').agg({'ap': 'mean', 'ndcg': 'mean', 'mrr': 'mean'}).sort_values('ap'))

results_df.to_csv(save_results_path)
print("All done! Saved results to", save_results_path)