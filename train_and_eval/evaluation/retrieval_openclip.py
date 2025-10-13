import torch
import argparse
import os
import pandas as pd

import open_clip

from PIL import Image

class CustomDataset:
    def __init__(self, csv_file, image_folder):
        self.df = pd.read_csv(csv_file)
        self.image_folder = image_folder
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['id']
        caption = row['captions']
        
        image_path = os.path.join(self.image_folder, f"{image_id}.jpg")
        image = Image.open(image_path).convert('RGB')
        
        return image, caption

def eval_custom_dataset(model, dataset, preprocess, tokenizer, device, args):
    image_features = []
    text_features = []
    pred_true = 0
    
    with torch.no_grad():
        for index in range(len(dataset)):
            image, caption = dataset[index]
            
            # Process image
            image_input = preprocess(image).unsqueeze(0).to(device)
            image_feature = model.encode_image(image_input)
            if isinstance(image_feature, tuple):
                image_feature = image_feature[0]
            image_features.append(image_feature)

            text_input = tokenizer([f"a photo of{caption}"]).to(device)
            text_feature = model.encode_text(text_input)
            if isinstance(text_feature, tuple):
                text_feature = text_feature[0]
            text_features.append(text_feature)
            
            if (index + 1) % 2000 == 0:
                print(f"{index + 1}: {len(dataset)}")

        image_features = torch.stack(image_features).squeeze()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = torch.stack(text_features).squeeze()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T
        num_samples = len(dataset)

        print("I2T (Image to Text) Results:")
        # R@1
        pred_true = 0
        for i in range(num_samples):
            pred = similarity[i]
            b = pred.argsort()[-1:]
            if i in b:
                pred_true += 1
        print(f"R@1: {pred_true / num_samples:.4f}")

        # R@5
        pred_true = 0
        for i in range(num_samples):
            pred = similarity[i]
            b = pred.argsort()[-5:]
            if i in b:
                pred_true += 1
        print(f"R@5: {pred_true / num_samples:.4f}")

        # R@10
        pred_true = 0
        for i in range(num_samples):
            pred = similarity[i]
            b = pred.argsort()[-10:]
            if i in b:
                pred_true += 1
        print(f"R@10: {pred_true / num_samples:.4f}")

        print("T2I (Text to Image) Results:")
        similarity = similarity.T
        
        # R@1
        pred_true = 0
        for i in range(num_samples):
            pred = similarity[i]
            b = pred.argsort()[-1:]
            if i in b:
                pred_true += 1
        print(f"R@1: {pred_true / num_samples:.4f}")

        # R@5
        pred_true = 0
        for i in range(num_samples):
            pred = similarity[i]
            b = pred.argsort()[-5:]
            if i in b:
                pred_true += 1
        print(f"R@5: {pred_true / num_samples:.4f}")

        # R@10
        pred_true = 0
        for i in range(num_samples):
            pred = similarity[i]
            b = pred.argsort()[-10:]
            if i in b:
                pred_true += 1
        print(f"R@10: {pred_true / num_samples:.4f}")


      

def eval_model(args):
    # Model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    model = model.to(args.device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)
        
    dataset = CustomDataset(args.csv_file, args.image_folder)
    eval_custom_dataset(model, dataset, preprocess, tokenizer, args.device, args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="ViT-B-16")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--csv-file", type=str, default="path/to/csv_file")
    parser.add_argument("--image-folder", type=str, default="path/to/image_folder")
    
    args = parser.parse_args()

    eval_model(args)