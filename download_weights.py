import os
import requests
import argparse
from tqdm import tqdm
from pathlib import Path

def download_file(url, dest_path):
    """
    Download a file from a URL to the specified destination path with a progress bar.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def main(args):
    """
    Download pre-trained MobileSAM weights.
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define URLs for different model weights
    model_urls = {
        'mobile_sam': 'https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt',
    }
    
    # Download the weights
    if args.model == 'all':
        for model_name, url in model_urls.items():
            output_path = os.path.join(args.output_dir, f"{model_name}.pt")
            if os.path.exists(output_path) and not args.force:
                print(f"{model_name} weights already exist at {output_path}. Skipping download.")
                continue
                
            print(f"Downloading {model_name} weights...")
            download_file(url, output_path)
            print(f"Downloaded {model_name} weights to {output_path}")
    else:
        if args.model not in model_urls:
            raise ValueError(f"Model {args.model} not found. Available models: {', '.join(model_urls.keys())}")
            
        output_path = os.path.join(args.output_dir, f"{args.model}.pt")
        if os.path.exists(output_path) and not args.force:
            print(f"{args.model} weights already exist at {output_path}. Skipping download.")
            return
            
        print(f"Downloading {args.model} weights...")
        download_file(model_urls[args.model], output_path)
        print(f"Downloaded {args.model} weights to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pre-trained MobileSAM weights")
    parser.add_argument("--model", type=str, default="mobile_sam", choices=["mobile_sam", "all"], 
                        help="Model weights to download")
    parser.add_argument("--output_dir", type=str, default="weights", 
                        help="Output directory for downloaded weights")
    parser.add_argument("--force", action="store_true", 
                        help="Force download even if weights already exist")
    
    args = parser.parse_args()
    main(args) 