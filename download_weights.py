import os
import argparse
import urllib.request
import sys

def download_file(url, destination, force=False):
    """
    Downloads a file from a URL with a progress visualization.
    
    Args:
        url: Source URL for the file
        destination: Local path where the file will be saved
        force: Boolean flag to override existing files if True
    """
    # Check for existing file
    if os.path.exists(destination) and not force:
        print(f"File already exists at {destination}. Use --force to override.")
        return
    
    # Create directory structure if needed
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Initialize download
    print(f"Starting download from {url}")
    print(f"Saving to {destination}")
    
    def report_progress(block_num, block_size, total_size):
        """Displays download progress with a visual progress bar"""
        if total_size > 0:
            # Calculate completion percentage
            percent = min(block_num * block_size * 100 / total_size, 100)
            
            # Generate progress bar visualization
            bar_length = 30
            filled_length = int(bar_length * percent / 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            # Calculate and display download stats
            downloaded_mb = block_num * block_size / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            sys.stdout.write(f"\r|{bar}| {percent:.1f}% ({downloaded_mb:.1f}MB/{total_mb:.1f}MB)")
            sys.stdout.flush()
            
            # Add newline when complete
            if block_num * block_size >= total_size:
                sys.stdout.write("\n")
    
    try:
        # Execute the download with progress tracking
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print(f"\nDownload complete: {destination}")
    except Exception as e:
        # Error handling
        print(f"\nDownload failed: {e}")
        if os.path.exists(destination):
            os.remove(destination)  # Remove partial downloads
        raise

def main(args):
    """
    Main function for model weight acquisition.
    Handles downloading the appropriate model weights based on selection.
    Different models offer varying tradeoffs between size and accuracy.
    """
    # Model weights repositories
    weights_urls = {
        "vit_t": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",  # MobileSAM (~40MB)
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",  # SAM base (~375MB)
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",  # SAM huge (~2.5GB)
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",  # SAM large (~1.2GB)
    }
    
    # Process model selection options
    if args.model == "mobile_sam":
        # Map mobile_sam option to vit_t (MobileSAM)
        model_types = ["vit_t"]
    elif args.model == "all":
        # Select all available models
        model_types = list(weights_urls.keys())
    elif args.model in weights_urls.keys():
        # Direct model selection
        model_types = [args.model]
    else:
        print(f"Error: Model '{args.model}' not recognized.")
        print(f"Available models: {', '.join(list(weights_urls.keys()) + ['mobile_sam', 'all'])}")
        print(f"Note: 'mobile_sam' (or 'vit_t') is recommended for this project due to its efficiency.")
        return
    
    # Process each selected model
    for model_type in model_types:
        print(f"\nDownloading {model_type} model...")
        url = weights_urls[model_type]
        
        if model_type == "vit_t":
            # Apply correct filename for MobileSAM to maintain compatibility
            destination = os.path.join(args.output_dir, "mobile_sam.pt")
        else:
            destination = os.path.join(args.output_dir, f"{model_type}.pth")
        
        download_file(url, destination, args.force)
        print(f"Successfully downloaded {model_type} model to {destination}")
    
    print(f"\nAll requested model weights have been downloaded successfully.")

if __name__ == "__main__":
    # Entry point for direct script execution
    parser = argparse.ArgumentParser(description="Download pre-trained SAM or MobileSAM weights")
    parser.add_argument('--model', type=str, default='vit_t', 
                        choices=['vit_t', 'vit_b', 'vit_h', 'vit_l', 'mobile_sam', 'all'],
                        help='Model type: vit_t/mobile_sam (MobileSAM, ~40MB), vit_b (SAM base, ~375MB), vit_h (SAM huge, ~2.5GB), vit_l (SAM large, ~1.2GB), or all')
    parser.add_argument('--output_dir', type=str, default='weights', 
                        help='Directory for saving downloaded model weights')
    parser.add_argument('--force', action='store_true',
                        help='Force download even if files already exist')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Execute main function
    main(args) 