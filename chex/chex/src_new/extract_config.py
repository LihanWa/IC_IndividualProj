import torch
import argparse
import json
import os
from pprint import pprint

def extract_config_from_checkpoint(checkpoint_path, output_path=None):
    """
    Extract configuration information from checkpoint file
    
    Args:
        checkpoint_path: Checkpoint file path
        output_path: Optional, output configuration file path
    
    Returns:
        Configuration dictionary
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
    
    # Extract configuration information
    if 'experiment_config' in checkpoint:
        config = checkpoint['experiment_config']
        print("Configuration extracted successfully")
    else:
        print("'experiment_config' field not found in checkpoint")
        # Try to find other possible configuration fields
        possible_config_keys = [k for k in checkpoint.keys() if 'config' in k.lower()]
        if possible_config_keys:
            print(f"Found possible configuration fields: {possible_config_keys}")
            config = checkpoint[possible_config_keys[0]]
        else:
            print("No configuration information found")
            return None
    
    # Save configuration to file
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Configuration saved to: {output_path}")
    
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract configuration information from checkpoint")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint file path")
    parser.add_argument("--output_path", type=str, default="extracted_config.json", help="Output configuration file path")
    parser.add_argument("--print", action="store_true", help="Whether to print configuration information")
    
    args = parser.parse_args()
    
    config = extract_config_from_checkpoint(args.ckpt_path, args.output_path)
    
    if config and args.print:
        print("\nExtracted configuration information:")
        pprint(config) 