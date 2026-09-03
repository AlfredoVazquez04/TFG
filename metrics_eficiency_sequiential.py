import os
import json
import yaml
import argparse
import torch
import multiprocessing as mp

from model.utils import get_model 
from metrics.classMetrics import EfficiencyMetrics

def evaluate_single_model(base_model, dims, input_size, dataset, out_channels, batch_size, result_queue):
    """
    Evaluates a single model in an isolated process to measure its efficiency.
    
    Args:
        base_model (str): Base name of the architecture.
        dims (int): Spatial dimension (2 or 3).
        input_size (list): Size of the input tensor.
        dataset (str): Name of the dataset.
        out_channels (int): Number of output channels.
        batch_size (int): Batch size for the evaluation.
        result_queue (multiprocessing.Queue): Queue to send results back to the main process.
    """
    full_name = f"{base_model}_{dims}d"
    print(f"\nEvaluating Efficiency: {full_name}")
    
    try:
        args = argparse.Namespace()
        args.model = base_model
        args.dimension = f"{dims}d"
        args.dataset = dataset
        args.roi_size = tuple(input_size)
        args.in_channels = 1
        args.out_channels = out_channels

        config_path = f'config/{args.dataset}/{args.model}_{args.dimension}.yaml'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)
            for key, value in config.items():
                setattr(args, key, value)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(args).to(device)
        model.eval()
        
        results = EfficiencyMetrics()(model=model, size=input_size, dimension=dims, batch_size=batch_size)
        
        print(f"Batch Size: {batch_size} | Input: {input_size}")
        print(f"FLOPs: {results['flops']} | Params: {results['params']}")
        
        success_data = {
            "dimensions": f"{dims}D",
            "input_size": input_size,
            "batch_size": batch_size,
            "flops_formatted": results["flops"],
            "params_formatted": results["params"]
        }
        
        result_queue.put((full_name, True, success_data))
        
    except Exception as e:
        print(f"Error processing {full_name}: {e}")
        result_queue.put((full_name, False, str(e)))

def calculate_efficiency():
    """
    Executes the efficiency evaluation sequentially for all 2D and 3D models 
    defined in the execution plan and saves the results into a JSON file.
    """

    # add models names with image size if it's necessary
    models_2d = {
        'swin_unetr': [512, 512], 
        'unet': [512, 512], 
        'unetr': [512, 512], 
        'attention_unet': [512, 512], 
        'unet++': [512, 512], 
        'segformer': [512, 512], 
        'unetr++': [512, 512], 
        'uxlstm': [512, 512], 
        'nnmamba': [512, 512], 
        'segmamba': [256, 256],
    }
    
    # add models names with roi size if it's necessary
    models_3d = {
        'swin_unetr': [96, 96, 96], 
        'unet': [96, 96, 96], 
        'unetr': [96, 96, 96], 
        'attention_unet': [96, 96, 96], 
        'unet++': [96, 96, 96], 
        'segformer': [96, 96, 96], 
        'unetr++': [96, 96, 96], 
        'uxlstm': [96, 96, 96], 
        'nnmamba': [96, 96, 96], 
        'segmamba': [96, 96, 96],
    }

    dataset_name = 'amos22'
    out_channels = 16
    
    execution_plan = [(models_2d, 2), (models_3d, 3)]
    summary = {"dataset_target": dataset_name, "metrics_summary": {}}
    
    manager = mp.Manager()
    result_queue = manager.Queue()

    for models_dict, dims in execution_plan:
        batch_size = 8 if dims == 2 else 1
        
        for base_model, input_size in models_dict.items():
            p = mp.Process(
                target=evaluate_single_model, 
                args=(base_model, dims, input_size, dataset_name, out_channels, batch_size, result_queue)
            )
            p.start()
            p.join()
            
            if not result_queue.empty():
                full_name, success, data = result_queue.get()
                summary["metrics_summary"][full_name] = data if success else {"error": data}

    output_file = "jsons/efficiency_metrics_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nEfficiency summary successfully saved in: {output_file}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    calculate_efficiency()