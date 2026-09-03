import os
import glob
import json
import numpy as np
from metrics.classMetrics import RemovirtMetrics

class SequentialMetrics:
    """
    Evaluates the performance of multiple 2D and 3D medical image segmentation models
    sequentially, calculates clinical metrics (DSC, HD), and exports the results as JSON.
    """
    
    models_2d = [
        'swin_unetr', 
        'unet', 
        'unetr', 
        'attention_unet', 
        'unet++',                             
        'segformer', 
        'unetr++', 
        'uxlstm', 
        'nnmamba', 
        'segmamba'
    ]    

    models_3d = [
        'swin_unetr', 
        'unet', 
        'unetr', 
        'attention_unet', 
        'unet++',                             
        'segformer', 
        'unetr++', 
        'uxlstm', 
        'nnmamba', 
        'segmamba'
    ]             

    root_path = '/mnt/disco4t/alfredo/reconstructed_results'
    gt_img_path = "/mnt/disco4t/alfredo/data/amos22/imagesVa/"
    gt_lbl_path = "/mnt/disco4t/alfredo/data/amos22/labelsVa/"
    name_dataset = 'amos22'

    all_classes = [
        "background", "spleen", "right kidney", "left kidney", 
        "gall bladder", "esophagus", "liver", "stomach", 
        "arota", "postcava", "pancreas", "right adrenal gland", 
        "left adrenal gland", "duodenum", "bladder", "prostate/uterus"
    ]   

    def __call__(self):
        """
        Executes the evaluation pipeline for the defined 2D and 3D models.
        Iterates over the reconstructed predictions, computes metrics against 
        the ground truth, and accumulates the results.
        """
        metrics = RemovirtMetrics(self.all_classes)
        
        execution_plan = [
            (self.models_2d, 2, '.nii.gz'),
            (self.models_3d, 3, '.nii.gz')
        ]

        for models, dimensions, ext in execution_plan:
            for base_model_name in models:
                full_model_name = f"{base_model_name}_{dimensions}d"
                network_path = os.path.join(self.root_path, full_model_name)
                
                if not os.path.exists(network_path):
                    print(f"Directory not found: {network_path}. Skipping...")
                    continue

                print(f"\n--- Evaluating Model: {full_model_name} ---")
                
                dataset_accumulator = {cls: {"DSC": [], "HD": []} for cls in self.all_classes}

                for pred_filename in glob.glob(os.path.join(network_path, f'*{ext}')):
                    base_name_pred = os.path.basename(pred_filename)  
                    
                    gt_name = base_name_pred.replace("_Pred", "").replace(ext, "")
                    
                    img_path = os.path.join(self.gt_img_path, f"{gt_name}{ext}")
                    lbl_path = os.path.join(self.gt_lbl_path, f"{gt_name}{ext}")
                    
                    if not os.path.exists(lbl_path):
                        print(f"Label not found for {gt_name}. Skipping...")
                        continue

                    print(f"Calculating metrics for {gt_name}...")
                    
                    try:
                        results = metrics([img_path, lbl_path], pred_filename)
                        
                        for cls_name, cls_metrics in results.results.items():
                            if cls_metrics.get("DSC") is not None:
                                dataset_accumulator[cls_name]["DSC"].append(cls_metrics["DSC"])
                            if cls_metrics.get("HD") is not None:
                                dataset_accumulator[cls_name]["HD"].append(cls_metrics["HD"])
                    
                    except Exception as e:
                        print(f"Error processing {gt_name}: {e}")

                self._save_professional_summary(full_model_name, dimensions, dataset_accumulator)

    def _save_professional_summary(self, model_name, dimensions, accumulator):
        """
        Calculates the mean and standard deviation of the computed metrics 
        across the entire dataset and saves a formatted JSON summary.
        
        Args:
            model_name (str): The full name of the evaluated model.
            dimensions (int): The spatial dimension of the model (2 or 3).
            accumulator (dict): Dictionary containing the accumulated metric values.
        """
        summary = {
            "dataset": self.name_dataset,
            "model": model_name,
            "dimensions": f"{dimensions}D",
            "metrics_summary": {}
        }

        for cls_name, metrics in accumulator.items():
            summary["metrics_summary"][cls_name] = {}
            
            for metric_name in ["DSC", "HD"]:
                values = metrics[metric_name]
                if values:
                    summary["metrics_summary"][cls_name][metric_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "count": len(values),
                        "raw_values": [float(v) for v in values]
                    }
                else:
                    summary["metrics_summary"][cls_name][metric_name] = None

        output_file = f"jsons/metrics_{model_name.upper()}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Summary for {model_name} successfully saved in: {output_file}")

if __name__ == "__main__":
    SequentialMetrics()()