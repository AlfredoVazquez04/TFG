import glob
import os
import json

all_classes = [
    "background", "spleen", "right kidney", "left kidney", 
    "gall bladder", "esophagus", "liver", "stomach", 
    "arota", "postcava", "pancreas", "right adrenal gland", 
    "left adrenal gland", "duodenum", "bladder", "prostate/uterus"
]

dimension = '3D'

def get_performance(metrics_files):
    """
    Extracts mean and std (DSC and HD) values from JSON files,
    calculates averages and a weighted score.

    Args:
        metrics_files (list): list of strings with the paths to the json files.
    """
    results = {}

    print(f"Performance per {dimension} architecture (excluding background)\n")

    for file_path in metrics_files:
        with open(file_path, mode="r", encoding="utf-8") as read_file:
            data = json.load(read_file)

        model_name = data.get("model", os.path.basename(file_path).replace('.json', ''))
        metrics_summary = data.get("metrics_summary", {})

        class_means = []
        class_stds = []
        hd_means = []
        hd_stds = []

        for cls in all_classes:
            if cls == "background":
                continue 

            if cls in metrics_summary:
                cls_data = metrics_summary[cls]

                if isinstance(cls_data, dict) and "DSC" in cls_data:
                    dsc_data = cls_data["DSC"]
                    hd_data = cls_data["HD"]

                    if isinstance(dsc_data, dict):
                        class_means.append(dsc_data.get("mean", 0.0))
                        class_stds.append(dsc_data.get("std", 0.0))
                        hd_means.append(hd_data.get("mean", 0.0))
                        hd_stds.append(hd_data.get("std", 0.0))
                    else:
                        class_means.append(0.0)
                        class_stds.append(0.0)
                        hd_means.append(0.0)
                        hd_stds.append(0.0)
                else:
                    class_means.append(0.0)
                    class_stds.append(0.0)
                    hd_means.append(0.0)
                    hd_stds.append(0.0) 

        if class_means:
            overall_mean = sum(class_means) / len(class_means)
            overall_std = sum(class_stds) / len(class_stds)
            overall_hd_mean = sum(hd_means) / len(hd_means)
            overall_hd_std = sum(hd_stds) / len(hd_stds)

            weighted_score = (0.5 * overall_mean) - (0.5 * overall_hd_mean)

            results[model_name] = {
                'mean': overall_mean, 
                'std': overall_std, 
                'hd_mean': overall_hd_mean, 
                'hd_std': overall_hd_std,
                'score': weighted_score
            }

            print(f"Model: {model_name}")
            print(f"Evaluated organs: {len(class_means)}")
            print(f"Average DSC: {overall_mean:.4f} ± {overall_std:.4f}")
            print(f"Average HD: {overall_hd_mean:.4f} ± {overall_hd_std:.4f} mm")
            print(f"Weighted Score: {weighted_score:.4f}\n")
        else:
            print(f"No DSC metrics found in {file_path}\n")

    return results

if __name__ == "__main__":
    
    root_dir = '.' # replace with your directory path if needed
    path_pattern = os.path.join(root_dir, f'*{dimension}.json')
    metrics_models = glob.glob(path_pattern)

    if not metrics_models:
        print(f"Files with pattern '*{dimension}.json' not found.")
    else:
        performance_results = get_performance(metrics_models)

        if performance_results:
            best_model = max(performance_results, key=lambda k: performance_results[k]['score'])
            
            best_mean = performance_results[best_model]['mean']
            best_std = performance_results[best_model]['std']
            hd_mean = performance_results[best_model]['hd_mean']
            hd_std = performance_results[best_model]['hd_std']
            best_score = performance_results[best_model]['score']

            print(f"\nBest {dimension} Architecture")
            print(f"Model: {best_model.upper()}")
            print(f"Average DSC: {best_mean:.4f} ± {best_std:.4f}")
            print(f"Average HD: {hd_mean:.4f} ± {hd_std:.4f} mm")
            print(f"Combined Score: {best_score:.4f}")
        else:
            print("Not enough data to evaluate the best model.")