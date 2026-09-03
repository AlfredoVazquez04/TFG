import json
from metrics.classMetrics import StatisticsMetrics

def extract_organ_means(json_path: str, metric: str = "DSC") -> list:
    """
    Reads a JSON metrics file and extracts a list of mean values 
    for all evaluated organs (excluding the background) for a specific metric.

    Args:
        json_path (str): Path to the JSON file containing the metrics summary.
        metric (str, optional): The metric to extract (e.g., "DSC" or "HD"). Defaults to "DSC".

    Returns:
        list: A list containing the mean values of the specified metric for each organ.
    """
    with open(json_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
        
    values = []
    
    for organ, metrics in data.get("metrics_summary", {}).items():
        if organ == "background":
            continue 
            
        if metrics and metric in metrics:
            values.append(metrics[metric]["mean"])
            
    return values

def main():
    """
    Main execution pipeline. Extracts DSC and HD metrics for the best 
    2D and 3D models and performs statistical hypothesis testing to evaluate 
    the significance of the performance differences.
    """
    best_2d_path = "jsons/metrics_SEGFORMER_2D.json"  # complete with the best 2D model
    best_3d_path = "jsons/metrics_UXLSTM_3D.json"     # complete with the best 3D model
    
    dice_2d = extract_organ_means(best_2d_path, metric="DSC")
    dice_3d = extract_organ_means(best_3d_path, metric="DSC")
    
    hd_2d = extract_organ_means(best_2d_path, metric="HD")
    hd_3d = extract_organ_means(best_3d_path, metric="HD")
    
    statistics = StatisticsMetrics(alpha=0.05)
    
    statistics(dice_2d, dice_3d, metric_name="Dice (DSC)")
    statistics(hd_2d, hd_3d, metric_name="Hausdorff (HD95)")

if __name__ == "__main__":
    main()