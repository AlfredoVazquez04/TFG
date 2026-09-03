import os
import glob
import re
import numpy as np
import nibabel as nib
from collections import defaultdict
from tqdm import tqdm

def reconstruct_models(results_dir: str, gt_dir: str, output_base_dir: str, axis: int = 2):
    """
    Reconstructs 3D NIfTI volumes from 2D slice predictions.
    
    Iterates through 2D model result directories, groups individual .npy slice 
    predictions by patient ID, stacks them along the specified axis, and saves 
    the reconstructed volumes using the affine matrix from the original ground truth.

    Args:
        results_dir (str): Path to the directory containing 2D model predictions.
        gt_dir (str): Path to the directory containing original 3D ground truth labels.
        output_base_dir (str): Destination path for the reconstructed 3D volumes.
        axis (int, optional): The axis along which the slices were extracted (default is 2, Z-axis).
    """
    model_dirs = [d for d in glob.glob(os.path.join(results_dir, "*_2d")) if os.path.isdir(d)]
    print(f"Found {len(model_dirs)} 2D models for reconstruction.\n")

    pattern = re.compile(r'(amos_\d+)_slice_(\d+)_Pred\.npy')

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"{'='*40}\nReconstructing model: {model_name}\n{'='*40}")
        
        out_model_dir = os.path.join(output_base_dir, model_name)
        os.makedirs(out_model_dir, exist_ok=True)

        npy_files = glob.glob(os.path.join(model_dir, "*.npy"))
        patients = defaultdict(list)
        
        for f in npy_files:
            filename = os.path.basename(f)
            match = pattern.search(filename)
            if match:
                patient_id = match.group(1)
                slice_idx = int(match.group(2))
                patients[patient_id].append((slice_idx, f))
                
        if not patients:
            print(f"[WARNING] No valid .npy files found in {model_dir}. Skipping...")
            continue

        for patient_id, slices in tqdm(patients.items(), desc=f"Processing {model_name} patients"):
            gt_path = os.path.join(gt_dir, f"{patient_id}.nii.gz")
            
            if not os.path.exists(gt_path):
                gt_path = os.path.join(gt_dir, f"{patient_id}.nii")
                if not os.path.exists(gt_path):
                    print(f"\n[ERROR] Ground truth not found for {patient_id}. Skipping...")
                    continue
            
            img_gt = nib.load(gt_path)
            vol_shape = img_gt.shape
            affine = img_gt.affine  
            
            pred_vol = np.zeros(vol_shape, dtype=np.uint8)
            
            for slice_idx, slice_path in slices:
                slice_data = np.load(slice_path)
                slice_data = np.squeeze(slice_data)
                
                try:
                    if axis == 0:
                        pred_vol[slice_idx, :, :] = slice_data
                    elif axis == 1:
                        pred_vol[:, slice_idx, :] = slice_data
                    else: 
                        pred_vol[:, :, slice_idx] = slice_data
                except IndexError:
                    print(f"\n[ERROR] Index out of bounds for {patient_id}, slice {slice_idx}. Check orientation.")
                    continue
            
            out_img = nib.Nifti1Image(pred_vol, affine)
            out_filepath = os.path.join(out_model_dir, f"{patient_id}_Pred.nii.gz")
            nib.save(out_img, out_filepath)

if __name__ == "__main__":
    
    DIR_RESULTS_2D = "/mnt/disco4t/alfredo/resultsNew"
    DIR_GT_ORIGINAL = "/mnt/disco4t/alfredo/data/amos22/labelsVa" 
    DIR_OUTPUT_3D = "/mnt/disco4t/alfredo/reconstructed_results"
    
    SLICE_AXIS = 2
    
    print("Initializing 2D -> 3D mass reconstruction pipeline...")
    reconstruct_models(DIR_RESULTS_2D, DIR_GT_ORIGINAL, DIR_OUTPUT_3D, axis=SLICE_AXIS)
    print("\nPipeline completed successfully.")