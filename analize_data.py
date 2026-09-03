import os
import nibabel as nib
import numpy as np

def calculate_background(label_dir):
    """
    Function to get information about the percentage of the background class in the dataset.

    Args:
        label_dir (str): string with the path to the dataset to evaluate.
    """
    total_vox = 0
    background_vox = 0
    files = 0
    
    print(f"Label directory: {label_dir}\n")
    
    for f in os.listdir(label_dir):
        if f.endswith('.nii.gz'):
            path = os.path.join(label_dir, f)
            
            try:
                img = nib.load(path)
                
                data = np.asarray(img.dataobj) 
                
                total_vol = data.size
                background_vol = np.sum(data == 0)
                
                total_vox += total_vol
                background_vox += background_vol
                files += 1
                
                print(f"File: {f} | Local background: {(background_vol/total_vol)*100:.2f}%")
                
            except Exception as e:
                print(f"Error in file {f}: {e}")
    
    if total_vox == 0:
        print(f"The are not files in the directory {label_dir} or the label files are empty.")
        return
        
    final_percent = (background_vox / total_vox) * 100
    
    print(f"There are {files} files")
    print(f"Total vox analized: {total_vox:,}")
    print(f"Total vox with background class: {background_vox:,}")
    print(f"Final Percentage: {final_percent:.4f} %")

if __name__ == "__main__":
    
    ruta_dataset_labels = "/mnt/disco4t/alfredo/data/amos22/labelsTr" # replace with your data path
    
    calculate_background(ruta_dataset_labels)