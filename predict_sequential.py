import subprocess
import time
import json
from datetime import datetime
from multiprocessing import Process

from utils import check_gpu_memory



class SequentialPredictions:

    models_2d = {
            'swin_unetr': 1,          
            'unet': 1,                
            'unetr': 1,               
            'attention_unet': 1,      
            'unet++': 1,                             
            'segformer': 1,            
            'unetr++': 1,             
            'uxlstm': 1,              
            'nnmamba': 1,             
            'segmamba': 1,          
        }
    
    models_3d = { 
            'swin_unetr': 1,          
            'unet': 1,                
            'unetr': 1,               
            'attention_unet': 1,      
            'unet++': 1,                           
            'segformer': 1,            
            'unetr++': 1,             
            'uxlstm': 1,              
            'nnmamba': 1 ,            
            'segmamba': 1,              
        }
    
    dimensions = ['2d'] # '2d/3d',

    len_battery_test = 17
    

    def __call__(self):
        """Method to run the models in sequence and parallel.
        """        

        processes = []

        for dimension in self.dimensions:
            if dimension == '2d':
                models = self.models_2d
            else:
                models = self.models_3d

            for model in list(models.keys()):
                
                gpu_memory = check_gpu_memory() # Verify GPU memory
                while gpu_memory < 10024:  # Wait until GPU memory is available  9124
                    print("Need more GPU memory. Waiting...")
                    time.sleep(55)  
                    gpu_memory = check_gpu_memory()
               
                print(f"\nTest {model} ({dimension})...")  # Start model battery test
                try:
                    process = Process(
                        target=self.run_model, 
                        args=(model, dimension, str(models[model])))
                    process.start()
                    processes.append(process)

                except: pass
                time.sleep(30)  
                
                
        for process in processes:
            process.join()

        print("All models are tested")
        

    @staticmethod
    def run_model(model, dimension, run_version):
        """Function to run the model to predict the battery test.
        When the prediction is finished, the time with more parameters are saved in a json file.
        The place for the json file is in the respective folder results.

        Args:
            model (str): Name of the model
            dimension (str): Number of dimensions (2d or 3d)
            run_version (str): Version of the model
        """   
        path = '/mnt/disco4t/alfredo/resultsNew/'   
        args = [
            '--mode', 'Predict',
            '--dataset', 'amos22',
            '--cache_rate', '0.0',
            '--spatial_dims', '2',
            '--out_channels', '16',
            '--data_dir', '/mnt/disco4t/alfredo/data/amos22_2d',
            '--roi_size', '512', '512', # 96x96x96 or 512x512
            '--model', model, 
            '--dimension', dimension,
            '--path_prediction', path,
            ]
        
        cmd = ['python3', 'train.py'] + args
        try:
            start_time = datetime.now()
            subprocess.run(cmd)
            end_time = datetime.now()
            elapsed_time = end_time - start_time
            
            result = {
                'model': model,
                'dimension': dimension,
                'run_version': run_version,
                'total_batteries_tested': SequentialPredictions.len_battery_test,
                'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
                'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
                'elapsed_time': str(elapsed_time),
            }

            with open(path+model+'_'+dimension+'/time_prediction_set.json', 'w') as json_file:
                json.dump(result, json_file, indent=4)
                json_file.write(',\n') 

        except Exception as e: print(e)

    

if __name__ == "__main__":
    SequentialPredictions()()
