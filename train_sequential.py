import subprocess
import time
import os
import json
from datetime import datetime
from multiprocessing import Process

from utils import check_gpu_memory



class SequentialTrain:

    models_3d = {
                # 'swin_unetr': 1,          # finished
                # 'unet': 1,                # finished
                # 'unetr': 1,               # finished
                # 'attention_unet': 1,      # finished
                # 'unet++': 1,              # finished               
                # 'segformer': 1,           # finished 
                # 'unetr++': 1,             # finished
                # 'uxlstm': 1,              # finished
                # 'nnmamba': 1 ,            # finished
                # 'segmamba': 1,            # finished  
                # 'medsam' : 1,             # TODO: implement
            }

    
    dimensions = ['3d'] # '2d',

    dataset = 'amos22'
    

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
                while gpu_memory < 15124:  # Wait until GPU memory is available  9124
                    print("Need more GPU memory. Waiting...")
                    time.sleep(55)  
                    gpu_memory = check_gpu_memory()
               
                print(f"\nTrain {model} ({dimension})...")  # Start model battery test
                try:
                    '''
                    process = Process(
                        target=self.run_model, 
                        args=(model, dimension, str(models[model]), self.dataset))
                    process.start()
                    processes.append(process)
                    '''
                    self.run_model(model, dimension, str(models[model]), self.dataset)
                except Exception as e:
                    print(f"Error found: {e}")

                time.sleep(60)  
                
                
        for process in processes:
            process.join()

        print("All models are tested")

    @staticmethod
    def run_model(model, dimension, batch_size, dataset):
        """Function to run the model to predict the battery test.
        When the prediction is finished, the time with more parameters are saved in a json file.
        The place for the json file is in the respective folder results.

        Args:
            model (str): Name of the model
            dimension (str): Number of dimensions (2d or 3d)
            run_version (str): Version of the model
        """        
        args = [
            "--mode", "Train",
            "--model", str(model),
            "--max_epochs", "300",
            "--cache_rate", "0.0",
            "--out_channels", "16",
            "--batch_size", str(batch_size),
            "--spatial_dims", "3",
            "--data_dir", "/home/alfredo/data/amos22",
            "--trainmode", "init",
            "--roi_size", "96", "96", "96",
            "--dataset", str(dataset),
            "--dimension", str(dimension)

            ]
        
        cmd = ['python3', 'train.py'] + args
        subprocess.run(cmd)


        # result = {
        #     'model': model,
        #     'dimension': dimension,
        #     'run_version': run_version,
        #     'dataset': SequentialTrain.dataset, 
        #     'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
        # }

        # base_path = f'./logs/{SequentialTrain.dataset}/{model}_{dimension}/lightning_logs/version_{run_version}'
        # file_path = f'{base_path}/time_train_version_{run_version}.json'

        # os.makedirs(base_path, exist_ok=True)
        # with open(file_path, 'w') as json_file:
        #     json.dump(result, json_file, indent=4)
        #     json_file.write(',\n')

        
        # base_path = f'./logs/{SequentialTrain.dataset}/{model}_{dimension}/lightning_logs/version_{run_version}'
        # file_path = f'{base_path}/time_train_version_{run_version}.json'

        # with open(file_path, 'r') as f:
        #     result = json.load(f)

        # end_time = datetime.now()  
        # elapsed_time = end_time - datetime.strptime(result['start_time'], "%Y-%m-%d %H:%M:%S")  

        # result['end_time'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
        # result['elapsed_time'] = str(elapsed_time)

        # with open(file_path, 'w') as f:
        #     json.dump(result, f, indent=4)

        
        

    

if __name__ == "__main__":
    SequentialTrain()()
