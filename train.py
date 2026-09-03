import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import logging
import ast

import scipy.io
import glob
import argparse
from datetime import datetime
from xmlrpc.client import boolean

import numpy as np
import yaml

import nibabel as nib


import torch
torch.autograd.set_detect_anomaly(True, check_nan=True)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete


from monai.config import print_config
from monai.metrics import DiceMetric

from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
    list_data_collate,
    MetaTensor,
)


from training.dataset.utils import get_dataset
from model.utils import get_model
from utils import (
    save_nifti,
    save_configure,
    configure_logger,
    get_latest_run_version_ckpt_epoch_no,
    sample_stack,   
)


class Net(pl.LightningModule):
    """Class that defines the Lightning Module that will be used for training, validation and testing.
    """    

    def __init__(self, args):
        """Constructor of the class. Initialize the Lightning Module that will be used for training, validation and testing.

        Args:
            args (argparse.Namespace): Arguments from the command line.
        """        
        super().__init__()
        self.save_hyperparameters()
        
        self.keys = ["image", "label"]

        try:
            self.args = args
            self._model = get_model(args)

            self.post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
            self.post_label = AsDiscrete(to_onehot=args.out_channels)

            # Loss metrics
            self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
            self.dice_metric = DiceMetric(
                include_background=False, reduction="mean", get_not_nans=False)
            self.dice_metric_test = DiceMetric(
                include_background=False, reduction="mean", get_not_nans=False)
            
            self.roi_size = args.roi_size
            self.inference_batch_size = args.inference_batch_size
            self.batch_size = args.batch_size

            self.best_val_dice = 0
            self.best_val_epoch = 0
            self.best_test_dice = 0
            self.best_test_epoch = 0
            self.max_epochs = args.max_epochs 

            self.metric_test_values = []
            self.metric_values = []
            self.epoch_loss_values = []
            self.validation_step_outputs = []
            self.test_step_outputs = []
            self.training_step_outputs = []

        except Exception as e: 
            print(e)
            pass


    def forward(self, x):
        """Function that performs a forward pass on the network.

        Args:
            x (torch.Tensor | monai.data.meta_tensor.MetaTensor): Input data to the network 

        Returns:
            (torch.Tensor | monai.data.meta_tensor.MetaTensor): Output data from the network
        """ 
        # print(f"Input shape: {x.shape}")
        x = x.to(torch.float32)
        return self._model(x)


    def configure_optimizers(self):
        """Function that configures the optimizer to be used during training.

        Returns:
           torch.optim.adamw.AdamW : Optimizer to be used during training
        """        
        optimizer = torch.optim.AdamW(
            self._model.parameters(), 
            lr=1e-5, 
            weight_decay=1e-4
        )
        return optimizer 
    

    def sanitize_labels(self, labels, num_classes):
        """
        Function to make sure that the labels are in the range (0, num classes)

        Args:
            labels (array, tensor): tensor/array with the labels
            num_classes (int): max number of classes of the dataset

        Returns:
            _type_: _description_
        """
        if hasattr(labels, "as_tensor"):
            labels = labels.as_tensor()
            
        labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)

        labels = labels.to(torch.long)
        
        if len(labels.shape) == 3:
            labels = labels.unsqueeze(1)
            
        labels = torch.clamp(labels, min=0) % num_classes
        
        return labels.contiguous()
    

    def training_step(self, batch, batch_idx):
        """Function that performs a training step on the network. 

        Args:
            batch (dict): The batch of data to be used for training
            batch_idx (int): The index of the batch

        Returns:
            dict: Dictionary containing the loss and the tensorboard logs
        """        
        images, labels = batch[self.keys[0]], batch[self.keys[1]]
        images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        labels = labels.contiguous()

        if torch.isnan(images).any(): print("NaN in input images")
        
        output = self.forward(images)
        
        target_output = output[0] if isinstance(output, (list, tuple)) else output
                
        if torch.isnan(target_output).any():
            return None #
        
        num_canales_reales = target_output.shape[1] if hasattr(target_output, 'shape') else output.shape[1]
        labels = self.sanitize_labels(labels=labels, num_classes=num_canales_reales)
        loss = self.loss_function(target_output, labels)
        
        if torch.isnan(loss):
            print("WARN! the loss is NaN")
            
        self.training_step_outputs.append({"loss": loss.detach()})
        self.log('train_loss', loss.item(), prog_bar=True)
        return loss

    def on_training_epoch_end(self):
        """Function that performs an action at the end of the training epoch.
        """        
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())
        self.training_step_outputs.clear()  # free memory


    def validation_step(self, batch, batch_idx):
        """Function that performs a validation step on the network. 

        Args:
            batch (dict): The batch of data to be used for training
            batch_idx (int): The index of the batch

        Returns:
            dict: Dictionary containing the loss and the tensorboard logs
        """  
        images, labels = batch[self.keys[0]], batch[self.keys[1]]        
        images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        labels = labels.contiguous()

        outputs = sliding_window_inference(
            images, 
            self.roi_size, 
            self.inference_batch_size, 
            self.forward
        )

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
            
        num_chan = outputs.shape[1] if hasattr(outputs, 'shape') else outputs.shape[1]
        labels = self.sanitize_labels(labels=labels, num_classes=num_chan)

        loss = self.loss_function(outputs, labels)

        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        self.dice_metric(y_pred=outputs, y=labels)
        
        self.log("val_loss", loss, batch_size=1) 
        d = {"val_loss": loss.detach(), "val_number": len(outputs)}
        self.validation_step_outputs.append(d)

        return d


    def on_validation_epoch_end(self):
        """Function that performs an action at the end of the validation epoch.

        Returns:
            dict: Dictionary containing the tensorboard logs
        """        
        val_loss, num_items = 0, 0

        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        print(
            f"\nCurrent epoch: {self.current_epoch} "
            f"Current mean dice: {mean_val_dice:.4f}"
            f"\nBest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        
        self.metric_values.append(mean_val_dice)
        self.validation_step_outputs.clear()  # free memory

        return {"log": tensorboard_logs}
    

    def test_step(self, batch, batch_idx):
        """Function that performs the test step on the network. 

        Args:
            batch (dict): The batch of data to be used for training
            batch_idx (int): The index of the batch

        Returns:
            dict: Dictionary containing the loss and the tensorboard logs
        """  
        images, labels = batch["image"], batch["label"]
        images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        labels = labels.contiguous()
        labels = self.sanitize_labels(labels=labels, num_classes=args.out_channels)
        
        outputs = sliding_window_inference(
            images, 
            self.roi_size, 
            self.inference_batch_size, 
            self.forward
        )

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
            
        num_chan = outputs.shape[1] if hasattr(outputs, 'shape') else outputs.shape[1]
        labels = torch.clamp(labels.long(), min=0, max=num_chan - 1)
            
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        self.dice_metric_test(y_pred=outputs, y=labels)

        self.log("test_loss", loss, batch_size=1) 
        d = {"test_loss": loss.detach(), "test_number": len(outputs)}
        self.test_step_outputs.append(d)

        return d


    def on_test_epoch_end(self):
        """Function that performs an action at the end of the test epoch.

        Returns:
            dict: Dictionary containing the tensorboard logs
        """        
        test_loss, num_items = 0, 0

        for output in self.test_step_outputs:
            test_loss += output["test_loss"].sum().item()
            num_items += output["test_number"]

        mean_test_dice = self.dice_metric_test.aggregate().item()
        self.dice_metric_test.reset()
        mean_test_loss = torch.tensor(test_loss / num_items)
        tensorboard_logs = {
            "test_dice": mean_test_dice,
            "test_loss": mean_test_loss,
        }

        if mean_test_dice > self.best_test_dice:
            self.best_test_dice = mean_test_dice
            self.best_test_epoch = self.current_epoch

        print(
            f"\nCurrent epoch: {self.current_epoch} "
            f"Current mean dice test: {mean_test_dice:.4f}"
            f"\nbest mean dice test: {self.best_test_dice:.4f} "
            f"at epoch: {self.best_test_epoch}"
        )

        self.metric_test_values.append(mean_test_dice)
        self.test_step_outputs.clear()

        return {"log": tensorboard_logs}



class MainModule:
    """Class that defines the main module that will be used to train, test and predict with different medical models.
    """    

    def __call__(self, args):
        """Call method that will be used to train, test and predict with different medical models.

        Args:
            args (argparse.Namespace): Arguments from the command line.

        Raises:
            ValueError: Invalid mode. Choose between Train, Test or Predict
        """     

        # num_of_gpus = torch.cuda.device_count()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        #print_config()
        #print("Number of GPUs available: {}. Device used: {}".format(num_of_gpus, device))
        #logging.basicConfig(level=logging.INFO)

        root_dir = '/mnt/disco4t/alfredo'
        log_dir = os.path.join(root_dir, 'logs', args.dataset, args.model, args.dimension)
        #log_dir = os.path.join(root_dir, 'logs_3' ) ## TODO  CHANGE 

        if args.mode == "Train":
            self.train(args, log_dir)

        elif args.mode == "Test":   
            self.test(args, log_dir, root_dir=root_dir)

        elif args.mode == "Predict":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.predict(args, log_dir, device, root_dir=root_dir)

        else:
            try:
                raise ValueError("Invalid mode. Choose between Train, Test or Predict")
            except ValueError as err:
                print(err.args)
        

    def train(self, args, log_dir):
        """Function to train the model. Call the specific dataset and model, and train the model.

        Args:
            args (argparse.Namespace): Arguments from the command line.
            log_dir (str): Log directory
        """    

        logging.info("Training mode")
        print("[INFO] Training mode\n")

        logging.info(
            f"\nDataset: {args.dataset},\n"
            + f"Model: {args.model},\n"
            + f"Dimension: {args.dimension}"
        )

        logging.info("Creating Model")

        if args.trainmode == 'cont':
            log_dir_model = os.path.join(log_dir, 'lightning_logs')
            ckpt_path = get_latest_run_version_ckpt_epoch_no(lightning_logs_dir=log_dir_model)
            net = Net.load_from_checkpoint(checkpoint_path=ckpt_path, args=args)
            print("Continue training from checkpoint: ", ckpt_path)

        else:
            net = Net(args)

        tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir)
        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                filename= args.model + '-{epoch:02d}-{val_loss:.2f}',
                                                save_top_k=1, mode='min')

        early_stopping_callback = EarlyStopping(
            monitor='val_loss', 
            patience=10,         
            mode='min',         
            verbose=True,
            check_finite=True    
        )

        logging.info("Preparing trainer")
        trainer = pl.Trainer(
            accelerator="gpu", 
            precision="32",
            devices=[0],
            max_epochs=net.max_epochs,
            gradient_clip_val=0.1,         
            logger=tb_logger,
            enable_checkpointing=True,
            num_sanity_val_steps=1,
            log_every_n_steps= 1,
            callbacks=[checkpoint_callback, early_stopping_callback]
        )

        logging.info("Preparing Dataset")
        dataset = get_dataset(args)
        logging.info("Created Dataset and Trainer")

        #configure_logger(0, log_dir+f"/fold.txt")

        logging.info("Start training")
        start = datetime.now()
        print('Start training', start)
        trainer.fit(net, dataset)
        print('Training duration:', datetime.now() - start)
        logging.info(f"Best Dice: {net.best_val_dice:.4f} in epoch {net.best_val_epoch}")

        
        logging.info("Start evaluation")
        trainer.test(net, dataloaders=dataset)
        logging.info("Evaluation Done")
        logging.info(f"Dice: {net.best_val_dice:.4f}")


    def test(self, args, log_dir, device, root_dir="."):
        """Function to test the model. Call the specific dataset and model, and test the model.

        Args:
            args (argparse.Namespace): Arguments from the command line.
            log_dir (str): Log directory
            device (torch.device): Device to be used
            root_dir (str, optional): Root path dir. Defaults to ".".
        """        
        logging.info("Test mode")
        print("[INFO] Test mode\n")

        logging.info(
            f"\nDataset: {args.dataset},\n"
            + f"Model: {args.model},\n"
            + f"Dimension: {args.dimension}"
        )

        logging.info("Loading Model")
        net = Net(args)
        log_dir = os.path.join(log_dir, 'lightning_logs')
        ckpt_path = get_latest_run_version_ckpt_epoch_no(
            lightning_logs_dir=log_dir, 
            run_version=args.run_version
            )
        ckpt_model = Net.load_from_checkpoint(checkpoint_path=ckpt_path)

        ## TODO: Implement our metrics
            

    def predict(self, args, log_dir, device, root_dir="."):
        """Function to predict with the model. Call the specific dataset and model, and predict with the model.

        Args:
            args (argparse.Namespace): Arguments from the command line.
            log_dir (str): Log directory
            device (torch.device): Device to be used
            root_dir (str, optional): Root path dir. Defaults to ".".
        """        
        logging.info("Predict mode")
        print("[INFO] Predict mode\n")

        logging.info(
            f"\nDataset: {args.dataset},\n"
            + f"Model: {args.model},\n"
            + f"Dimension: {args.dimension}"
        )

        logging.info("Loading Model")
        net = Net(args)
        log_dir_ckpt = os.path.join(log_dir, 'lightning_logs')
        ckpt_path = get_latest_run_version_ckpt_epoch_no(
            lightning_logs_dir=log_dir_ckpt, 
            )
        ckpt_model = Net.load_from_checkpoint(checkpoint_path=ckpt_path)

        dataset = get_dataset(args)
        pred_transforms, post_transform = dataset.get_preprocessing_transform_pred()

        # Load the data
        pred_files = dataset.load_images_prediction()

        pred_ds = CacheDataset(
            data=pred_files, 
            transform=pred_transforms,
            cache_rate=args.cache_rate, 
            num_workers=args.num_workers,
        )
        
        pred_loader = torch.utils.data.DataLoader(
            pred_ds, 
            batch_size=args.inference_batch_size, 
            num_workers=args.num_workers,
            collate_fn=list_data_collate,
        )

        ckpt_model.freeze()
        ckpt_model.eval()
        ckpt_model.to(device)

        with torch.no_grad():
            for i, pred_data in enumerate(pred_loader):

                pred_outputs = sliding_window_inference(
                    pred_data[net.keys[0]].to(device),  
                    args.roi_size, 
                    args.inference_batch_size, 
                    ckpt_model, 
                    overlap=0.8
                )

                if isinstance(pred_outputs, tuple):
                    pred_outputs = pred_outputs[0]

                best_pred = torch.argmax(pred_outputs, dim=1, keepdim=True).detach().cpu()[0]
                
                best_pred = best_pred.to(torch.float32)

                input_tensor = pred_data[net.keys[0]][0]

                safe_operations = []
                for op in input_tensor.applied_operations:
                    op_copy = op.copy()
                    class_name = op_copy.get('class', '')
                    
                    if any(x in class_name for x in ['Scale', 'Normalize', 'Shift', 'Intensity']):
                        continue
                        
                    if 'extra_info' in op_copy and 'mode' in op_copy['extra_info']:
                        op_copy['extra_info']['mode'] = 'nearest'
                        
                    safe_operations.append(op_copy)

                best_pred_meta = MetaTensor(
                    best_pred, 
                    meta=input_tensor.meta, 
                    applied_operations=safe_operations
                )

                best_pred_reshape = post_transform.inverse(
                    {net.keys[0]: best_pred_meta} 
                )
                
                name = os.path.split(
                    pred_files[i][net.keys[0]]
                )[-1].split('.')[0]
                
                out_path = os.path.join(args.path_prediction, f"{args.model}_{args.dimension}")
                os.makedirs(out_path, exist_ok=True)
                
                final_pred_tensor = best_pred_reshape[net.keys[0]].squeeze()
                final_pred_tensor = torch.round(final_pred_tensor).to(torch.uint8)
                
                if str(args.dimension).lower() == '2d':
                    save_path_npy = os.path.join(out_path, f"{name}_Pred.npy")
                    np.save(save_path_npy, final_pred_tensor.numpy())
                else:
                    save_nifti(
                        final_pred_tensor,
                        name=name+"_Pred",
                        path_out_images=out_path + "/"
                    )

                if args.show:
                    sample_stack(
                        pred_data[net.keys[0]][0, 0, ...], 
                        title=name, 
                        show=args.show,
                        path_out_images=out_path + "/"
                    )
                    sample_stack(
                        final_pred_tensor,
                        title=name + "_Pred", 
                        map="plasma", 
                        show=args.show,
                        path_out_images=out_path + "/"
                    )
                
                #TODO CHANGE
                #if i == 1:
                #    break



def get_parser():
    """Function to get the parser with the arguments.

    Raises:
        ValueError: The specified configuration doesn't exist

    Returns:
        argparse.Namespace: Arguments from the command line.
    """    
    parser = argparse.ArgumentParser(description="Framework to train, test and predict with different medical models")

    parser.add_argument("--max_epochs", default=800, type=int, help="Max number of epochs for training")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for training")
    parser.add_argument("--cache_rate", default=1.0, type=float, help="Cache rate for training")
    parser.add_argument("--pin_memory", default=False, type=bool, help="Pin memory for training")

    parser.add_argument("--percentage_train", default=0.8, type=float, help="Percentage of training data")
    
    parser.add_argument("--spatial_dims", default=3, type=int, help="Numero de dimension espaciais (2D ou 3D)")
    parser.add_argument("--in_channels", default=1, type=int, help="Input image channels (i.e. 3 for color images, 1 for gray images)")
    parser.add_argument("--out_channels", default=14, type=int, help="Number of classes")
    parser.add_argument("--data_dir", default='../Datasets/BTCV_/', type=str, help="Training data directory")
    parser.add_argument("--mode", default='Predict', type=str, help="Work mode (Train, Test, Predict)")
    parser.add_argument("--trainmode", default='init', type=str, help="Continue training from checkpoint (cont) or start from scratch (init)")
    parser.add_argument("--roi_size", nargs='+', default=[96, 96, 96], type=int, help="Slide window size")    
    parser.add_argument("--inference_batch_size", default=1, type=int, help="Batch size for inference")
    parser.add_argument('--folders_img_lbl', type=bool, default=True, help="If images and labels are in different folders")
    
    parser.add_argument("--show", default=False, type=boolean, help="Visualizar resultados on-line")

    parser.add_argument('--model', type=str, default='segformer', help="Network model name. Available models: unet, unetr, swin_unetr, unet++, attention_unet, resunet, medformer, vnet, segformer")
    parser.add_argument('--dimension', type=str, default='3d', help="Dimension of the model (2d or 3d)")
    parser.add_argument('--dataset', type=str, default='btcv', help="Name of the dataset")
    parser.add_argument('--run_version', type=int, default=24, help="Version of the checkpoint for testing or predicting")
    parser.add_argument('--path_prediction', type=str, default="./results/", help="Path to save the predictions")


    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()
    args.roi_size = tuple(args.roi_size)    
    config_path = 'config/%s/%s_%s.yaml'%(args.dataset, args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    print('Loading configurations from %s'%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)
    

    return args




if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    args = get_parser()

    MainModule()(args)
