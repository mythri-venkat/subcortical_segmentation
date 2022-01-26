import os

import numpy as np
import torch
import nibabel as nib

from unetseg3d.unet3d.utils import get_logger
from unetseg3d.unet3d.metrics import get_evaluation_metric,get_evaluation_metrics
import pandas as pa
import torch.nn.functional as F
from . import utils


logger = get_logger('UNetPredictor')


class _AbstractPredictor:
    def __init__(self, model, output_dir, config, **kwargs):
        self.model = model
        self.output_dir = output_dir
        self.config = config
        self.predictor_config = kwargs

    @staticmethod
    def volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    @staticmethod
    def get_output_dataset_names(number_of_datasets, prefix='predictions'):
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]

    def __call__(self, test_loader):
        raise NotImplementedError

class NiiPredictor(_AbstractPredictor):
    """
    Applies the model on the given dataset and saves the result as Nii File.

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        output_dir (str): path to the output directory (optional)
        config (dict): global config dict
    """

    def __init__(self, model, output_dir, config, **kwargs):
        super().__init__(model, output_dir, config, **kwargs)
        self.device = self.config['device']
        self.eval_criterion = get_evaluation_metrics(self.config)
        self.roi_patches=kwargs['roi_patches'] if 'roi_patches' in kwargs else False


    def __call__(self, test_loader):
        # assert isinstance(test_loader.dataset, AbstractHDF5Dataset)

        logger.info(f"Processing '{test_loader.dataset.file_path}'...")
        # output_file = _get_output_file(dataset=test_loader.dataset, output_dir=self.output_dir)

        out_channels = self.config['model'].get('out_channels')

        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            logger.info(f"Saving only channel '{prediction_channel}' from the network output")

        device = self.device
        output_heads = self.config['model'].get('output_heads', 1)

        logger.info(f'Running prediction on {len(test_loader)} batches...')

        


        # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
        self.model.eval()
        # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
        self.model.testing = True
        # Run predictions on the entire input dataset
        with torch.no_grad():
            eval_scores = []
            for t in test_loader:
                batch,target, subject,atlas=self._split_training_batch(t)  
                # send batch to device
                batch = batch.to(device)
                target = target.to(device)

                #output =  self.model(batch)

                predictions=[]
                bnoutputs=[]
                binterps=[]
                if self.roi_patches:
                    boxes= utils.get_roi(atlas)
                    for i in range(len(boxes)):
                        input_cropped,target_cropped,binterp = utils.get_patches(batch,target,boxes[i])
                        binterps.append(binterp)
                        pred = self.model(input_cropped)
                        if isinstance(pred,tuple):
                            prediction=pred[0]
                        else:
                            prediction = pred
                        predictions.append(prediction)
                        # bnoutputs.append(prediction[:,i,...] > 0.5)
                        # eval_score.append(self.eval_criterion(bnoutputs[i], target_cropped))

                
                    prediction = utils.stitch_patches(predictions,boxes,batch.shape,binterps)
                else:
                    prediction = self.model(batch)
                    prediction = torch.argmax(prediction,1)
                
                if isinstance(self.eval_criterion,list):
                    evals = []
                    for eval_crit in self.eval_criterion:
                        eval_score = eval_crit(prediction,target)
                        evals.append(eval_score.cpu().numpy())
                    eval_scores.append(evals)

                    print('Results: ')
                    print([type(self.eval_criterion[i]).__name__ + ':' + str(np.mean(evals[i][1:])) for i in range(len(self.eval_criterion))])
                    
                else:
                    eval_score = self.eval_criterion(prediction,target)
                    eval_scores.append(eval_score.cpu().numpy())
                    print(np.mean(eval_score.cpu().numpy()[1:]))
                output_file=self._save_results(prediction,subject)
                
                
                # save results
                logger.info(f'Saving predictions to: {output_file}')
            avg12,avg14=self._evaluate_save_results(eval_scores)
            logger.info(f'Results: {avg12},{avg14}')

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) if not type(x) is str else x for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        subjects = None
        atlas = None
        if len(t) == 2:
            input, target = t
        elif len(t) == 3:
            input, target, subjects = t
        else:
            input,target,subjects,atlas = t
            
        return input, target, subjects,atlas

    def _evaluate_save_results(self,eval_scores):
        # if isinstance(eval_scores[0],list)
            
        eval_scores = np.array(eval_scores)
        
        evals = len(eval_scores.shape)
        if evals == 2:
            eval_scores = np.expand_dims(eval_scores,1)
        if os.path.isdir(os.path.dirname(self.config['model_path'])):
            outfile = os.path.dirname(self.config['model_path']) +'/summary.csv'
        else:
            outfile = 'summary.csv'
        dct={}        
        avg =  np.mean(eval_scores,0)[0].tolist()
        avg.extend([np.mean(eval_scores[:,0,3:]),np.mean(eval_scores[:,0,1:])])
        std = np.std(eval_scores,0)[0].tolist()
        std.extend([np.std(eval_scores[:,0,3:]),np.std(eval_scores[:,0,1:])])
        dct['dice_mean'] = avg
        dct['dice_std'] = std
        if evals == 3:
            avg =  np.mean(eval_scores,0)[1].tolist()
            avg.extend([np.mean(eval_scores[:,1,3:]),np.mean(eval_scores[:,1,1:])])
            std = np.std(eval_scores,0)[1].tolist()
            std.extend([np.std(eval_scores[:,1,3:]),np.std(eval_scores[:,1,1:])])
            dct['hd_mean'] = avg
            dct['hd_std'] = std
        df = pa.DataFrame(dct)
        df.to_csv(outfile)
        return dct['dice_mean'][-2],dct['dice_mean'][-1]


    def _save_results(self,prediction,subject):
        prediction = prediction.squeeze(0).cpu().numpy().astype(np.int32)
        outfile = self.output_dir+subject[0]+'.nii.gz'
        img = nib.Nifti1Image(prediction,np.eye(4))
        nib.save(img,outfile)
        return outfile

