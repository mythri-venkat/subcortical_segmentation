import os

from matplotlib.pyplot import box

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from unetseg3d.datasets.utils import get_train_loaders
from unetseg3d.unet3d.losses import get_loss_criterion
from unetseg3d.unet3d.metrics import get_evaluation_metric
from unetseg3d.unet3d.model import get_model
from unetseg3d.unet3d.utils import EarlyStopping, get_logger, get_tensorboard_formatter, create_sample_plotter, create_optimizer, \
    create_lr_scheduler, get_number_of_learnable_parameters
from . import utils
import torch.nn.functional as F
import random
import math
import optuna
from optuna.storages import RetryFailedTrialCallback

logger = get_logger('UNet3DTrainer')

def _create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)
    
    # get tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # get sample plotter
    sample_plotter = create_sample_plotter(trainer_config.pop('sample_plotter', None))

    if resume is not None:
        # continue training from a given checkpoint
        return UNet3DTrainer.from_checkpoint(model=model,
                                             optimizer=optimizer,
                                             lr_scheduler=lr_scheduler,
                                             loss_criterion=loss_criterion,
                                             eval_criterion=eval_criterion,
                                             loaders=loaders,
                                             tensorboard_formatter=tensorboard_formatter,
                                             sample_plotter=sample_plotter,
                                             **trainer_config)
    elif pre_trained is not None:
        # fine-tune a given pre-trained model
        return UNet3DTrainer.from_pretrained(model=model,
                                             optimizer=optimizer,
                                             lr_scheduler=lr_scheduler,
                                             loss_criterion=loss_criterion,
                                             eval_criterion=eval_criterion,
                                             tensorboard_formatter=tensorboard_formatter,
                                             sample_plotter=sample_plotter,
                                             device=config['device'],
                                             loaders=loaders,
                                             **trainer_config)
    else:
        # start training from scratch
        return UNet3DTrainer(model=model,
                             optimizer=optimizer,
                             lr_scheduler=lr_scheduler,
                             loss_criterion=loss_criterion,
                             eval_criterion=eval_criterion,
                             device=config['device'],
                             loaders=loaders,
                             tensorboard_formatter=tensorboard_formatter,
                             sample_plotter=sample_plotter,
                             **trainer_config)


class UNet3DTrainerBuilder:
    @staticmethod
    def build(config):
        # Create the model
        model = get_model(config['model'])
        # use DataParallel if more than 1 GPU available
        device = config['device']
        if torch.cuda.device_count() > 1 and not device.type == 'cpu':
            model = nn.DataParallel(model)
            logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

        # put the model on GPUs
        logger.info(f"Sending the model to '{config['device']}'")
        model = model.to(device)

        # Log the number of learnable parameters
        logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

        # Create loss criterion
        loss_criterion = get_loss_criterion(config)
        # Create evaluation metric
        eval_criterion = get_evaluation_metric(config)

        # Create data loaders
        loaders = get_train_loaders(config)

        # Create the optimizer
        optimizer = create_optimizer(config['optimizer'], model)

        # Create learning rate adjustment strategy
        lr_scheduler = create_lr_scheduler(config.get('lr_scheduler', None), optimizer)

        # Create model trainer
        trainer = _create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                  loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders)

        return trainer

class UNet3DTrainerBuilderOpt:
    
    def __init__(self,config):
        self.config = config

    def build(self,trial):
        config = self.config.copy()
        config['model']['num_levels'] = trial.suggest_int("num_levels",3,4)
        config['model']['f_maps'] = trial.suggest_categorical("f_maps",[16,20,24,28,32])
        config['model']['layer_order'] = trial.suggest_categorical("layer_order",["icr","bcr"])
        config['optimizer']['name'] = "Adam" #trial.suggest_categorical("optimizer_name",["Adam","SGD"])
        config['optimizer']['learning_rate'] = trial.suggest_float("learning_rate",1e-5,1e-2,log=True)
        config['optimizer']['weight_decay'] = trial.suggest_float("weight_decay",1e-4,1e-2,log=True)
        config['loss']['name'] = trial.suggest_categorical("loss",["CrossEntropyLoss","CEDiceLoss"])
        config['lr_scheduler']['milestones'] = [trial.suggest_int("milestone_1",30,40),trial.suggest_int("milestone_2",100,120)]
        config['loaders']['batch_size'] = trial.suggest_int("batch_size",1,3)


        trial_number = RetryFailedTrialCallback.retried_trial_number(trial)
        config['trainer']['checkpoint_dir'] = config['trainer']['checkpoint_dir_name']+'/'+str(trial.number)
        if trial_number is not None and os.path.isfile(config['trainer']['checkpoint_dir']+'/'+str(trial_number)+'/last_checkpoint.pytorch'):
            config['resume']=config['trainer']['checkpoint_dir']+'/'+str(trial_number)+'/last_checkpoint.pytorch'

        logger.info(config)
        # Create the model
        model = get_model(config['model'])
        # use DataParallel if more than 1 GPU available
        device = config['device']
        if torch.cuda.device_count() > 1 and not device.type == 'cpu':
            model = nn.DataParallel(model)
            logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

        # put the model on GPUs
        logger.info(f"Sending the model to '{config['device']}'")
        model = model.to(device)

        # Log the number of learnable parameters
        logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

        # Create loss criterion
        loss_criterion = get_loss_criterion(config)
        # Create evaluation metric
        eval_criterion = get_evaluation_metric(config)

        # Create data loaders
        loaders = get_train_loaders(config)

        # Create the optimizer
        optimizer = create_optimizer(config['optimizer'], model)

        # Create learning rate adjustment strategy
        lr_scheduler = create_lr_scheduler(config.get('lr_scheduler', None), optimizer)

        # Create model trainer
        trainer = _create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                  loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders)

        metric = trainer.fit(trial)

        return metric

class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        sample_plotter (callable): saves sample inputs, network outputs and targets to a given directory
            during validation phase
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs=100, max_num_iterations=int(1e5),
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 tensorboard_formatter=None, sample_plotter=None,
                 skip_train_validation=False,roi_patches=False,interp=False, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.roi_patches = roi_patches
        self.interp = interp

        # logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter
        self.sample_plotter = sample_plotter

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        self.skip_train_validation = skip_train_validation

    @classmethod
    def from_checkpoint(cls, resume, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders,
                        tensorboard_formatter=None, sample_plotter=None, **kwargs):
        logger.info(f"Loading checkpoint '{resume}'...")
        state = utils.load_checkpoint(resume, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(resume)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   torch.device(state['device']),
                   loaders, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=kwargs['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   skip_train_validation=state.get('skip_train_validation', False),
                   tensorboard_formatter=tensorboard_formatter,
                   roi_patches = kwargs['roi_patches'],
                   interp = kwargs['interp'],
                   sample_plotter=sample_plotter)

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, loaders,
                        max_num_epochs=100, max_num_iterations=int(1e5),
                        validate_after_iters=100, log_after_iters=100,
                        validate_iters=None, num_iterations=1, num_epoch=0,
                        eval_score_higher_is_better=True, best_eval_score=None,
                        tensorboard_formatter=None, sample_plotter=None,
                        skip_train_validation=False,roi_patches=False,interp=False,**kwargs):
        logger.info(f"Logging pre-trained model from '{pre_trained}'...")
        utils.load_pretrained_checkpoint(pre_trained, model, None)
        if 'checkpoint_dir' not in kwargs:
            checkpoint_dir = os.path.split(pre_trained)[0]
        else:
            checkpoint_dir = kwargs.pop('checkpoint_dir')
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, loaders, checkpoint_dir,
                   eval_score_higher_is_better=eval_score_higher_is_better,
                   best_eval_score=best_eval_score,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   max_num_iterations=max_num_iterations,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   validate_iters=validate_iters,
                   tensorboard_formatter=tensorboard_formatter,
                   sample_plotter=sample_plotter,
                   skip_train_validation=skip_train_validation,
                   roi_patches=roi_patches,
                   interp=interp)

    def fit(self,trial=None):
        eval_score=None
        for epoch in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate,eval_score = self.train(eval_score)
            if trial is not None and eval_score is not None:
                trial.report(eval_score, epoch)
            if trial is not None and trial.should_prune():
                raise optuna.exceptions.TrialPruned()             

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return eval_score

            self.num_epoch += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")
        return eval_score

    def train(self,eval_score_val=None):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()
        
        for t in self.loaders['train']:
            logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                        f'Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            input, target,atlas = self._split_training_batch(t)
            #output = self.model(input)
            weight = None
            outputs=[]
            binterps = []
            if self.roi_patches:
                boxes = utils.get_roi(atlas,True)
                idxshuffle = list(range(15))
                random.shuffle(idxshuffle)
                boxes=[boxes[i] for i in idxshuffle]
                
                for i in range(len(boxes)):
                    # i=np.random.randint(0,15)
                    input_cropped,target_cropped,binterp = utils.get_patches(input,target,boxes[i],interp=self.interp)
                    binterps.append(binterp)
                    output = self.model(input_cropped)
                    loss = self.loss_criterion(output,target_cropped)
                    outputs.append(output)
                    train_losses.update(loss.item(), self._batch_size(input_cropped))

                    # compute gradients and update parameters
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                output = self.model(input)
                loss = self.loss_criterion(output,target)
                train_losses.update(loss.item(), self._batch_size(input))

                # compute gradients and update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()





            if self.num_iterations % self.log_after_iters == 0:
                # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # the evaluation metric as well as images in tensorboard will be incorrectly computed
                if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                    if self.roi_patches:
                        for i,output in enumerate(outputs):
                            outputs[i]=self.model.final_activation(output)
                       
                        output = utils.stitch_patches(outputs,boxes,input.shape,binterps)
                    else:
                        output = torch.argmax(self.model.final_activation(output),1)
                        
                # compute eval criterion
                if not self.skip_train_validation:
                    eval_score = torch.mean(self.eval_criterion(output,target)[1:])
                    train_eval_scores.update(eval_score.item(), self._batch_size(input))

                # log stats, params and images
                logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                self._log_params()
                # self._log_images(input, target, output, 'train_')

            if self.should_stop():
                return True,eval_score_val

            self.num_iterations += 1
        
            # if self.num_iterations % self.validate_after_iters == 0:
            # set the model in eval mode
        self.model.eval()
        # evaluate on validation set
        eval_score_val = self.validate()
        # set the model back to training mode
        self.model.train()

        # adjust learning rate if necessary
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(eval_score_val)
        else:
            self.scheduler.step()
        # log current learning rate in tensorboard
        self._log_lr()
        # remember best validation metric
        is_best = self._is_best_eval_score(eval_score_val)

        # save checkpoint
        self._save_checkpoint(is_best)
        return False,eval_score_val

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        if self.sample_plotter is not None:
            self.sample_plotter.update_current_dir()

        with torch.no_grad():
            for i, t in enumerate(self.loaders['val']):
                logger.info(f'Validation iteration {i}')

                input, target,atlas = self._split_training_batch(t)
                weight =None
                #output = self.model(input)
                if self.roi_patches:
                    outputs=[]
                    binterps = []
                    boxes = utils.get_roi(atlas)                    
                    for i in range(len(boxes)):
                        input_cropped,target_cropped,binterp = utils.get_patches(input,target,boxes[i],interp=self.interp)
                        binterps.append(binterp)
                        output = self.model(input_cropped)
                        loss = self.loss_criterion(output, target_cropped)
                        outputs.append(output)
                        val_losses.update(loss.item(), self._batch_size(input_cropped))
                else:
                    output = self.model(input)
                    if weight is None:
                        loss = self.loss_criterion(output, target)
                    else:
                        loss = self.loss_criterion(output, target, weight)

                    val_losses.update(loss.item(), self._batch_size(input))


                # if model contains final_activation layer for normalizing logits apply it, otherwise
                # the evaluation metric will be incorrectly computed
                if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                    if self.roi_patches:
                        for i,output in enumerate(outputs):
                            outputs[i] = self.model.final_activation(output)
                        output = utils.stitch_patches(outputs,boxes,input.shape,binterps)
                    else:
                        output = torch.argmax(self.model.final_activation(output),1)

                # if i % 100 == 0:
                #     self._log_images(input, target, output, 'val_')

                eval_score = torch.mean(self.eval_criterion(output, target)[1:])
                val_scores.update(eval_score.item(), self._batch_size(input))

                if self.sample_plotter is not None:
                    self.sample_plotter(i, input, output, target, 'val')

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            self._log_stats('val', val_losses.avg, val_scores.avg)
            logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            return val_scores.avg

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        atlas = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, atlas = t
        return input, target, atlas

    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output = self.model(input)
        
        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
            

        else:
            state_dict = self.model.state_dict()
            


        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters,
            'skip_train_validation': self.skip_train_validation
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, prefix=''):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations, dataformats='CHW')

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)




def _create_trainer_mtl(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)
    
    # get tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # get sample plotter
    sample_plotter = create_sample_plotter(trainer_config.pop('sample_plotter', None))

    if resume is not None:
        # continue training from a given checkpoint
        return MTLUNet3DTrainer.from_checkpoint(model=model,
                                             optimizer=optimizer,
                                             lr_scheduler=lr_scheduler,
                                             loss_criterion=loss_criterion,
                                             eval_criterion=eval_criterion,
                                             loaders=loaders,
                                             tensorboard_formatter=tensorboard_formatter,
                                             sample_plotter=sample_plotter,
                                             **trainer_config)
    elif pre_trained is not None:
        # fine-tune a given pre-trained model
        return MTLUNet3DTrainer.from_pretrained(model=model,
                                             optimizer=optimizer,
                                             lr_scheduler=lr_scheduler,
                                             loss_criterion=loss_criterion,
                                             eval_criterion=eval_criterion,
                                             tensorboard_formatter=tensorboard_formatter,
                                             sample_plotter=sample_plotter,
                                             device=config['device'],
                                             loaders=loaders,
                                             **trainer_config)
    else:
        # start training from scratch
        return MTLUNet3DTrainer(model=model,
                             optimizer=optimizer,
                             lr_scheduler=lr_scheduler,
                             loss_criterion=loss_criterion,
                             eval_criterion=eval_criterion,
                             device=config['device'],
                             loaders=loaders,
                             tensorboard_formatter=tensorboard_formatter,
                             sample_plotter=sample_plotter,
                             **trainer_config)


class MTLUNet3DTrainerBuilder:
    @staticmethod
    def build(config):
        # Create the model
        model = get_model(config['model'])
        # use DataParallel if more than 1 GPU available
        device = config['device']
        if torch.cuda.device_count() > 1 and not device.type == 'cpu':
            model = nn.DataParallel(model)
            logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

        # put the model on GPUs
        logger.info(f"Sending the model to '{config['device']}'")
        model = model.to(device)

        # Log the number of learnable parameters
        logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

        # Create loss criterion
        loss_criterion = get_loss_criterion(config)
        # Create evaluation metric
        eval_criterion = get_evaluation_metric(config)

        # Create data loaders
        loaders = get_train_loaders(config)

        # Create the optimizer
        optimizer = create_optimizer(config['optimizer'], model)

        # Create learning rate adjustment strategy
        lr_scheduler = create_lr_scheduler(config.get('lr_scheduler', None), optimizer)

        # Create model trainer
        trainer = _create_trainer_mtl(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                  loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders)

        return trainer


class MTLUNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        sample_plotter (callable): saves sample inputs, network outputs and targets to a given directory
            during validation phase
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs=100, max_num_iterations=int(1e5),
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 tensorboard_formatter=None, sample_plotter=None,
                 skip_train_validation=False, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.roi_patches = kwargs['roi_patches'] 

        # logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter
        self.sample_plotter = sample_plotter

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        self.skip_train_validation = skip_train_validation

    @classmethod
    def from_checkpoint(cls, resume, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders,
                        tensorboard_formatter=None, sample_plotter=None, **kwargs):
        logger.info(f"Loading checkpoint '{resume}'...")
        state = utils.load_checkpoint(resume, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(resume)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   torch.device(state['device']),
                   loaders, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=kwargs['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   skip_train_validation=state.get('skip_train_validation', False),
                   tensorboard_formatter=tensorboard_formatter,
                   roi_patches = kwargs['roi_patches'],
                   sample_plotter=sample_plotter)

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, loaders,
                        max_num_epochs=100, max_num_iterations=int(1e5),
                        validate_after_iters=100, log_after_iters=100,
                        validate_iters=None, num_iterations=1, num_epoch=0,
                        eval_score_higher_is_better=True, best_eval_score=None,
                        tensorboard_formatter=None, sample_plotter=None,
                        skip_train_validation=False,**kwargs):
        logger.info(f"Logging pre-trained model from '{pre_trained}'...")
        utils.load_pretrained_checkpoint(pre_trained, model, None)
        if 'checkpoint_dir' not in kwargs:
            checkpoint_dir = os.path.split(pre_trained)[0]
        else:
            checkpoint_dir = kwargs.pop('checkpoint_dir')
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, loaders, checkpoint_dir,
                   eval_score_higher_is_better=eval_score_higher_is_better,
                   best_eval_score=best_eval_score,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   max_num_iterations=max_num_iterations,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   validate_iters=validate_iters,
                   tensorboard_formatter=tensorboard_formatter,
                   sample_plotter=sample_plotter,
                   skip_train_validation=skip_train_validation,
                   roi_patches=kwargs['roi_patches'])

    def fit(self):
        self.es = EarlyStopping(mode='min',patience=15)
        self.es_aux = EarlyStopping(mode='min',patience=15)
        self.baux = False
        self.bmain = False
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epoch += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()
        train_losses_main = utils.RunningAverage()
        train_losses_aux = utils.RunningAverage()
        

        # sets the model in training mode
        self.model.train()
        
        for t in self.loaders['train']:
            logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                        f'Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            input, target,atlas,tissue = self._split_training_batch(t)
            #output = self.model(input)
            weight = None
            outputs=[]
            binterps = []
            
            boxes = utils.get_roi(atlas)
            # idxshuffle = list(range(15))

            # random.shuffle(idxshuffle)
            # boxes=[boxes[i] for i in idxshuffle]
            
            for i in range(len(boxes)):
                # i=np.random.randint(0,15)
                input_cropped,target_cropped,tissue_cropped = utils.get_patches(input,target,boxes[i],tissue)
                
                output,output_aux = self.model(input_cropped)
                loss,loss_main,loss_aux = self.loss_criterion(output,target_cropped,output_aux,tissue_cropped,not self.baux)
                outputs.append(output)
                train_losses.update(loss.item(), self._batch_size(input_cropped))
                train_losses_main.update(loss_main.item(), self._batch_size(input_cropped))
                train_losses_aux.update(loss_aux.item(), self._batch_size(input_cropped))

                # compute gradients and update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



            if self.num_iterations % self.validate_after_iters == 0:
                # set the model in eval mode
                self.model.eval()
                # evaluate on validation set
                eval_score,eval_loss,eval_loss_aux = self.validate()
                # set the model back to training mode
                self.model.train()
                self.bmain = self.es.step(eval_loss)
                if not self.baux:
                    self.baux = self.es_aux.step(eval_loss_aux)
                    if self.baux:
                        logger.info('Early Stopping for auxiliary task.')
                # adjust learning rate if necessary
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_score)
                else:
                    self.scheduler.step()
                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # the evaluation metric as well as images in tensorboard will be incorrectly computed
                if (hasattr(self.model, 'final_activation') or hasattr(self.model.module, 'final_activation')):# and (self.model.final_activation is not None or self.model.module.final_activation is not None):
                    final_activation = nn.Softmax(dim=1) #self.model.final_activation if hasattr(self.model,'final_activation') else self.model.module.final_activation
                    if self.roi_patches:
                        bnoutputs=[]
                        for i,output in enumerate(outputs):
                            outputs[i]=final_activation(output)
                       
                        output = utils.stitch_patches(outputs,boxes,input.shape,binterps)
                    else:
                        output = torch.argmax(final_activation(output),1)
                        
                # compute eval criterion
                if not self.skip_train_validation:
                    # eval_score = torch.mean(torch.Tensor([self.eval_criterion(op, target[:,int(boxes[i][0]):int(boxes[i][1]),int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][4]):int(boxes[i][5])]) for i,op in enumerate(bnoutputs)]))
                    eval_score = torch.mean(self.eval_criterion(output,target)[1:])
                    train_eval_scores.update(eval_score.item(), self._batch_size(input))

                # log stats, params and images
                logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_stats('train', train_losses.avg, train_eval_scores.avg,train_losses_main.avg,train_losses_aux.avg)
                self._log_params()
                # self._log_images(input, target, output, 'train_')

            if self.should_stop() and self.bmain:
                return True

            self.num_iterations += 1

        return False

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_losses_main = utils.RunningAverage()
        val_losses_aux = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        if self.sample_plotter is not None:
            self.sample_plotter.update_current_dir()

        with torch.no_grad():
            for i, t in enumerate(self.loaders['val']):
                logger.info(f'Validation iteration {i}')

                input, target,atlas,tissue = self._split_training_batch(t)
                weight =None
                #output = self.model(input)
                
                outputs=[]
                binterps = []
                boxes = utils.get_roi(atlas)                    
                for i in range(len(boxes)):
                    input_cropped,target_cropped,tissue_cropped = utils.get_patches(input,target,boxes[i],tissue)
                
                    output,output_aux = self.model(input_cropped)
                    loss,loss_main,loss_aux = self.loss_criterion(output,target_cropped,output_aux,tissue_cropped,not self.baux)
                    outputs.append(output)
                    val_losses.update(loss.item(), self._batch_size(input_cropped))
                    val_losses_main.update(loss_main.item(), self._batch_size(input_cropped))
                    val_losses_aux.update(loss_aux.item(), self._batch_size(input_cropped))

                # if model contains final_activation layer for normalizing logits apply it, otherwise
                # the evaluation metric will be incorrectly computed

                if (hasattr(self.model, 'final_activation') or hasattr(self.model.module, 'final_activation')):# and (self.model.final_activation is not None or self.model.module.final_activation is not None):
                    final_activation = nn.Softmax(dim=1) #self.model.final_activation if hasattr(self.model,'final_activation') else self.model.module.final_activation
                    if self.roi_patches:
                        bnoutputs=[]
                        for i,output in enumerate(outputs):
                            outputs[i]=final_activation(output)
                       
                        output = utils.stitch_patches(outputs,boxes,input.shape,binterps)
                    else:
                        output = torch.argmax(final_activation(output),1)

                # if i % 100 == 0:
                #     self._log_images(input, target, output, 'val_')

                eval_score = torch.mean(self.eval_criterion(output, target)[1:])
                val_scores.update(eval_score.item(), self._batch_size(input))

                if self.sample_plotter is not None:
                    self.sample_plotter(i, input, output, target, 'val')

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            self._log_stats('val', val_losses.avg, val_scores.avg,val_losses_main.avg,val_losses_aux.avg)
            logger.info(f'Validation finished. Loss: {val_losses.avg}.Main Loss: {val_losses_main.avg}.Aux Loss: {val_losses_aux.avg}. Evaluation score: {val_scores.avg}')
            return val_scores.avg,val_losses.avg,val_losses_aux.avg

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        atlas = None
        tissue=None
        if len(t) == 2:
            input, target = t
        elif len(t) == 3:
            input, target, atlas = t
        else:
            input,target,atlas,tissue=t
        return input, target, atlas,tissue

    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output = self.model(input)
        
        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
            

        else:
            state_dict = self.model.state_dict()
            


        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters,
            'skip_train_validation': self.skip_train_validation
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg,loss_main_avg,loss_aux_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_loss_main_avg': loss_main_avg,
            f'{phase}_loss_aux_avg': loss_aux_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, prefix=''):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations, dataformats='CHW')

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
