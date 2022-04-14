import importlib
import logging
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import math
import torch.nn.functional as F
import itertools

plt.ioff()
plt.switch_backend('agg')


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state

def load_pretrained_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict',logger=None):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        logger.info(f"Checkpoint '{checkpoint_path}' does not exist")
        return None

    state = torch.load(checkpoint_path, map_location='cpu')
    del state[model_key]['final_conv.weight']
    del state[model_key]['final_conv.bias']
    model.load_state_dict(state[model_key],strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state

def get_patches(input,target,box,tissue=None,interp=False):

    input_cropped = input[:,:,int(box[0]):int(box[1]),int(box[2]):int(box[3]),int(box[4]):int(box[5])]
    target_cropped = target[:,int(box[0]):int(box[1]),int(box[2]):int(box[3]),int(box[4]):int(box[5])]
    if tissue is not None:
        tissue_cropped = tissue[:,int(box[0]):int(box[1]),int(box[2]):int(box[3]),int(box[4]):int(box[5])]
    binterp = False
    patch_size=(48,48,48)
    if (box[1]-box[0]) > 48:
        patch_size = (80,80,80)
    if interp:
        binterp = True
        input_cropped = F.interpolate(input_cropped,size=patch_size,mode='trilinear')
        target_cropped = target_cropped.float().unsqueeze(1)/15
        target_cropped = F.interpolate(target_cropped,size=patch_size,mode='nearest').squeeze(1)
        target_cropped = (target_cropped*15).long()
    if tissue is not None:
        return input_cropped,target_cropped,tissue_cropped
    else:
        return input_cropped,target_cropped,binterp


def stitch_patches(outputs,boxes,shape,binterps,ncls=15):
    
    b,c,w,h,d = shape
    output = torch.zeros(b,ncls,w,h,d)
    output = output.to(outputs[0].device)
    counter = torch.zeros(b,ncls,w,h,d)

    counter = counter.to(outputs[0].device)
    for i,box in enumerate(boxes):
        fs = outputs[i]
        # if (box[1]-box[0]) < 48:
        if binterps[i]:
            fs = F.interpolate(fs,size=(int(box[1]-box[0]),int(box[3]-box[2]),int(box[5]-box[4])),mode='trilinear')
        output[:,:,int(box[0]):int(box[1]),int(box[2]):int(box[3]),int(box[4]):int(box[5])]+=fs
        counter[:,:,int(box[0]):int(box[1]),int(box[2]):int(box[3]),int(box[4]):int(box[5])]+=1

        # # fs = fs.to(self.device)
        # c = output[:,:,int(box[0]):int(box[1]),int(box[2]):int(box[3]),int(box[4]):int(box[5])]<fs
        # d = torch.where(c)
        # output[:,:,box[0]+d[2],box[2]+d[3],box[4]+d[4]]=fs[c]
    output =output/counter
    output = torch.argmax(output,1)
    return output

def get_roi(atlas=None,training=False,pwr=8):

    boxes = get_cropped_structure(atlas,training=training,pwr=pwr)
    return boxes




def bbox2_3D(img,icls=0):

    r = torch.any(torch.any(img == icls, dim=2),dim=2)
    c = torch.any(torch.any(img == icls, dim=1),dim=2)
    z = torch.any(torch.any(img == icls, dim=1),dim=1)
    # print(len(torch.where(r))[0])
    rmin, rmax = torch.where(r)[1][[0, -1]]
    cmin, cmax = torch.where(c)[1][[0, -1]]
    zmin, zmax = torch.where(z)[1][[0, -1]]
    

    return [rmin,rmax,cmin,cmax,zmin,zmax]

def getpwr(n,lbl_shape=(80,80,80),pwr=8):
    if n%pwr == 0:
        return n
    
    for i in range(2,int(lbl_shape[0]/8)):
        npwr = pwr*i
        if n <= pwr:
            return pwr
        elif n <= npwr:
            return npwr
        else:
            continue

    

def get_cropped_structure(lbl,ncls=15,training=False,patch_shape=(48,48,48),pwr=8):
    boxes=[]
    lbl_shape = lbl.shape[1:]

    for icls in range(ncls):
        box = bbox2_3D(lbl,icls)
        #print("c",(box[1]-box[0],box[3]-box[2],box[5]-box[4]))
        if icls == 0:
            box[0],box[1],box[2],box[3],box[4],box[5] = 0,lbl_shape[0],0,lbl_shape[1],0,lbl_shape[2]
        center = [int((box[1] + box[0]) / 2), int((box[3] + box[2]) / 2), int((box[5] + box[4]) / 2)]
        # if training:
            
        #     center[0]+= np.random.randint(-2,3)
        #     center[1]+= np.random.randint(-2,3)
        #     center[2]+= np.random.randint(-2,3)
        b1=getpwr(box[1]-box[0],pwr)
        b2=getpwr(box[3]-box[2],pwr)
        b3=getpwr(box[5]-box[4],pwr)
        # print(b1,b2,b3)
        if b1 == lbl_shape[0] and b2 == lbl_shape[1] and b3 ==lbl_shape[2]:
            center = [int(lbl_shape[0] / 2), int(lbl_shape[1] / 2), int(lbl_shape[2] / 2)]

        
        box[0]=center[0]-int(math.floor(b1/2))
        box[1]=center[0]+int(math.floor(b1/2))

        if box[0]<0:
            box[1]+= (-box[0])
            box[0]=0
            
        if box[1]>lbl_shape[0]:
            box[0]-= box[1]-lbl_shape[0]
            box[1]=lbl_shape[0]
            
        box[2]=center[1]-int(math.floor(b2/2))
        box[3]=center[1]+int(math.floor(b2/2))
        if box[2]<0:
            box[3]+=(-box[2])
            box[2]=0
            
        if box[3]>lbl_shape[0]:
            box[2]-= (box[3]-lbl_shape[0])
            box[3]=lbl_shape[0]
            
        box[4]=center[2]-int(math.floor(b3/2))
        box[5]=center[2]+int(math.floor(b3/2))
        if box[4]<0:
            box[5]+= (-box[4])
            box[4]=0
            
        if box[5]>lbl_shape[0]:
            box[4]-= (box[5]-lbl_shape[0])
            box[5]=lbl_shape[0]
            
        # print("c",(box[1]-box[0],box[3]-box[2],box[5]-box[4]))
        
        boxes.append(box)
    # boxes.sort()
    # boxes = list(k for k,_ in itertools.groupby(boxes))
    # print(len(boxes))
    # print(boxes)
    return boxes


def save_network_output(output_path, output, logger=None):
    if logger is not None:
        logger.info(f'Saving network output to: {output_path}...')
    output = output.detach().cpu()[0]
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('predictions', data=output, compression='gzip')


loggers = {}


def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    logger = get_logger('PatchFinder')
    in_channels = model.in_channels

    patch_shapes = [(64, 128, 128), (96, 128, 128),
                    (64, 160, 160), (96, 160, 160),
                    (64, 192, 192), (96, 192, 192)]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype('float32')

        patch = torch \
            .from_numpy(patch) \
            .view((1, in_channels) + patch.shape) \
            .to(device)

        logger.info(f"Current patch size: {shape}")
        model(patch)


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class _TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, name, batch):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert img.ndim == 2 or img.ndim == 3, 'Only 2D (HW) and 3D (CHW) images are accepted for display'

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert C == 1 or C == 3, 'Only (1, H, W) or (3, H, W) images are supported'

            return tag, img

        tagged_images = self.process_batch(name, batch)

        return list(map(_check_img, tagged_images))

    def process_batch(self, name, batch):
        raise NotImplementedError


class DefaultTensorboardFormatter(_TensorboardFormatter):
    def __init__(self, skip_last_target=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_last_target = skip_last_target

    def process_batch(self, name, batch):
        if name == 'targets' and self.skip_last_target:
            batch = batch[:, :-1, ...]

        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))
        else:
            # batch has no channel dim: NDHW
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))



def get_tensorboard_formatter(config):
    if config is None:
        return DefaultTensorboardFormatter()

    class_name = config['name']
    m = importlib.import_module('unetseg3d.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**config)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)


def create_optimizer(optimizer_config, model,model0=None):
    name = optimizer_config.get('name','Adam')
    if name == 'Adam':
        learning_rate = optimizer_config['learning_rate']
        weight_decay = optimizer_config.get('weight_decay', 0)
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        params= model.parameters() if model0 is None else list(model.parameters())+list(model0.parameters())
        optimizer = optim.Adam(params, lr=learning_rate, betas=betas, weight_decay=weight_decay)
    else:
        learning_rate = optimizer_config['learning_rate']
        weight_decay = optimizer_config.get('weight_decay', 0)
        momentum = optimizer_config.get('momentum', 0)
        params= model.parameters() if model0 is None else list(model.parameters())+list(model0.parameters())
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def create_lr_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    lr_config1 = lr_config.copy()
    class_name = lr_config1.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config1['optimizer'] = optimizer
    return clazz(**lr_config1)


def create_sample_plotter(sample_plotter_config):
    if sample_plotter_config is None:
        return None
    class_name = sample_plotter_config['name']
    m = importlib.import_module('unetseg3d.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**sample_plotter_config)


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
