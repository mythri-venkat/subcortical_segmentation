import importlib

import numpy as np
import torch
import SimpleITK as sitk
from unetseg3d.unet3d.losses import compute_per_channel_dice
from unetseg3d.unet3d.utils import get_logger, expand_as_one_hot, convert_to_numpy

logger = get_logger('EvalMetric')


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon)[1:])

class DiceCoefficientFull:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon
        self.num_class = kwargs['num_class'] if 'num_class' in kwargs.keys() else 15

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        
        encoded_input = expand_as_one_hot(input,C=self.num_class) 
        encoded_target = expand_as_one_hot(target,C=self.num_class) 
        x=compute_per_channel_dice(encoded_input, encoded_target, epsilon=self.epsilon)
        # x=torch.mean(x[1:])
        return x

class HausdorffDistance:

    def __init__(self, num_classes=15,**kwargs):
        self.num_classes = num_classes
        self.hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()


    def __call__(self,predictions, labels):
        dice_scores = np.zeros((self.num_classes))
        
        predictions = predictions.squeeze()
        labels = labels.squeeze()

        p = sitk.GetImageFromArray(predictions.cpu().numpy().astype(np.uint8))
        l = sitk.GetImageFromArray(labels.cpu().numpy().astype(np.uint8))

        for i in range(self.num_classes):
            # if np.sum(p ==i) ==0:
            #     continue
            lTestImage = sitk.BinaryThreshold(p, i, i, 1, 0)
            lResultImage = sitk.BinaryThreshold(l, i, i, 1, 0)

            self.hausdorff_distance_filter.Execute(lTestImage, lResultImage)

            hd_value = self.hausdorff_distance_filter.GetHausdorffDistance()
            dice_scores[i] = hd_value

        return torch.Tensor(dice_scores.astype(np.float32))

class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        assert input.size() == target.size()

        per_batch_iou = []
        for _input, _target in zip(input, target):
            binary_prediction = self._binarize_predictions(_input, n_classes)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))

    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)




def get_evaluation_metric(config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    def _metric_class(class_name):
        m = importlib.import_module('unetseg3d.unet3d.metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)

def get_evaluation_metrics(config):
    metrics=[]
    def _metric_class(class_name):
        m = importlib.import_module('unetseg3d.unet3d.metrics')
        clazz = getattr(m, class_name)
        return clazz
    for m in config['eval_metric']:
        metric_class = _metric_class(m['name'])
        metrics.append(metric_class(**m))
    return metrics

