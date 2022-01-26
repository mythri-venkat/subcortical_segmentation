import nibabel as nib
import pandas as pd
import numpy as np
import sys
import json
import SimpleITK as sitk


def dice_np(predictions, labels, num_classes=15):
    """Calculates the categorical Dice similarity coefficients for each class
        between labels and predictions.

    Args:
        predictions (np.ndarray): predictions
        labels (np.ndarray): labels
        num_classes (int): number of classes to calculate the dice
            coefficient for

    Returns:
        np.ndarray: dice coefficient per class
    """

    dice_scores = np.zeros((num_classes))
    for i in range(num_classes):

        tmp_den = (np.sum(predictions == i) + np.sum(labels == i))
        tmp_dice = 2. * np.sum((predictions == i) * (labels == i)) / \
            tmp_den if tmp_den > 0 else 1.
        dice_scores[i] = tmp_dice
    return dice_scores.astype(np.float32)

def hd(predictions, labels, num_classes=15):
    """Calculates the categorical Dice similarity coefficients for each class
        between labels and predictions.

    Args:
        predictions (np.ndarray): predictions
        labels (np.ndarray): labels
        num_classes (int): number of classes to calculate the dice
            coefficient for

    Returns:
        np.ndarray: dice coefficient per class
    """

    dice_scores = np.zeros((num_classes))

    
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    p = sitk.GetImageFromArray(predictions.astype(np.uint8))
    l = sitk.GetImageFromArray(labels.astype(np.uint8))

    for i in range(num_classes):
        lTestImage = sitk.BinaryThreshold(p, i, i, 1, 0)
        lResultImage = sitk.BinaryThreshold(l, i, i, 1, 0)

        hausdorff_distance_filter.Execute(lTestImage, lResultImage)

        hd_value = hausdorff_distance_filter.GetHausdorffDistance()
        dice_scores[i] = hd_value

    return dice_scores.astype(np.float32)

def mhd(predictions, labels, num_classes=15):
    """Calculates the categorical Dice similarity coefficients for each class
        between labels and predictions.

    Args:
        predictions (np.ndarray): predictions
        labels (np.ndarray): labels
        num_classes (int): number of classes to calculate the dice
            coefficient for

    Returns:
        np.ndarray: dice coefficient per class
    """
    dice_scores = np.zeros((num_classes))

    import SimpleITK as sitk
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    p = sitk.GetImageFromArray(predictions.astype(np.uint8))
    l = sitk.GetImageFromArray(labels.astype(np.uint8))

    for i in range(num_classes):
        if np.sum(predictions ==i) ==0:
            continue
        lTestImage = sitk.BinaryThreshold(p, i, i, 1, 0)
        lResultImage = sitk.BinaryThreshold(l, i, i, 1, 0)

        hausdorff_distance_filter.Execute(lTestImage, lResultImage)

        hd_value = hausdorff_distance_filter.GetAverageHausdorffDistance()
        dice_scores[i] = hd_value

    return dice_scores.astype(np.float32)

i=sys.argv[1]
splitpath = '../splits/{}.pkl'.format(i)
predpath = '/ssd_scratch/cvit/mythri.v/predictions/'

savepath = './'

with open(splitpath,'r') as f:
    lsttest= json.load(f)['test']

scores = []
lsthd = []
lstmhd = []

for path in lsttest:
    subj = path.split("/")[-1]
    print(subj)
    pred = nib.load(predpath+subj+'.nii.gz').get_fdata()
    gt = nib.load(path+'_seg_ana_1mm_center_cropped.nii.gz').get_fdata()
    dice = dice_np(pred,gt)
    print(np.mean(dice[1:]))
    scores.append(dice)
    lsthd.append(hd(np.squeeze(pred),np.squeeze(gt)))
    lstmhd.append(mhd(np.squeeze(pred),np.squeeze(gt)))

dct = {}
scores_np = np.array(scores)
avg_dice =np.mean(scores_np[:,1:],axis=0)
std_dice=np.std(scores_np[:,1:],axis=0)
avg_dice_lst = avg_dice.tolist()

avg_dice_lst.extend([np.mean(avg_dice),np.mean(avg_dice[2:])])
std_dice_lst = std_dice.tolist()
std_dice_lst.extend([np.mean(std_dice),np.mean(std_dice[2:])])
dct['avg_dice']=avg_dice_lst
dct['std_dice']=std_dice_lst

hd_np = np.array(lsthd)
avg_hd = np.mean(hd_np[:,1:],axis=0)
std_hd = np.std(hd_np[:,1:],axis=0)
avg_hd_lst = avg_hd.tolist()
avg_hd_lst.extend([np.mean(avg_hd),np.mean(avg_hd[2:])])
std_hd_lst = std_hd.tolist()
std_hd_lst.extend([np.mean(std_hd),np.mean(std_hd[2:])])
dct['avg_hd']=avg_hd_lst
dct['std_hd']=std_hd_lst

mhd_np = np.array(lstmhd)
mavg_hd = np.mean(mhd_np[:,1:],axis=0)
mstd_hd = np.std(mhd_np[:,1:],axis=0)
mavg_hd_lst = mavg_hd.tolist()
mavg_hd_lst.extend([np.mean(mavg_hd),np.mean(mavg_hd[2:])])
mstd_hd_lst = mstd_hd.tolist()
mstd_hd_lst.extend([np.mean(mstd_hd),np.mean(mstd_hd[2:])])
dct['avg_mhd']=mavg_hd_lst
dct['std_mhd']=mstd_hd_lst
df = pd.DataFrame(dct)
dirpath = sys.argv[2] if len(sys.argv) == 3 else '.'
df.to_csv('{0}/summary_test_{1}.csv'.format(dirpath,i.split("_")[1]))
print(dct['avg_dice'])
print(dct['std_dice'])
print("dice:",dct['avg_dice'][-2])

