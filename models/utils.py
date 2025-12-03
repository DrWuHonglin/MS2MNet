import datetime
import glob
import re

import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import itertools
from torchvision.utils import make_grid
from PIL import Image
from skimage import io
import os
import pandas as pd

# Parameters
## SwinFusion
# WINDOW_SIZE = (64, 64) # Patch size
WINDOW_SIZE = (256, 256)  # Patch size

STRIDE = 32  # Stride for testing
IN_CHANNELS = 3  # Number of input channels (e.g. RGB)
FOLDER = "/root/autodl-tmp/project/data/"  # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10  # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]  # Label names
N_CLASSES = len(LABELS)  # Number of classes
WEIGHTS = torch.ones(N_CLASSES).cuda()  # Weights for class balancing
CACHE = True  # Store the dataset in-memory

train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
test_ids = ['5', '21', '15', '30']
# test_ids = ['2', '4', '5', '6', '8', '10', '12', '14', '15','16', '20', '21','22', '24', '27']
DATASET = 'Vaihingen'
Stride_Size = 32
MAIN_FOLDER = FOLDER + 'Vaihingen/'
DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
# 获取当前脚本（utils.py）所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造相对于当前脚本的results文件夹路径
results_dir = os.path.join(current_dir, 'results')
# train_ids = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
#              '4_12', '6_8', '6_12', '6_7', '4_11']
# test_ids = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']
# DATASET = 'Potsdam'
# Stride_Size = 64
# MAIN_FOLDER = FOLDER + 'Potsdam/'
# DATA_FOLDER = MAIN_FOLDER + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
# DSM_FOLDER = MAIN_FOLDER + '1_DSM_normalisation/dsm_potsdam_{}_normalized_lastools.jpg'
# LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
# ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'

# ISPRS color palette
# Let's define the standard ISPRS color palette
palette = {0: (255, 255, 255),  # Impervious surfaces (white)
           1: (0, 0, 255),  # Buildings (blue)
           2: (0, 255, 255),  # Low vegetation (cyan)
           3: (0, 255, 0),  # Trees (green)
           4: (255, 255, 0),  # Cars (yellow)
           5: (255, 0, 0),  # Clutter (red)
           6: (0, 0, 0)}  # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def save_img(tensor, name):
    tensor = tensor.cpu().permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')


class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        # self.boundary_files = [BOUNDARY_FOLDER.format(id) for id in ids]
        self.dsm_files = [DSM_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        # self.boundary_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Default epoch size is 10 000 samples
        if DATASET == 'Potsdam':
            return BATCH_SIZE * 1000
        elif DATASET == 'Vaihingen':
            return BATCH_SIZE * 1000
        else:
            return None

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            ## Potsdam IRRG
            if DATASET == 'Potsdam':
                ## RGB
                data = io.imread(self.data_files[random_idx])[:, :, :3].transpose((2, 0, 1))
                ## IRRG
                # data = io.imread(self.data_files[random_idx])[:, :, (3, 0, 1, 2)][:, :, :3].transpose((2, 0, 1))
                data = 1 / 255 * np.asarray(data, dtype='float32')
            else:
                ## Vaihingen IRRG
                data = io.imread(self.data_files[random_idx])
                data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        # if random_idx in self.boundary_cache_.keys():
        #     boundary = self.boundary_cache_[random_idx]
        # else:
        #     boundary = np.asarray(io.imread(self.boundary_files[random_idx])) / 255
        #     boundary = boundary.astype(np.int64)
        #     if self.cache:
        #         self.boundary_cache_[random_idx] = boundary

        if random_idx in self.dsm_cache_.keys():
            dsm = self.dsm_cache_[random_idx]
        else:
            # DSM is normalized in [0, 1]
            dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype='float32')
            min = np.min(dsm)
            max = np.max(dsm)
            dsm = (dsm - min) / (max - min)
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        # boundary_p = boundary[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        # data_p, boundary_p, label_p = self.data_augmentation(data_p, boundary_p, label_p)
        data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))


## We load one tile from the dataset and we display it
# img = io.imread('./ISPRS_dataset/Vaihingen/top/top_mosaic_09cm_area11.tif')
# fig = plt.figure()
# fig.add_subplot(121)
# plt.imshow(img)
#
# # We load the ground truth
# gt = io.imread('./ISPRS_dataset/Vaihingen/gts_for_participants/top_mosaic_09cm_area11.tif')
# fig.add_subplot(122)
# plt.imshow(gt)
# plt.show()
#
# # We also check that we can convert the ground truth into an array format
# array_gt = convert_from_color(gt)
# print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)


# Utils

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def CrossEntropy2d(input, target, weight=None, reduction='mean'):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" % (kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:5])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")

    return accuracy


def metrics_data_to_csv(predictions, gts, label_values=LABELS, csv_name="1",epochs=1):
    result_dict = {}
    result_dict["time"] = datetime.datetime.now().replace(second=0, microsecond=0)
    result_dict["epochs"] = epochs
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))
    # csv字典数据添加
    result_dict["total_accuracy"] = "%.2f" % (accuracy)
    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
        # csv字典数据添加
        if label_values[l_id] == "clutter":
            clutter = label_values[l_id]
            clutter_score = "%.4f" % (score)
        else:
            result_dict[label_values[l_id]] = "%.4f" % (score)
    print("---")
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
        # csv字典数据添加
        if label_values[l_id] == "clutter":
            clutter = label_values[l_id]
            clutter_score = "%.4f" % (score)
        else:
            result_dict[label_values[l_id]] = "%.4f" % (score)
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    # csv字典数据添加
    result_dict["mean F1Score"] = "%.4f" % (np.nanmean(F1Score[:5]))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" % (kappa))
    # csv字典数据添加
    result_dict["Kappa"] = "%.4f" % (kappa)

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIOU = np.nanmean(MIoU[:5])
    # csv字典数据添加
    result_dict["mean MIoU"] = "%.4f" % (MIOU)
    # csv字典数据添加
    result_dict["MIoU"] = ', '.join(str(item) for item in MIoU)
    result_dict[clutter] = clutter_score
    print('mean MIoU: %.4f' % (MIOU))
    print("---")

    # 创建一个DataFrame
    df = pd.DataFrame([result_dict])
    # 将DataFrame写入CSV文件
    csv_filename = os.path.join(results_dir, csv_name +".csv")  # 指定CSV文件名
    # 检查CSV文件是否存在
    if not os.path.exists(csv_filename):
        # 如果文件不存在，保存第一次数据并包含列名
        df.to_csv(csv_filename, index=False)
    else:
        # 如果文件存在，以追加模式打开文件并写入后续数据（不包含列名）
        with open(csv_filename, 'a', newline='') as f:
            df.to_csv(f, header=False, index=False)
    return accuracy


def clean_models(saved_models=[], csv_filename="baseline_Resnet50_PAFEMNoAttention_cwf"):
    """
    清理模型文件，保留准确率最高的模型，删除其他模型。

    参数:
    saved_models (list): 已保存的模型文件列表
    csv_filename (str): 模型文件的前缀名称
    """
    # 如果saved_models为空，从文件夹中加载所有模型文件
    if not saved_models:
        saved_models = glob.glob(os.path.join('.', 'checkpoint', f'{csv_filename}_epoch_*'))

    best_model = None
    best_acc = 0.0

    # 正则表达式提取准确率
    pattern = re.compile(r'{}_epoch_\d+_(\d+\.\d+)'.format(csv_filename))

    for model_path in saved_models:
        match = pattern.search(model_path)
        if match:
            acc = float(match.group(1))
            if acc > best_acc:
                best_acc = acc
                best_model = model_path

    # 删除其他模型
    for model_path in saved_models:
        if model_path != best_model:
            os.remove(model_path)
    print('Best model saved at: ', best_model)


def FocalLoss2d(input, target, gamma=2, alpha=0.25, weight=None, reduction='mean'):
    """
    Focal Loss 用于处理类别不平衡问题，给予易分类样本更小的权重
    """
    dim = input.dim()
    if dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
    else:
        output = input

    # 计算交叉熵
    logpt = -F.cross_entropy(output, target, weight=weight, reduction='none')
    pt = torch.exp(logpt)

    # 计算Focal Loss
    loss = -((1 - pt) ** gamma) * logpt

    # 应用alpha平衡因子
    if alpha is not None:
        if alpha > 0:
            alpha_t = alpha * (target > 0).float() + (1 - alpha) * (target == 0).float()
            loss = alpha_t * loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def BoundaryAwareLoss(input, target, weight=None, boundary_weight=1.5, reduction='mean'):
    """
    边界感知损失函数，增加对分割边界的关注
    """
    # 计算图像梯度（边界）
    target_one_hot = F.one_hot(target, num_classes=input.size(1)).permute(0, 3, 1, 2).float()

    # 使用Sobel算子计算梯度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=input.device).reshape(1, 1,
                                                                                                                   3,
                                                                                                                   3).repeat(
        input.size(1), 1, 1, 1)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=input.device).reshape(1, 1,
                                                                                                                   3,
                                                                                                                   3).repeat(
        input.size(1), 1, 1, 1)

    edge_x = F.conv2d(target_one_hot, sobel_x, padding=1, groups=input.size(1))
    edge_y = F.conv2d(target_one_hot, sobel_y, padding=1, groups=input.size(1))
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)

    # 创建边界权重图
    boundary_mask = (edge > 0.1).float()

    # 应用边界权重
    weight_map = torch.ones_like(target, dtype=torch.float32, device=input.device) + boundary_mask.sum(
        dim=1) * boundary_weight

    # 计算加权交叉熵损失
    dim = input.dim()
    if dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        weight_map = weight_map.view(-1)

    loss = F.cross_entropy(output, target, weight=weight, reduction='none')
    weighted_loss = loss * weight_map

    if reduction == 'mean':
        return weighted_loss.mean()
    elif reduction == 'sum':
        return weighted_loss.sum()
    else:
        return weighted_loss


def lovasz_grad(gt_sorted):
    """
    计算Lovasz扩展梯度
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # 需要至少2个样本才能计算梯度
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Lovasz softmax损失，适用于多分类的语义分割任务
    probas: [B, C, H, W] 预测概率
    labels: [B, H, W] 真值标签
    classes: 'all' 表示所有类, 'present' 表示在图像中存在的类
    per_image: 是否按照图像计算损失
    ignore: 忽略的标签值
    """
    if per_image:
        loss = torch.mean(torch.stack([lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore))
                                       for prob, lab in zip(probas, labels)]))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    计算平坦的lovasz softmax损失
    """
    if probas.numel() == 0:
        return probas.sum() * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes

    for c in class_to_sum:
        fg = (labels == c).float()  # 前景
        if fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('单通道预测时，类别必须为1')
            else:
                class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    return torch.mean(torch.stack(losses))


def flatten_probas(probas, labels, ignore=None):
    """
    展平预测概率和标签
    """
    if probas.dim() == 3:
        probas = probas.unsqueeze(1)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B*H*W, C
    labels = labels.view(-1)
    if ignore is not None:
        mask = (labels != ignore)
        probas = probas[mask]
        labels = labels[mask]
    return probas, labels


def CombinedLoss(input, target, weight=None, alpha_focal=0.5, alpha_boundary=0.3, alpha_lovasz=0.2, reduction='mean'):
    """
    组合损失函数，灵活组合多种损失函数以适应不同训练阶段

    参数:
    - input: 模型输出的预测
    - target: 真实标签
    - weight: 类别权重
    - alpha_focal: Focal Loss权重，设为0时禁用
    - alpha_boundary: 边界感知损失权重，设为0时禁用
    - alpha_lovasz: Lovasz Loss权重，设为0时禁用
    - reduction: 损失归约方式
    """
    # 基础损失总是使用交叉熵
    loss = CrossEntropy2d(input, target, weight, reduction)

    # 根据设置的权重选择性添加其他损失
    if alpha_focal > 0:
        focal_loss = FocalLoss2d(input, target, gamma=2.0, weight=weight, reduction=reduction)
        loss = loss + alpha_focal * focal_loss

    if alpha_boundary > 0:
        boundary_loss = BoundaryAwareLoss(input, target, weight, reduction=reduction)
        loss = loss + alpha_boundary * boundary_loss

    if alpha_lovasz > 0:
        # 计算Lovasz Loss (针对IOU优化)
        probs = F.softmax(input, dim=1)
        lovasz_loss = lovasz_softmax(probs, target)
        loss = loss + alpha_lovasz * lovasz_loss

    return loss


def DeepSupervisionLoss(outputs, target, weight=None, aux_weights=[0.4, 0.2], alpha_focal=0.5, alpha_boundary=0.3,
                        alpha_lovasz=0.2):
    """
    深度监督损失函数，结合主输出和辅助输出的损失
    适合Vaihingen和Potsdam数据集的特定需求:
    - Focal Loss: 处理类别不平衡问题，特别是小目标（如汽车）
    - 边界感知损失: 提高建筑物、道路等地物的边界精度
    - Lovasz Loss: 优化IoU指标，提高整体分割质量

    Args:
        outputs: 网络输出元组/列表，[main_out, aux_out1, aux_out2, ...]
        target: 标签
        weight: 类别权重，可提高小目标类别的权重
        aux_weights: 辅助输出的权重列表，按顺序对应每个辅助输出
        alpha_focal: Focal Loss权重，处理类别不平衡（0时禁用）
        alpha_boundary: 边界感知损失权重，提高边界精度（0时禁用）
        alpha_lovasz: Lovasz Loss权重，优化IoU（0时禁用）
    """
    if not isinstance(outputs, (tuple, list)):
        return CombinedLoss(outputs, target, weight, alpha_focal, alpha_boundary, alpha_lovasz)

    # 主输出损失
    main_loss = CombinedLoss(outputs[0], target, weight, alpha_focal, alpha_boundary, alpha_lovasz)

    # 辅助输出损失 - 对辅助输出仅使用交叉熵和Focal Loss，简化计算
    aux_losses = 0
    for i, aux_out in enumerate(outputs[1:]):
        if i < len(aux_weights):  # 确保有对应的权重
            # 对辅助输出使用简化版的损失组合
            if alpha_focal > 0:
                aux_loss = 0.5 * CrossEntropy2d(aux_out, target, weight) + 0.5 * FocalLoss2d(aux_out, target, gamma=2.0,
                                                                                             weight=weight)
            else:
                aux_loss = CrossEntropy2d(aux_out, target, weight)

            aux_losses += aux_weights[i] * aux_loss

    # 总损失
    total_loss = main_loss + aux_losses

    return total_loss

def set_seed(seed):

    # Python内置random模块
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA操作的随机数生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

        # 以下设置可提高性能，但可能影响可重复性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    results_dict = {'time': datetime.datetime(2024, 5, 26, 12, 2, 26, 146467), 'pixels processed': '17677675', 'total_accuracy': '92.28', 'roads': '0.9359', 'buildings': '0.9765', 'low veg.': '0.8144', 'trees': '0.9138', 'cars': '0.8935', 'clutter': '0.0000', 'mean F1Score': '0.9068', 'Kappa': '0.8960', 'MIoU': '0.8794739885407972, 0.9541319267169432, 0.6868869907327542, 0.8412681093959666, 0.8074577410679985, 0.0', 'mean MIoU': '0.8338'}

    # 假设这是你第一次保存的数据
    df_first = pd.DataFrame([results_dict])
    # 构造要保存的文件完整路径
    csv_filename = os.path.join(results_dir, "test.csv")
    # 检查CSV文件是否存在
    if not os.path.exists(csv_filename):
        # 如果文件不存在，保存第一次数据并包含列名
        df_first.to_csv(csv_filename, index=False)
    else:
        # 如果文件存在，以追加模式打开文件并写入后续数据（不包含列名）
        with open(csv_filename, 'a', newline='') as f:
            df_first.to_csv(f, header=False, index=False)