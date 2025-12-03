from __future__ import print_function

from tqdm import tqdm
import time
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init

from utils_loss import *
from torch.autograd import Variable
from IPython.display import clear_output
from pynvml import *
from MS2MNet import MS2MNet

# 设置随机种子确保实验可复现
# set_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
print("Device :", nvmlDeviceGetName(handle))

# 加载模型
net = MS2MNet().cuda()

# 加载数据集
print("训练集: ", train_ids)
print("测试集: ", test_ids)
print("批次大小: ", BATCH_SIZE)
print("步长: ", Stride_Size)
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

# 优化器设置
base_lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15, 25, 35], gamma=0.1)

# 损失函数权重配置（针对Vaihingen和Potsdam数据集特点优化）
# Vaihingen和Potsdam数据集共享特点：
# 1. 边界区域模糊不清（如建筑物与道路之间的边界）
# 2. 小目标（如汽车）数量少但重要
# 3. 类别不平衡（植被通常占较大面积，而汽车等占较小面积）
if DATASET == 'Potsdam':
    # Potsdam数据集有更高的分辨率和更多的小目标
    focal_weight = 0.7  # 更高的Focal Loss权重以处理类别不平衡
    boundary_weight = 0.3  # 边界感知损失权重
    aux_weights = [0.4, 0.2]  # 深度监督辅助输出权重
else:  # Vaihingen
    focal_weight = 0.6  # Focal Loss权重
    boundary_weight = 0.4  # 更高的边界感知损失权重（Vaihingen建筑物边界更重要）
    aux_weights = [0.3, 0.2]  # 深度监督辅助输出权重


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, csv_name="1",
         epochs=1):
    # 确保网络处于评估模式
    net.eval()

    # 在测试集上使用网络
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in
                       test_ids)
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # 切换网络到推理模式
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids)):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total)):
                # 构建张量
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = torch.from_numpy(image_patches).cuda()

                min = np.min(dsm)
                max = np.max(dsm)
                dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = torch.from_numpy(dsm_patches).cuda()

                # 执行推理 - 在eval模式下只会返回主输出
                outs = net(image_patches, dsm_patches.unsqueeze(dim=1))
                outs = outs.data.cpu().numpy()

                # 填充结果数组
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()

    accuracy = metrics_data_to_csv(predictions=np.concatenate([p.ravel() for p in all_preds]),
                                   gts=np.concatenate([p.ravel() for p in all_gts]).ravel(),
                                   csv_name=csv_name, epochs=epochs)
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy

def train(net, optimizer, epochs=50, scheduler=None, weights=WEIGHTS, csv_name="1"):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 0
    acc_best = 85.0

    # 保存训练起始时间用于记录
    start_time = time.time()

    for e in range(1, epochs + 1):
        epoch_start_time = time.time()
        net.train()

        for batch_idx, (data, dsm, target) in enumerate(
                tqdm(train_loader, desc=f"Epoch {e}/{epochs}", total=len(train_loader))):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
            optimizer.zero_grad()

            # 现在outputs是一个包含主输出和辅助输出的元组
            outputs = net(data, dsm.unsqueeze(dim=1))

            # 根据训练阶段使用不同的损失函数组合 - 优化后的50个epoch训练策略
            if e <= 30:
                # 早期训练阶段：使用Focal Loss处理类别不平衡
                loss = DeepSupervisionLoss(
                    outputs,
                    target,
                    weight=weights,
                    aux_weights=aux_weights,
                    alpha_focal=focal_weight,
                    alpha_boundary=0.0,  # 不使用边界损失
                    alpha_lovasz=0.0  # 不使用Lovasz
                )
            elif e <= 60:
                # 中期训练阶段：引入边界感知损失提高边界精度
                loss = DeepSupervisionLoss(
                    outputs,
                    target,
                    weight=weights,
                    aux_weights=aux_weights,
                    alpha_focal=focal_weight * 0.8,  # 降低Focal Loss权重
                    alpha_boundary=boundary_weight,  # 增加边界感知损失
                    alpha_lovasz=0.0  # 不使用Lovasz
                )
            else:
                # 后期训练阶段：引入Lovasz Loss细化分割质量，忽略背景类
                loss = DeepSupervisionLoss(
                    outputs,
                    target,
                    weight=weights,
                    aux_weights=aux_weights,
                    alpha_focal=focal_weight * 0.6,  # 进一步降低Focal Loss权重
                    alpha_boundary=boundary_weight,  # 保持边界感知损失
                    alpha_lovasz=0.2  # 引入Lovasz Loss优化IoU
                )

            loss.backward()
            optimizer.step()
            losses[iter_] = loss.data
            losses_array = losses[max(0, iter_ - 100):iter_]
            if len(losses_array) > 0:
                mean_losses[iter_] = np.mean(losses_array)
            else:
                mean_losses[iter_] = 0

            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                # 使用主输出(outputs[0])进行预测
                pred = np.argmax(outputs[0].data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('训练 (epoch {}/{}) [{}/{} ({:.0f}%)]\t损失: {:.6f}\t准确率: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss, outputs)

            # 更智能的评估和保存策略 - 优化的50个epoch训练计划
            # 后期阶段，每1000次迭代评估一次
            if iter_ % 2000 == 0 and e >= 30:
                net.eval()
                acc = test(net, test_ids, all=False, stride=Stride_Size, csv_name=csv_name, epochs=e)
                net.train()
                torch.save(net.state_dict(), f'./checkpoint/{csv_filename}_epoch_{e}_{acc:.2f}.pth')
                if acc > acc_best:
                    #torch.save(net.state_dict(), f'./checkpoint/{csv_filename}_epoch_{e}_{acc:.2f}.pth')
                    acc_best = acc

            # 中期阶段，每2000次迭代评估一次
            # elif iter_ % 5000 == 0 and e >= 15 and e < 30:
            #     net.eval()
            #     acc = test(net, test_ids, all=False, stride=Stride_Size, csv_name=csv_name, epochs=e)
            #     net.train()
            #     if acc > acc_best:
            #         torch.save(net.state_dict(), f'./checkpoint/{csv_filename}_epoch_{e}_{acc:.2f}.pth')
            #         acc_best = acc

        # 每个epoch结束执行以下操作
        if scheduler is not None:
            scheduler.step()

        # 计算每个epoch的训练时间
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time

        # 每个epoch结束后显示当前的学习率和训练时间
        print(f"Epoch {e}/{epochs} 完成，耗时: {epoch_time:.2f}秒, 总时间: {total_time / 60:.2f}分钟")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

    print('acc_best: ', acc_best)
    clean_models(csv_filename=csv_filename)


csv_filename = "Vaihingen_MS2MNet_Result"

# 训练模型
time_start = time.time()
train(net, optimizer, 100, scheduler=scheduler, csv_name="train_" + csv_filename)
time_end = time.time()
print('总训练时间: ', time_end - time_start)