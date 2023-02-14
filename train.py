import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import show_config, time_synchronized
from utils.utils_fit import fit_one_epoch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 模型的超参数
model_cfg = dict(
    description="pytorch deeplabv3plus fusion training",
    # ---------- 数据集超参数 -----------
    data_path="../../dataset/SUIMdevkit",  # dataset root
    # ---------- 卷积模型超参数 ----------
    backbone="deeplabv3plus_fusion",
    num_classes=7,
    input_shape=[512, 512],  # the size of input image
    # TODO: ASPP下采样倍率发生变化
    downsample_factor=4,  # 主分支在ASPP前的下采样倍率
    aux_branch=True,  # auxilier loss 辅助分类器
    # ---------- 硬件的超参数 ----------
    cuda=True,
    amp=True,  # fp16 混合精度训练
    distributed=False,  # 用于指定是否使用单机多卡分布式运行
    sync_bn=False,  # 是否使用sync_bn，DDP模式多卡可用
    # ---------- 训练Epoch和Batch size超参数 ----------
    init_model_weights=True,  # 初始化模型的权重参数
    freeze_train=False,
    freeze_batch_size=16,
    unfreeze_batch_size=16,
    model_path="",
    init_epoch=0,
    freeze_epochs=0,
    unfreeze_epochs=500,
    # ---------- 训练的优化器超参数 ----------
    optimizer="adam",  # sgd, adam
    momentum=0.9,
    weight_decay=1e-4,  # weight decay (default: 1e-4)
    lr_decay_type="cos",  # "cos", "step"
    # ---------- 损失函数的超参数 ----------
    dice_loss=False,
    focal_loss=False,
    # ---------- 模型验证和保存的超参数 ----------
    save_freq=5,
    save_dir="./logs",
    eval_flag=True,  # 是否在训练时进行评估，评估对象为验证集
    eval_period=5,  # 评估周期
)


def main(model_cfg):
    # ---------- 硬件的超参数 ----------
    Cuda = model_cfg["cuda"]
    distributed = model_cfg["distributed"]
    sync_bn = model_cfg["sync_bn"]
    fp16 = model_cfg["amp"]

    # ---------- 卷积模型超参数 ----------
    num_classes = model_cfg["num_classes"]  # num_classes + background
    backbone = model_cfg["backbone"]
    downsample_factor = model_cfg["downsample_factor"]
    input_shape = model_cfg["input_shape"]  # 输入图片的大小
    aux_branch = model_cfg["aux_branch"]  # 使用辅助分类器
    model_path = model_cfg["model_path"]  # 卷积模型的预训练权重
    Init_Epoch = model_cfg["init_epoch"]
    Freeze_Epoch = model_cfg["freeze_epochs"]
    Freeze_batch_size = model_cfg["freeze_batch_size"]
    UnFreeze_Epoch = model_cfg["unfreeze_epochs"]
    Unfreeze_batch_size = model_cfg["unfreeze_batch_size"]
    Freeze_Train = model_cfg["freeze_train"]  # 是否进行冻结训练 默认先冻结主干训练后解冻训练

    # ---------- 训练的优化器超参数 ----------
    #   其它训练参数：学习率、优化器、学习率下降有关
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=7e-3
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01

    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=7e-3
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------
    optimizer_type = model_cfg["optimizer"]
    momentum = model_cfg["momentum"]
    weight_decay = model_cfg["weight_decay"]
    lr_decay_type = model_cfg["lr_decay_type"]  # 使用到的学习率下降方式，可选的有'step'、'cos'
    # Init_lr = model_cfg["init_lr"]

    # 根据训练优化器的类型设定初始学习率
    if optimizer_type == "sgd":
        Init_lr = 0.1 * Unfreeze_batch_size / 256
        # Init_lr = 7e-3  # 0.007
    elif optimizer_type == "adam":
        Init_lr = 5e-4 * Unfreeze_batch_size / 64  # B=16, Init_lr=0.000125
        # Init_lr = 5e-4  # 0.0005
    Min_lr = Init_lr * 0.01

    save_period = model_cfg["save_freq"]  # 多少个epoch保存一次权值
    save_dir = os.path.join(
        model_cfg["save_dir"], model_cfg["backbone"]
    )  # 权值与日志文件保存的文件夹

    eval_flag = model_cfg["eval_flag"]  # 是否在训练时进行评估，评估对象为验证集
    eval_period = model_cfg["eval_period"]  # 代表多少个epoch评估一次

    SUIMdevkit_path = model_cfg["data_path"]  # 数据集路径
    dice_loss = model_cfg["dice_loss"]
    focal_loss = model_cfg["focal_loss"]  # 是否使用focal loss来防止正负样本不平衡【实验观察focal loss的效果】
    # ------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)  # 给不同的种类设置不同的损失权值
    num_workers = min([os.cpu_count(), Freeze_batch_size, Unfreeze_batch_size, 8])
    ngpus_per_node = torch.cuda.device_count()  # 设置用到的显卡
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(
                f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training..."
            )
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    # ---------- 实例化卷积神经网络模型 ----------
    model = DeepLab(num_classes, backbone, downsample_factor, aux_branch)
    # ----------------------------------------

    # 载入预训练权重参数或初始化模型的权重参数
    if model_path != "":
        if local_rank == 0:
            print("Load weights {}.".format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ---------- 显示没有匹配上的Key ----------
        if local_rank == 0:
            print(
                "\nSuccessful Load Key:",
                str(load_key)[:500],
                "……\nSuccessful Load Key Num:",
                len(load_key),
            )
            print(
                "\nFail To Load Key:",
                str(no_load_key)[:500],
                "……\nFail To Load Key num:",
                len(no_load_key),
            )
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    elif model_cfg["init_model_weights"]:
        weights_init(model, init_type="kaiming")
    else:
        raise ValueError("请检查模型载入权重参数的形式（自动初始化或加载预训练权重）")

    # 记录Loss
    if local_rank == 0:
        time_str = datetime.datetime.strftime(
            datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S"
        )
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # 混合精度训练
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    # 多卡同步Bn
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # 多卡并行训练
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                module=model_train, device_ids=[local_rank], find_unused_parameters=True
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # 读取数据集对应的txt
    with open(
        os.path.join(SUIMdevkit_path, "SUIM2022/ImageSets/Segmentation/train.txt"), "r"
    ) as f:
        train_lines = f.readlines()
    with open(
        os.path.join(SUIMdevkit_path, "SUIM2022/ImageSets/Segmentation/val.txt"), "r"
    ) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes,
            backbone=backbone,
            model_path=model_path,
            input_shape=input_shape,
            Init_Epoch=Init_Epoch,
            Freeze_Epoch=Freeze_Epoch,
            UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size,
            Unfreeze_batch_size=Unfreeze_batch_size,
            Freeze_Train=Freeze_Train,
            Init_lr=Init_lr,
            Min_lr=Min_lr,
            optimizer_type=optimizer_type,
            momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period,
            save_dir=save_dir,
            num_workers=num_workers,
            num_train=num_train,
            num_val=num_val,
        )
        # ---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        # ----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print(
                "\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"
                % (optimizer_type, wanted_step)
            )
            print(
                "\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"
                % (num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step)
            )
            print(
                "\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"
                % (total_step, wanted_step, wanted_epoch)
            )

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    UnFreeze_flag = False
    # 冻结模型的参数
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False
    # 如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    # ---------- 配置训练的优化器 ----------
    #  判断当前batch_size，自适应调整学习率
    nbs = 16
    lr_limit_max = 5e-4 if optimizer_type == "adam" else 1e-1
    lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
    # if backbone == "xception":
    #     lr_limit_max = 1e-4 if optimizer_type == "adam" else 1e-1
    #     lr_limit_min = 1e-4 if optimizer_type == "adam" else 5e-4
    Init_lr_fit = min(
        max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max
    )  # Init_lr_fit = 7e-3
    Min_lr_fit = min(
        max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2
    )  # Min_lr_fit = 7e-5

    # 根据optimizer_type选择优化器
    optimizer = {
        "adam": optim.Adam(
            params=model.parameters(),
            lr=Init_lr_fit,
            betas=(momentum, 0.999),
            weight_decay=weight_decay,
        ),
        "sgd": optim.SGD(
            params=model.parameters(),
            lr=Init_lr_fit,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay,
        ),
    }[optimizer_type]

    # 获得学习率下降的公式
    lr_scheduler_func = get_lr_scheduler(
        lr_decay_type,
        Init_lr_fit,
        Min_lr_fit,
        total_iters=UnFreeze_Epoch,
    )

    # 判断每一个世代的长度
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    # ---------- 实例化训练集和测试集 ----------
    train_dataset = DeeplabDataset(
        train_lines,
        input_shape,
        num_classes,
        train=True,
        dataset_path=SUIMdevkit_path,
    )
    val_dataset = DeeplabDataset(
        val_lines,
        input_shape,
        num_classes,
        train=False,
        dataset_path=SUIMdevkit_path,
    )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            shuffle=False,
        )
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    # 将训练数据载入内存
    gen = DataLoader(
        dataset=train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=deeplab_dataset_collate,
        sampler=train_sampler,
    )

    gen_val = DataLoader(
        dataset=val_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=deeplab_dataset_collate,
        sampler=val_sampler,
    )

    # 记录eval的map曲线
    if local_rank == 0:
        eval_callback = EvalCallback(
            net=model,
            input_shape=input_shape,
            num_classes=num_classes,
            image_ids=val_lines,
            dataset_path=SUIMdevkit_path,
            log_dir=log_dir,
            cuda=Cuda,
            eval_flag=eval_flag,
            period=eval_period,
        )
    else:
        eval_callback = None

    # ---------------------------------------#
    #   开始模型训练
    # ---------------------------------------#
    start_time = time_synchronized()
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # ---------------------------------------#
        #   如果模型有冻结学习部分
        #   则解冻，并设置参数
        # ---------------------------------------#
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size

            # -------------------------------------------------------------------#
            #   判断当前batch_size，自适应调整学习率
            # -------------------------------------------------------------------#
            nbs = 16
            lr_limit_max = 5e-4 if optimizer_type == "adam" else 1e-1
            lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
            # if backbone == "xception":
            #     lr_limit_max = 1e-4 if optimizer_type == "adam" else 1e-1
            #     lr_limit_min = 1e-4 if optimizer_type == "adam" else 5e-4
            Init_lr_fit = min(
                max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max
            )
            Min_lr_fit = min(
                max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2),
                lr_limit_max * 1e-2,
            )
            # ---------------------------------------#
            #   获得学习率下降的公式
            # ---------------------------------------#
            lr_scheduler_func = get_lr_scheduler(
                lr_decay_type=lr_decay_type,
                lr=Init_lr_fit,
                min_lr=Min_lr_fit,
                total_iters=UnFreeze_Epoch,
            )

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            if distributed:
                batch_size = batch_size // ngpus_per_node

            gen = DataLoader(
                dataset=train_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=deeplab_dataset_collate,
                sampler=train_sampler,
            )
            gen_val = DataLoader(
                dataset=val_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=deeplab_dataset_collate,
                sampler=val_sampler,
            )

            UnFreeze_flag = True

        if distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(
            model_train,
            model,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            gen,
            gen_val,
            UnFreeze_Epoch,
            Cuda,
            dice_loss,
            focal_loss,
            cls_weights,
            aux_branch,
            num_classes,
            fp16,
            scaler,
            save_period,
            save_dir,
            local_rank,
        )

        if distributed:
            dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()

    total_time = time_synchronized() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("*********** 🚀🚀🚀🚀 Finish training! 🚀🚀🚀🚀 ***********")
    print("*********** training time {} ***********".format(total_time_str))


if __name__ == "__main__":
    main(model_cfg)
