import os

from PIL import Image
from tqdm import tqdm

from deeplab_segmentation import DeeplabV3_Segmentation
from utils.utils_metrics import compute_mIoU, show_results

"""
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
"""

val_cfg = dict(
    description="model validation",
    # ---------- 验证模式的参数 ----------
    miou_mode=0,  # 0, 1, 2
    mix_type=1,  # 0混合, 1仅原图, 2仅原图中的目标_扣去背景 get_miou不起作用
    # ---------- 卷积模型的参数 ----------
    # best_epoch_weights.pth
    # last_epoch_weights.pth
    model_path="./logs/deeplabv3plus_fusion/05_DeepLabV3Plus_fusion_version2_MobileVit_d4_Normal_500epochs_bs16_adam/ep500-loss0.236-val_loss0.460.pth",
    backbone="deeplabv3plus_fusion",
    aux_branch=False,
    num_classes=7,
    name_classes=[
        "Background_waterbody",
        "Human_divers",
        "Wrecks_and_ruins",
        "Robots",
        "Reefs_and_invertebrates",
        "Fish_and_vertebrates",
        "sea_floor_and_rocks",
    ],
    input_shape=[512, 512],
    downsample_factor=4,
    deploy=True,
    cuda=True,
    # ---------- 文件夹的位置参数 ----------
    dataset_path="../../dataset/SUIMdevkit",
    file_name="train.txt",
    save_file_dir="./miou_out_train_500",
)


def main(val_cfg):
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = val_cfg["miou_mode"]
    num_classes = val_cfg["num_classes"]
    name_classes = val_cfg["name_classes"]
    SUIMdevkit_path = val_cfg["dataset_path"]

    image_ids = (
        open(
            os.path.join(
                SUIMdevkit_path, "SUIM2022/ImageSets/Segmentation", val_cfg["file_name"]
            ),
            "r",
        )
        .read()
        .splitlines()
    )
    gt_dir = os.path.join(SUIMdevkit_path, "SUIM2022/SegmentationClass/")
    miou_out_path = val_cfg["save_file_dir"]
    pred_dir = os.path.join(miou_out_path, "detection-results")

    # ---------- 生成预测的mask ----------
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("💾💾💾 Load model")
        deeplab = DeeplabV3_Segmentation(
            val_cfg["model_path"],
            val_cfg["num_classes"],
            val_cfg["backbone"],
            val_cfg["input_shape"],
            val_cfg["downsample_factor"],
            val_cfg["aux_branch"],
            val_cfg["mix_type"],
            val_cfg["cuda"],
            val_cfg["deploy"],
        )
        print("💾💾💾 Load model done")

        print("---------- Get predict result ----------")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(
                SUIMdevkit_path, "SUIM2022/JPEGImages/" + image_id + ".jpg"
            )
            image = Image.open(image_path)
            image = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("---------- Get predict result done ----------")

    # ---------- 计算预测的mask和真实的mask 混淆矩阵 ----------
    if miou_mode == 0 or miou_mode == 2:
        print("---------- Get miou ----------")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, pred_dir, image_ids, num_classes, name_classes
        )  # 执行计算mIoU的函数
        print("---------- Get miou done ----------")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)


if __name__ == "__main__":
    main(val_cfg)
