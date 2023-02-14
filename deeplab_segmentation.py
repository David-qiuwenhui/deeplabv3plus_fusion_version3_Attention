import colorsys
import copy
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.deeplabv3_plus import DeepLab
from utils.utils import (
    cvtColor,
    preprocess_input,
    resize_image,
    show_config,
    time_synchronized,
)

# -------------------------------------------------#
#   mix_type参数用于控制检测结果的可视化方式
#
#   mix_type = 0的时候代表原图与生成的图进行混合
#   mix_type = 1的时候代表仅保留生成的图
#   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
# -------------------------------------------------#


class DeeplabV3_Segmentation(object):
    # ---------------------------------------------------#
    #   初始化Deeplab
    # ---------------------------------------------------#
    def __init__(
        self,
        model_path,
        num_classes,
        backbone,
        input_shape,
        downsample_factor,
        aux,
        mix_type,
        cuda,
        deploy,
        **kwargs,
    ):
        self._defaults = {}
        self._defaults["model_path"] = model_path
        self._defaults["num_classes"] = num_classes
        self._defaults["backbone"] = backbone
        self._defaults["input_shape"] = input_shape
        self._defaults["downsample_factor"] = downsample_factor
        self._defaults["aux"] = aux
        self._defaults["mix_type"] = mix_type
        self._defaults["cuda"] = cuda
        self._defaults["deploy"] = deploy
        self.__dict__.update(self._defaults)

        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [
                (0, 0, 0),
                (0, 0, 128),
                (0, 128, 128),
                (128, 0, 0),
                (128, 0, 128),
                (128, 128, 0),
                (128, 128, 128),
            ]
        else:
            hsv_tuples = [
                (x / self.num_classes, 1.0, 1.0) for x in range(self.num_classes)
            ]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(
                    lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors,
                )
            )

        # ---------- 读取调色板 ----------
        palette_path = "./palette_suim.json"
        with open(palette_path, "rb") as f:
            palette_dict = json.load(f)
            palette = []
            for v in palette_dict.values():
                palette += v
        self.palette = palette

        # ---------------------------------------------------#
        #   获得模型
        # ---------------------------------------------------#
        self.generate()
        # 打印超参数
        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        self.net = DeepLab(
            self.num_classes, self.backbone, self.downsample_factor, aux_branch=self.aux
        )

        device_type = None
        if torch.cuda.is_available() and self.cuda:
            device_type = "cuda"
        else:
            device_type = "cpu"
        print(f"\033[1;36;46m 🔌🔌🔌🔌Use {device_type} for predicting \033[0m")
        device = torch.device(device_type)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 载入训练完成的模型权重
        self.net.load_state_dict(
            torch.load(self.model_path, map_location=device), strict=False
        )
        # repvgg切换到部署模式需要先载入模型的训练权重参数 再切换成部署模式
        if self.deploy:
            self.net.switch_to_deploy()
        self.net = self.net.eval()
        print("{} model, and classes loaded.".format(self.model_path))
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(
            image, (self.input_shape[1], self.input_shape[0])
        )
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
            0,
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images).main
            pr = pr.squeeze(dim=0)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[
                int((self.input_shape[0] - nh) // 2) : int(
                    (self.input_shape[0] - nh) // 2 + nh
                ),
                int((self.input_shape[1] - nw) // 2) : int(
                    (self.input_shape[1] - nw) // 2 + nw
                ),
            ]
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv2.resize(
                pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR
            )
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = orininal_h * orininal_w
            print("-" * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print("-" * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print(
                        "|%25s | %15s | %14.2f%%|"
                        % (str(name_classes[i]), str(num), ratio)
                    )
                    print("-" * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(
                np.array(self.colors, np.uint8)[np.reshape(pr, [-1])],
                [orininal_h, orininal_w, -1],
            )
            # ------------------------------------------------#
            #   将新图片转换成Image的形式
            # ------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
            # ------------------------------------------------#
            #   将新图与原图及进行混合
            # ------------------------------------------------#
            image = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(
                np.array(self.colors, np.uint8)[np.reshape(pr, [-1])],
                [orininal_h, orininal_w, -1],
            )
            # ------------------------------------------------#
            #   将新图片转换成Image的形式
            # ------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (
                np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)
            ).astype("uint8")
            # ------------------------------------------------#
            #   将新图片转换成Image的形式
            # ------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))

        return image

    def get_FPS(self, image, test_interval):
        from tqdm import tqdm

        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(
            image, (self.input_shape[1], self.input_shape[0])
        )
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
            0,
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images).main
            pr = pr.squeeze(dim=0)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[
                int((self.input_shape[0] - nh) // 2) : int(
                    (self.input_shape[0] - nh) // 2 + nh
                ),
                int((self.input_shape[1] - nw) // 2) : int(
                    (self.input_shape[1] - nw) // 2 + nw
                ),
            ]

        t1 = time_synchronized()
        for _ in tqdm(range(test_interval)):
            with torch.no_grad():
                # ---------------------------------------------------#
                #   图片传入网络进行预测
                # ---------------------------------------------------#
                pr = self.net(images).main
                pr = pr.squeeze(dim=0)
                # ---------------------------------------------------#
                #   取出每一个像素点的种类
                # ---------------------------------------------------#
                pr = (
                    F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                )
                # --------------------------------------#
                #   将灰条部分截取掉
                # --------------------------------------#
                pr = pr[
                    int((self.input_shape[0] - nh) // 2) : int(
                        (self.input_shape[0] - nh) // 2 + nh
                    ),
                    int((self.input_shape[1] - nw) // 2) : int(
                        (self.input_shape[1] - nw) // 2 + nw
                    ),
                ]
        tact_time = (time_synchronized() - t1) / test_interval
        return tact_time

    def get_miou_png(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(
            image, (self.input_shape[1], self.input_shape[0])
        )
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
            0,
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images).main
            pr = pr.squeeze(dim=0)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[
                int((self.input_shape[0] - nh) // 2) : int(
                    (self.input_shape[0] - nh) // 2 + nh
                ),
                int((self.input_shape[1] - nw) // 2) : int(
                    (self.input_shape[1] - nw) // 2 + nw
                ),
            ]
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv2.resize(
                pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR
            )
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        mask = Image.fromarray(np.uint8(pr))
        mask.putpalette(self.palette, rawmode="BGR")
        return mask
