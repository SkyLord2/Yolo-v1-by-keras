# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/10/01$ 16:58$
# @Author  : CDS
# @Email   : chengdongsheng@outlook.com
# @File    : YOLOv1.py
# Description :YOLO v1 模型搭建
# --------------------------------------

import sys
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

class YOLOv1Net(object):
    def __init__(self, training=True, model_path = None):
        self.classes = cfg.CLASSES                                                  # 分类类别（20个）
        self.num_class = len(self.classes)                                          # 类别数量
        self.image_size = cfg.IMAGE_SIZE                                            # 图片大小448
        self.cell_size = cfg.CELL_SIZE                                              # 网格数量7
        self.boxes_per_cell = cfg.BOXES_PER_CELL                                    # 每个网格预测的bounding box数量2
        self.out_size = self.cell_size**2 * (self.boxes_per_cell*5 + self.num_class)  # 输出张量的形状(7*7*(2*5+20))
        self.scale = 1.0 * self.image_size/self.cell_size                           # 没和grid cell的像素
        self.boundary1 = self.cell_size**2 * self.num_class                         # 网格数(49) * 类别数(20)
        self.boundary2 = self.boundary1 + self.cell_size**2 * self.boxes_per_cell   # 网格数(49) * 类别数(20) + 网格数(49) * 边界框数(2)
        self.object_scale = cfg.OBJECT_SCALE                                        # 损失函数中的参数：有对象的缩放系数
        self.noobject_scale = cfg.NOOBJECT_SCALE                                    # 损失函数中的参数：无对象的缩放系数
        self.class_scale = cfg.CLASS_SCALE                                          # 损失函数中的参数
        self.coord_scale = cfg.COORD_SCALE                                          # 损失函数中的参数

        self.learning_rate = cfg.LEARNING_RATE                                      # 学习率
        self.batch_size = cfg.BATCH_SIZE                                            # 批量
        self.alpha = cfg.ALPHA                                                      # 激活函数Leaky Relu的泄露参数

        self.offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                              (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
                                                                                    #

        self.training = training                                                    # 训练标记
        self.momentum = cfg.MOMENTUM
        self.epochs = cfg.MAX_ITER

        self.model = self.build_network(num_outputs=self.out_size, alpha=self.alpha, training= training)  # 构建网络，预测值，在本程序中，其格式为 [batch_size , 7 * 7 * （20 + 2 * 5）]，其中的20表示PASCAL VOC数据集的20个类别
        # 加载模型
        if (model_path is None):
            self.model_load(model_path)
    """
    1.正则化 keras.regularizer.Regularizer.l2
    2.LeakRelu激活函数 keras.layers.LeakyRelu
    3.图片缩放为224*224
    4.核心思想：首先将图片中最长的边缩放到目标尺寸，然后再对短边进行“补”
    """
    def build_network(self, num_outputs, alpha, keep_prob=0.5, training=True):
        keras.backend.clear_session()

        model = keras.Sequential()
        # 1
        model.add(keras.layers.Input(shape=(self.image_size, self.image_size, 3)))
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [3, 3], [3, 3], [0, 0]])),name="pad0"))
        # input_shape=(self.image_size, self.image_size, 3),
        model.add(keras.layers.Conv2D(filters=64, kernel_size=7, strides=(2, 2), padding="valid", name="conv1"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu1"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=2, name="pool1"))
        # 2
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [3, 3], [3, 3], [0, 0]])),name="pad1"))
        model.add(keras.layers.Conv2D(filters=192, kernel_size=7, strides=(1, 1), padding="valid", name="conv2"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu2"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=2, name="pool2"))
        # 3
        model.add(keras.layers.Conv2D(filters=128, kernel_size=1, strides=(1, 1), padding="valid", name="conv3"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu3"))
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad2"))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding="valid", name="conv4"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu4"))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=1, strides=(1, 1), padding="valid", name="conv5"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu5"))
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad3"))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="valid", name="conv6"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu6"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=2, name="pool3"))
        # 4
        model.add(keras.layers.Conv2D(filters=256, kernel_size=1, strides=(1, 1), padding="valid", name="conv7"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu7"))
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad4"))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="valid", name="conv8"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu8"))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=1, strides=(1, 1), padding="valid", name="conv9"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu9"))
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad5"))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="valid", name="conv10"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu10"))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=1, strides=(1, 1), padding="valid", name="conv11"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu11"))
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad6"))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="valid", name="conv12"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu12"))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=1, strides=(1, 1), padding="valid", name="conv13"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu13"))
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad7"))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="valid", name="conv14"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu14"))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=1, strides=(1, 1), padding="valid", name="conv15"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu15"))
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad8"))
        model.add(keras.layers.Conv2D(filters=1024, kernel_size=3, strides=(1, 1), padding="valid", name="conv16"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu16"))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), padding="same", strides=2, name="pool4"))
        # 5
        model.add(keras.layers.Conv2D(filters=512, kernel_size=1, strides=(1, 1), padding="valid", name="conv17"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu17"))
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad9"))
        model.add(keras.layers.Conv2D(filters=1024, kernel_size=3, strides=(1, 1), padding="valid", name="conv18"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu18"))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=1, strides=(1, 1), padding="valid", name="conv19"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu19"))
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad10"))
        model.add(keras.layers.Conv2D(filters=1024, kernel_size=3, strides=(1, 1), padding="valid", name="conv20"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu20"))
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad11"))
        model.add(keras.layers.Conv2D(filters=1024, kernel_size=3, strides=(1, 1), padding="valid", name="conv21"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu21"))
        # 有一个pad补全？？？？
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad12"))
        model.add(keras.layers.Conv2D(filters=1024, kernel_size=3, strides=(2, 2), padding="valid", name="conv22"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu22"))
        # 6
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad13"))
        model.add(keras.layers.Conv2D(filters=1024, kernel_size=3, strides=(1, 1), padding="valid", name="conv23"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu23"))
        model.add(keras.layers.Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad14"))
        model.add(keras.layers.Conv2D(filters=1024, kernel_size=3, strides=(1, 1), padding="valid", name="conv24"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu24"))
        # 7
        # 转置 由[batch, image_height,image_width,channels]变成[bacth, channels, image_height,image_width]
        model.add(keras.layers.Lambda(lambda x: tf.transpose(x, [0, 3, 1, 2], name='trans0'), name="trans0"))
        # 展平处理
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, name="fc1"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu25"))
        model.add(keras.layers.Dense(4096, name="fc2"))
        model.add(keras.layers.LeakyReLU(alpha=alpha, name="leaky_relu26"))
        model.add(keras.layers.Dropout(rate=keep_prob, name="dropout1"))
        model.add(keras.layers.Dense(num_outputs, activation=None, name="fc3"))
        # 调整输出的形状为 [batch_size, 7, 7, 30]
        model.add(keras.layers.Lambda(lambda x: tf.reshape(x, [-1, self.cell_size, self.cell_size, self.boxes_per_cell * 5 + self.num_class]), name="reshape_out"))
        return model

    def compile_model(self):
        """
        论文中的角动量为0.9，学习率为 1e-4
        :param model: 学习模型
        :return: None
        """
        optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)
        self.model.compile(optimizer = optimizer, loss = self.loss, metrics=['accuracy'])

    def train(self, data, labels, learning_scheduler = None):
        """
        论文中的学习率调整策略如下：在第一个迭代我们缓慢的将学习率从1e-3提升至1e-2。如果我们以较高的学习率开始训练，
        我们的模型通常会由于不稳定的梯度而发散。 我们继续以1e-2的学习率训练75个迭代，然后以1e-3的学习率训练30个迭代，
        最后以1e-4的学习率训练30个迭代。
        :param data: 训练数据
        :param labels: 标签
        :return:
        """
        if(learning_scheduler is None):
            def lr_scheduler(epoch):
                lr = 1e-4
                if(epoch <= 75):
                    lr = 1e-2
                elif(75 < epoch and epoch <= 105):
                    lr = 1e-3
                elif(105 < epoch and epoch <= 135):
                    lr = 1e-4
                return lr
            learning_scheduler = lr_scheduler
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(learning_scheduler)
        history = self.model.fit(data, labels, batch_size=self.batch_size, epochs=self.epochs, callbacks=[lr_schedule])
        return history

    def train_generator(self, generator, step_per_epoch, epochs, callbacks=None):
        self.model.fit_generator(generator, steps_per_epoch=step_per_epoch, epochs=epochs, callbacks=callbacks)

    def model_summary(self):
        self.model.summary()

    def model_save(self, path):
        """
        保存模型
        :param path: 保存路径 yolo_v1_model.h5
        :return:
        """
        self.model.save_weights(path)

    def model_load(self, path):
        self.model.load_weights(path)

    def plot_history(self, history):
        plt.figure(figsize=(16, 12))
        plt.semilogx(history.history["lr"], history.history["loss"])
        plt.grid(True)
        plt.axis([1e-5, 1e-1, 0, 5])

    def calc_iou(self,boxes1, boxes2, scope="iou"):
        """
        计算交并比
        Args:
            boxes1: 5-D 张量 预测边界框        [BATCH_SIZE, 7, 7, 2, 4]  ====> (x_center, y_center, w, h)
            boxes2: 5-D 张量 ground truth     [BATCH_SIZE, 7, 7, 2, 4]  ====> (x_center, y_center, w, h)
        Return:
            iou:    4-D 张量 [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        boxes1_t = tf.stack([
            boxes1[..., 0] - boxes1[..., 2] / 2.0,  # 左下角x坐标
            boxes1[..., 1] - boxes1[..., 3] / 2.0,  # 左下角y坐标
            boxes1[..., 0] + boxes1[..., 2] / 2.0,  # 右上角x坐标
            boxes1[..., 1] + boxes1[..., 3] / 2.0   # 右上角y坐标
        ], axis=-1)
        boxes2_t = tf.stack([
            boxes2[..., 0] - boxes2[..., 2] / 2.0,
            boxes2[..., 1] - boxes2[..., 3] / 2.0,
            boxes2[..., 0] + boxes2[..., 2] / 2.0,
            boxes2[..., 1] + boxes2[..., 3] / 2.0
        ], axis=-1)

        # 计算交叉面积
        # 左下角坐标
        ld = tf.maximum(boxes1_t[...,:2], boxes2_t[...,:2])
        # 右上角坐标
        ru = tf.minimum(boxes1_t[...,2:], boxes2_t[...,2:])
        # 交叉面积
        intersection = tf.maximum(0.0, ru - ld)
        inter_square = intersection[...,0] * intersection[...,1]
        # 两个bounding box各自的面积(w * h)
        square1 = boxes1[...,2] * boxes1[...,3]
        square2 = boxes2[...,2] * boxes2[...,3]
        # 总面积
        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
        # 将张量中的每一个元素的值压缩到0.0与1.0之间
        return tf.clip_by_value(inter_square/union_square, 0.0, 1.0)

    def loss(self, labels, predicts):
        """
        损失函数定义分为四个部分：
        ①边界框中心坐标的预测误差。
        ②边界框宽高的预测误差。
        ③包含目标的边界框置信度误差。
        ④不包含目标的边界框置信度误差。
        ⑤类别预测误差。
        :param predicts:    网络预测，4-D张量，形如：[batch_size, 7, 7, (2*5+20))]
        :param labels:      真实标签: 4-D张量, 形如：[batch_size, 7, 7, 25]
        :param scope:
        :return:
        """
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # 将predicts reshape到[batch_size, (7*7*(2*5+20))]=[batch_size, 1470]
        predicts = tf.reshape(predicts, [-1, self.cell_size * self.cell_size * (self.boxes_per_cell * 5 + self.num_class)])
        # self.boundary1 = 7*7*20
        # 预测的类别概率, 形状(batch_size, 7, 7, 20)
        predict_classes = tf.reshape(predicts[:, : self.boundary1], [self.batch_size, self.cell_size, self.cell_size, self.num_class])
        # self.boundary2 = 7*7*(20 + 2)  网格数(49) * 类别数(20) + 网格数(49) * 边界框数(2)
        # 预测置信度, 形状(batch_size, 7, 7, 2)
        predict_confidences = tf.reshape(predicts[:, self.boundary1 : self.boundary2], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
        # 预测边界框，中心坐标以及边界框的宽和高(x,y,w,h), 形状(batch_size, 7, 7, 2, 4)
        predict_boxes = tf.reshape(predicts[:, self.boundary2:], [self.batch_size,self.cell_size, self.cell_size, self.boxes_per_cell, 4])

        print("predict class shape: ", predict_classes.shape, " predict confidence shape: ", predict_confidences.shape, " predict boxes shape: ", predict_boxes.shape)

        # 对labels(batch_size, 7, 7, 25)进行拆分
        # 哪一个grid cell来负责预测目标(最后一个轴的第0个维度), 形状(batch_size, 7, 7, 1)
        response = tf.reshape(labels[...,0], (self.batch_size, self.cell_size, self.cell_size, 1))
        # grid cell 负责预测的边界框的位置(x_center,y_center,width,height), 形状(batch_size, 7, 7, 1, 4)
        boxes = tf.reshape(labels[...,1:5], (self.batch_size, self.cell_size, self.cell_size, 1, 4))
        # 复制boxes的第四个轴
        # tile对张量进行扩展，按照轴进行一定规则的复制[1,1,1,self.boxes_per_cell,1]
        # 形状(batch_size, 7, 7, 2, 4)
        boxes= tf.tile(boxes, [1,1,1,self.boxes_per_cell,1])/self.image_size
        # 对象的类别, 形状(batch_size, 7, 7, 20)
        classes = labels[...,5:]
        # 增加一个维度，维数为1，形状(1,7,7,2)
        offset = tf.reshape(tf.constant(self.offset, dtype=tf.float32), (1, self.cell_size, self.cell_size, self.boxes_per_cell))
        # 将第一维复制batch_size次, 形状(batch_size, 7, 7, 2)
        offset = tf.tile(offset,[self.batch_size, 1, 1, 1])
        # 将第一轴与第二轴交换位置, 形状(batch_size, 7, 7, 2)
        offset_tran = tf.transpose(offset, (0, 2, 1, 3))
        # 预测边界框的中心坐标与宽高, 形状(batch_size, 7, 7, 2, 4)
        predict_boxes_tran = tf.stack([
            (predict_boxes[..., 0] + offset)/self.cell_size,                # 边界框中心x坐标加上对应grid cell的偏移量
            (predict_boxes[..., 1] + offset_tran)/self.cell_size,           # 边界框中心y坐标加上对应grid cell的偏移量
            tf.square(predict_boxes[..., 2]),                               # 模型预测的是边界框的高和宽的平方根，而不是直接预测边界框的宽和高
            tf.square(predict_boxes[..., 3])
        ],axis=-1)
        # 计算预测边界框与ground truth的交并比, 形状(batch_size, 7, 7, 2)
        iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)
        # NMS 非极大值抑制，交并比更大的负责预测, 形状(batch_size, 7, 7, 2)
        # tf.reduce_max 按输入张量的第axis轴（从0开始）计算极值，keepdims:是否保持原有的维度
        object_mask = tf.reduce_max(iou_predict_truth, axis=3, keepdims=True)
        # 预测边界框中包含的对象的边界框
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response
        # 不包含的对象的边界框, 形状(batch_size, 7, 7, 2)
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        boxes_tran = tf.stack([
            boxes[..., 0] * self.cell_size - offset,
            boxes[..., 1] * self.cell_size - offset_tran,
            tf.sqrt(boxes[..., 2]),
            tf.sqrt(boxes[..., 3])
        ], axis=-1)

        print("response shape: ", response.shape, " boxes shape:", boxes.shape, " predict boxes tran shape: ",
              predict_boxes_tran.shape, " iou predict truth shape: ", iou_predict_truth.shape, " object mask shape: ",
              object_mask.shape, " noobject mask shape: ", noobject_mask.shape, " boxes trans shape: ",
              boxes_tran.shape)

        # ①
        # 类别损失，均方和误差
        # 只有有目标的bbox才计算类别损失
        class_dalta = response * (predict_classes - classes)
        class_loss = tf.reduce_mean(tf.square(class_dalta), axis=[1,2,3], name="class_loss") * self.class_scale
        # ②
        # 包含目标的置信度损失，平方和误差
        object_delta = (predict_confidences - iou_predict_truth) * object_mask
        object_loss = tf.reduce_mean(tf.square(object_delta), axis=[1,2,3], name="object_loss") * self.object_scale
        # ③
        # 不包含目标的置信度损失
        noobject_delta = noobject_mask * predict_confidences
        noobject_loss = tf.reduce_mean(tf.square(noobject_delta), axis=[1,2,3], name="noobject_loss") * self.noobject_scale
        # ④
        # 边界框损失，包含中心坐标以及宽高损失
        # 只计算包含目标的边界框损失
        coord_mask = tf.expand_dims(object_mask, axis=4)            # 在最后扩展一个维度，形状(batch_size, 7, 7, 2, 1)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)     #
        coord_lose = tf.reduce_mean(tf.square(boxes_delta), axis=[1, 2, 3, 4], name="coord_lose") * self.coord_scale

        class_loss_mean = tf.reduce_mean(class_loss)
        object_loss_mean = tf.reduce_mean(object_loss)
        noobject_loss_mean = tf.reduce_mean(noobject_loss)
        coord_lose_mean = tf.reduce_mean(coord_lose)
        total_lose = class_loss + object_loss + noobject_loss + coord_lose

        print("class delta shape: ", class_dalta.shape)
        print("class_loss", class_loss_mean, " object_loss", object_loss_mean, " noobject_loss", noobject_loss_mean,
              " coord_loss", coord_lose_mean)

        tf.summary.scalar("class_loss", class_loss_mean)
        tf.summary.scalar("object_loss", object_loss_mean)
        tf.summary.scalar("noobject_loss", noobject_loss_mean)
        tf.summary.scalar("coord_loss", coord_lose_mean)

        tf.summary.histogram("boxes_delta_x", boxes_delta[..., 0])
        tf.summary.histogram("boxes_delta_y", boxes_delta[..., 1])
        tf.summary.histogram("boxes_delta_w", boxes_delta[..., 2])
        tf.summary.histogram("boxes_delta_h", boxes_delta[..., 3])
        tf.summary.histogram("iou", iou_predict_truth)

        return total_lose

