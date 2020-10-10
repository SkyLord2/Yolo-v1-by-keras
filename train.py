# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
import config as cfg
from YOLOv1_beta_1 import YOLOv1Net
from utils.pascal_voc import pascal_voc

# 需要首先下载数据集

pascal = pascal_voc('train')

def get_train_data_by_batch():
    while 1:
        for i in range(0, len(pascal.gt_labels), 64):
            images, labels = pascal.get()
            yield (images, labels)


def lr_scheduler(epoch):
    lr = 1e-4
    if(epoch <= 75):
        lr = 1e-2
    elif(75 < epoch and epoch <= 105):
        lr = 1e-3
    elif(105 < epoch and epoch <= 135):
        lr = 1e-4
    return lr




if __name__ == '__main__':
    yolo = YOLOv1Net()
    yolo.compile_model()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    modelcheck = keras.callbacks.ModelCheckpoint("weights_{epoch:03d}-{val_loss:.4f}.h5",
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto',
                                    period=1)
    result = yolo.train_generator(get_train_data_by_batch(), steps_per_epoch=len(pascal.gt_labels) // 64,
                                      epochs=135, callbacks=[lr_schedule, modelcheck])
