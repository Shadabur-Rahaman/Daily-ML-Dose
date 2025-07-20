# Learning Rate Scheduler with Keras
from keras.callbacks import LearningRateScheduler
import math

def lr_schedule(epoch):
    return 0.01 * math.exp(-0.1 * epoch)

# Use this callback in model.fit
lr_scheduler = LearningRateScheduler(lr_schedule)

# Usage (add to callbacks list):
# model.fit(..., callbacks=[lr_scheduler])
