from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
with CustomObjectScope({'tf': tf}):
  model = load_model('./model/nn4.small2.v1.h5')
  print(model)