# example of loading the keras facenet model
# from keras.models import load_model
from keras.models import load_model
import tensorflow as tf
# load the model
model =tf.keras.models.load_model('facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)