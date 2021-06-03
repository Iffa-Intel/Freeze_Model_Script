from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np

def custom_loss(y_true, y_pred):
    loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
    return K.mean(loss)

model = keras.models.load_model('D:/CustomerOpenvino/aryaB/model906_30.h5', custom_objects={'custom_loss': custom_loss})

full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
x=tf.TensorSpec((1, 9, 288, 512), model.inputs[0].dtype))

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")

for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,logdir="./frozen_models",name="simple_frozen_graph.pb",as_text=False)

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,logdir="./frozen_models",name="simple_frozen_graph.pbtxt",as_text=True)
model.summary()
