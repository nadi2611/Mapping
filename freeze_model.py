import tensorflow as tf
from keras.models import load_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
path = 'final_models/model-0050.h5'
model = load_model(path, compile=False)

full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
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

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph.as_graph_def(),
                  logdir="./frozen_models",
                  name="frozen_graph.pb",
                  as_text=True)