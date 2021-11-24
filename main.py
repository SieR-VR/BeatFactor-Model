import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

with open("./dataset-test.json", "rt", encoding="UTF8") as dataset_full:
  dataset = json.load(dataset_full)

xs = tf.zeros([0, 5000], dtype=tf.dtypes.int32)

for data in dataset["xs"]:
  xs = tf.concat([xs, [tf.pad(tf.constant(data), [[0, 5000 - len(data)]])]], 0)
ys = dataset["names"]

loaded = tf.keras.models.load_model("./model")

for i in range(100):
  print(loaded.predict(xs[i:i+1])[0][0])
  print(ys[i])