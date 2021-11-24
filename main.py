import json
import tensorflow as tf

with open("./BeatFactor-dataset-train.json", "r") as dataset_full:
  dataset = json.load(dataset_full)

xs = tf.zeros([0, 2, 5000], dtype=tf.dtypes.int32)

for data in dataset["xs"]:
  padded_notes = tf.pad(tf.constant(data[0]), [[0, 5000 - len(data[0])]])
  padded_times = tf.pad(tf.constant(data[1]), [[0, 5000 - len(data[1])]])
  xs = tf.concat([xs, [[padded_notes, padded_times]]], 0)

ys = tf.constant(dataset["ys"])

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, embedding_dim, num_heads=8):
    super(MultiHeadAttention, self).__init__()
    self.embedding_dim = embedding_dim
    self.num_heads = num_heads

    assert embedding_dim % self.num_heads == 0

    self.projection_dim = embedding_dim // num_heads
    self.query_dense = tf.keras.layers.Dense(embedding_dim)
    self.key_dense = tf.keras.layers.Dense(embedding_dim)
    self.value_dense = tf.keras.layers.Dense(embedding_dim)
    self.dense = tf.keras.layers.Dense(embedding_dim)

  def scaled_dot_product_attention(self, query, key, value):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output, attention_weights

  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, inputs):
    # x.shape = [batch_size, seq_len, embedding_dim]
    batch_size = tf.shape(inputs)[0]

    # (batch_size, seq_len, embedding_dim)
    query = self.query_dense(inputs)
    key = self.key_dense(inputs)
    value = self.value_dense(inputs)

    # (batch_size, num_heads, seq_len, projection_dim)
    query = self.split_heads(query, batch_size)  
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    scaled_attention, _ = self.scaled_dot_product_attention(query, key, value)
    # (batch_size, seq_len, num_heads, projection_dim)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  

    # (batch_size, seq_len, embedding_dim)
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))
    outputs = self.dense(concat_attention)
    return outputs

class TransformerBlock(tf.keras.layers.Layer):
  def __init__(self, embedding_dim, num_heads, dff, rate=0.1):
    super(TransformerBlock, self).__init__()
    self.att = MultiHeadAttention(embedding_dim, num_heads)
    self.ffn = tf.keras.Sequential(
        [tf.keras.layers.Dense(dff, activation="relu"),
          tf.keras.layers.Dense(embedding_dim),]
    )
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, inputs, training):
    attn_output = self.att(inputs) # 첫번째 서브층 : 멀티 헤드 어텐션
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(inputs + attn_output) # Add & Norm
    ffn_output = self.ffn(out1) # 두번째 서브층 : 포지션 와이즈 피드 포워드 신경망
    ffn_output = self.dropout2(ffn_output, training=training)
    return self.layernorm2(out1 + ffn_output) # Add & Norm

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
  def __init__(self, max_len, vocab_size, embedding_dim):
    super(TokenAndPositionEmbedding, self).__init__()
    self.token_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.pos_emb = tf.keras.layers.Embedding(max_len, embedding_dim)

  def call(self, x):
    notes = x[:, 0:1, :]
    times = x[:, 1:2, :]
    notes = tf.reshape(notes, [-1, x.shape[-1]])
    times = tf.reshape(times, [-1, x.shape[-1]])
    times = self.pos_emb(times)
    notes = self.token_emb(notes)
    return notes + times

vocab_size = 217  # 빈도수 상위 2만개의 단어만 사용
max_len = 10000  # 문장의 최대 길이

embedding_dim = 16
num_heads = 2
dff = 16

inputs = tf.keras.layers.Input(shape=(2, 5000))
embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embedding_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embedding_dim, num_heads, dff)
x = transformer_block(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(20, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile("adam", "mean_squared_error")
history = model.fit(xs, ys, batch_size=5, epochs=10)
model.save("./BeatFactor-model")