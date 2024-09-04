# /Users/tteresattian/.pyenv/versions/3.10.12/bin/python -m pip install tensorflow
# /Users/tteresattian/.pyenv/versions/3.10.12/bin/python -c "import tensorflow as tf; print(tf.__version__)"
# /Users/tteresattian/.pyenv/versions/3.10.12/bin/python -m pip install tensorflow-hub
# /Users/tteresattian/.pyenv/versions/3.10.12/bin/python /Users/tteresattian/Desktop/intention_recog_model_JD/.vscode/tf_bert.py

# make a virtual environment: /Users/tteresattian/.pyenv/versions/3.10.12/bin/python -m venv myenv
# activate the virtual environment source myenv/bin/activate
# pip install tensorflow tensorflow-hub




import tensorflow as tf
import tensorflow_hub as hub  # Assuming you're using TensorFlow Hub for BERT and preprocessing layers


def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()
bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))

tf.keras.utils.plot_model(classifier_model)
