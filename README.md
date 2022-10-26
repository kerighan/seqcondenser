Keras-Condenser
===============

Condenser is a tensorflow layer that summarizes a variable size sequence into a fixed size vector that is independent of the sequence length. Preliminary results show that it outperforms common pooling methods (average, max), embedding of the [CLS] token and the usual weighted-attention mechanism.

To get an overview of the principle behind Condenser, [read this blog entry](https://medium.com/@mchenebaux/a-novel-text-classification-pipeline-using-self-attention-and-the-condenser-layer-9d1fddb0c2c4).

To install package:
```
pip install keras-condenser
```

How to use
----------

Below is a working example using [convectors](https://github.com/kerighan/convectors) and [keras_self_attention](https://github.com/CyberZHG/keras-self-attention):

```python
from convectors.layers import Lemmatize, Sequence, Tokenize
from keras_self_attention import SeqSelfAttention
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.models import Model

from condenser import Condenser

MAX_FEATURES = 100000
EMBEDDING_DIM = 300
MAXLEN = 600

# get training data
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# use convectors as a preprocessing pipeline
nlp = Tokenize()
nlp += Sequence(maxlen=MAXLEN, max_features=MAX_FEATURES)

# process train data
X_train = nlp(newsgroups_train.data)
y_train = newsgroups_train.target
# process test data
X_test = nlp(newsgroups_test.data)
y_test = newsgroups_test.target

# get number of features
n_features = nlp["Sequence"].n_features + 1

# build model
inp = Input(shape=(MAXLEN,))
x = Embedding(n_features, EMBEDDING_DIM, mask_zero=True)(inp)
x = SeqSelfAttention(units=64,
                     attention_width=10,
                     attention_activation='tanh',
                     kernel_regularizer=regularizers.l2(1e-5),
                     attention_regularizer_weight=1e-4,
                     attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(x)
x = SeqSelfAttention(units=64,
                     attention_width=10,
                     attention_activation='tanh',
                     kernel_regularizer=regularizers.l2(1e-5),
                     attention_regularizer_weight=1e-4,
                     attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(x)
x = Condenser(n_sample_points=15, reducer_dim=96)(x)
x = Dense(48, activation="tanh")(x)
out = Dense(20, activation="softmax")(x)

# create and fit model
model = Model(inp, out)
model.compile("nadam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train,
          batch_size=20, epochs=3,
          validation_data=(X_test, y_test),
          shuffle=True)
# >>> val_accuracy=0.8716
```