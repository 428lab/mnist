# MNISTによるMLP構築の写経
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# データを用意する
# x ... 入力データ
# y ... 出力データ
# train ... 学習用データ
# test ... 検証用データ
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 入力データをベクトルに変換する
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 出力データをone_hot表現に変形する
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# モデルを構築する
model = Sequential()

# 入力層
# ニューロン数 500, 活性化関数 ReLU
model.add(Dense(units=500, input_shape=(784,))) # Dense ... 全結合層
model.add(Activation('relu')) # Activation ... 活性化関数
# 隠れ層1
# ニューロン数 250, 活性化関数 ReLU
model.add(Dense(units=250))
model.add(Activation('relu'))
# 隠れ層2
# ニューロン数 100, 活性化関数 ReLU
model.add(Dense(units=100))
model.add(Activation('relu'))
# 出力層
# ニューロン数 10, 活性化関数 softmax関数
model.add(Dense(units=10))
model.add(Activation('softmax')) 

# loss ... 損失関数
# optimizer ... 最適化アルゴリズム
# metrics ... 評価関数
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# モデルに学習させる
model.fit(x_train, y_train, batch_size=1000, epochs=10, verbose=1, validation_data=(x_test, y_test))

# モデルを評価する
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print('損失:', test_loss)
print('評価:', test_accuracy)
