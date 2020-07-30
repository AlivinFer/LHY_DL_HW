import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

def load_data():
    # loading data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 28*28)
    x_test = x_test.reshape(10000, 28*28)
    x_train = x_train.astype('float32')  # 之前为 unit8(8位无符号整数)
    x_test = x_test.astype('float32')
    x_train /= 255  # x_train之前的灰度值最大为255，最小为0，这里将它们进行特征归一化，变成了在0到1之间的小数
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # 将类别标签转换位 one_hot 编码
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)  # 60000x10
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)  # 10000x10

    return (x_train, y_train), (x_test, y_test)


def HDR_model():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_data()
    # 定义模型
    model = Sequential()
    # 定义输入层，全连接网络，输入维度是784，有512个神经元，激活函数是Sigmoid
    model.add(Dense(input_dim=28*28, units=512, activation='relu'))
    # 正则化，避免过拟合（将隐含层的部分权重或输出随机归零）
    model.add(Dropout(0.2))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.2))
    # 定义输出层，有10个神经元，也就是10个输出，激活函数是 Softmax
    model.add(Dense(units=10, activation='softmax'))

    # 输出模型各层的参数状况
    model.summary()

    # 选择损失函数，优化器，评价指标(RMSprop 是一种自适应学习率方法)
    model.compile(optimizer=RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # 训练模型（verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录）
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    HDR_model()
