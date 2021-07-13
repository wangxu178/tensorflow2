"""
LetNet-5 实战
"""
import tensorboard
import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
import datetime

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)


# 加载手写数据集文件
def preprocess(x, y):
    """
    预处理函数
    """
    # [b, 28, 28], [b]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


current_time = datetime.datetime.now().strftime(('%Y%m%d-%H%M%S'))
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)
(x, y), (x_test, y_test) = datasets.mnist.load_data()  # 加载手写数据集数据
batchsz = 1000
train_db = tf.data.Dataset.from_tensor_slices((x, y))  # 转化为Dataset对象
train_db = train_db.shuffle(100000)  # 随机打散
train_db = train_db.batch(batchsz)  # 批训练
train_db = train_db.map(preprocess)  # 数据预处理
train_db = train_db.repeat(30)  # 复制30份数据
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(batchsz).map(preprocess)
# 通过Sequnentia容器创建LeNet-5
network = Sequential([
    layers.Conv2D(6, kernel_size=3, strides=1),  # 第一个卷积核，6个3X3的卷积核，
    layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
    layers.ReLU(),  # 激活函数
    layers.Conv2D(16, kernel_size=3, strides=1),  # 第二个卷积核，16个3X3的卷积核，
    layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
    layers.ReLU(),  # 激活函数
    layers.Flatten(),  # 打平层，方便全连接层处理
    layers.Dense(120, activation='relu'),  # 全连接层，120个结点
    layers.Dense(84, activation='relu'),  # 全连接层，84个结点
    layers.Dense(10, activation='relu')  # 全连接层，10个结点
])
# build 一次网络模型，给输入X的形状
network.build(input_shape=(1000, 28, 28, 1))
# 统计网络信息
print(network.summary)
# 创建损失函数的类，在实际计算时直接调用类实例
criteon = losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.RMSprop(0.01)
# 训练20个epoch
for step, (x, y) in enumerate(train_db):
    with tf.GradientTape() as tape:
        # 插入通道维度 =》[b,28,28,1]
        x = tf.expand_dims(x, axis=3)
        # 前向计算，获取10类别的概率分布 [b,784]=>[b,10]
        out = network(x)
        # 计算交叉熵损失函数，标量
        loss = criteon(y, out)
    # 自动计算梯度
    grads = tape.gradient(loss, network.trainable_variables)
    # 自动更新参数
    optimizer.apply_gradients(zip(grads, network.trainable_variables))
    if step % 100 == 0:
        with summary_writer.as_default():
            tf.summary.scalar('loss', float(loss), step=step)
    # 计算准确度
    if step % 100 == 0:
        correct, total = 0, 0
        for x, y in test_db:
            # 插入通道维度 =>[b,28,28,1]
            x = tf.expand_dims(x, axis=3)
            # 前向计算，获得10类别的预测分布 [b,784] => [b,10]
            out = network(x)
            # 真实的流程时先经过softmax,再argmax
            # 但是由于softmax不改变元素的大小相对关系，故省去
            pred = tf.argmax(out, axis=-1)
            y = tf.cast(y, tf.int64)
            y = tf.argmax(y, axis=-1)
            # 统计预测正确的数量
            correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
            # 统计预测样本的总数量
            total += x.shape[0]
        with summary_writer.as_default():
            tf.summary.scalar('acc', float(correct / total), step=step)

tf.saved_model.save(network, 'model-savedmodel')
