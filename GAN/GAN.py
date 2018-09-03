import tensorflow as tf
import numpy as np
import imageio
import itertools
import glob
import os
import matplotlib.pyplot as plt


class GAN():
    def __init__(self, hidden=256, noise=128, input=1296, reuse=True, total_epoch=100, batch_size=100,
                 leraning_rate=0.0002):
        self.root_directory = "C:/Users/jlee1/Desktop/Dataset/celeba_raw"
        self.hidden = hidden
        self.noise = noise
        self.input = input  # 36*36=1296 for celeba (28*28=784 for MNIST)

        self.total_epochs = total_epoch
        self.batch_size = batch_size
        self.learning_rate = leraning_rate
        self.reuse = reuse

        with tf.variable_scope("Generator") as scope:
            self.GW1 = tf.get_variable("GW1", shape=[self.noise, self.hidden],
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.Gb1 = tf.get_variable("Gb1", shape=[self.hidden], initializer=tf.contrib.layers.xavier_initializer())
            self.GW2 = tf.get_variable("GW2", shape=[self.hidden, self.input],
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.Gb2 = tf.get_variable("Gb2", shape=[self.input], initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("Discriminator") as scope:
            self.DW1 = tf.get_variable("DW1", shape=[self.input, self.hidden],
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.Db1 = tf.get_variable("Db1", shape=[self.hidden], initializer=tf.contrib.layers.xavier_initializer())
            self.DW2 = tf.get_variable("DW2", shape=[self.hidden, 1],
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.Db2 = tf.get_variable("Db2", shape=[1], initializer=tf.contrib.layers.xavier_initializer())

        self.X = tf.placeholder(tf.float32, [None, self.input])
        self.Z = tf.placeholder(tf.float32, [None, self.noise])

    def dataset_files(root, SUPPORTED_EXTENSIONS=["png", "jpg", "jpeg"]):
        return list(itertools.chain.from_iterable(
            glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))

    def batch(self, file_list, batch_size=16):
        fileQ = tf.train.string_input_producer(file_list, shuffle=False)
        reader = tf.WholeFileReader()

        filename, data = reader.read(fileQ)
        image = tf.image.decode_jpeg(data, channels=3)

        img = imageio.imread(file_list[0])
        w, h, c = img.shape
        shape = [w, h, 3]

        image.set_shape(shape)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size

        queue = tf.train.shuffle_batch(
            [image], batch_size=batch_size,
            num_threads=4, capacity=capacity,
            min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

        crop_queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
        resized = tf.image.resize_nearest_neighbor(crop_queue, [64, 64])

        return tf.to_float(resized)

    def generator(self, z):
        hidden = tf.nn.relu(tf.matmul(z, self.GW1) + self.Gb1)
        output = tf.nn.sigmoid(tf.matmul(hidden, self.GW2) + self.Gb2)

        return output

    def discriminator(self, input_data):
        hidden = tf.nn.relu(tf.matmul(input_data, self.DW1) + self.Db1)
        output = tf.nn.sigmoid(tf.matmul(hidden, self.DW2) + self.Db2)

        return output

    def random_noise(self):
        return np.random.normal(size=[self.batch_size, self.noise])

    def train(self, input_data):
        generated_fake_image = self.generator(self.Z)

        discriminated_fake_image = self.discriminator(generated_fake_image)
        discriminated_real_image = self.discriminator(self.X)

        # loss function modification
        # log(1-D(z)) is close to 0 at first which hinders learning process
        # So, instead of minimizing log(1-D(z)), we maximize log(D(z))

        g_loss = tf.reduce_mean(tf.log(discriminated_fake_image))
        d_loss = tf.reduce_mean(tf.log(discriminated_real_image) + tf.log(1 - discriminated_fake_image))

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        t_vars = tf.trainable_variables()

        g_vars = [var for var in t_vars if "Generator" in var.name]
        d_vars = [var for var in t_vars if "Discriminator" in var.name]

        g_train = optimizer.minimize(-g_loss, var_list=g_vars)
        d_train = optimizer.minimize(-d_loss, var_list=d_vars)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_x, train_y = self.batch(self.root_directory)
            total_batchs = int(train_x.shape[0] / self.batch_size)

            for epoch in range(self.total_epochs):
                for batch in range(total_batchs):
                    batch_x = train_x[batch * self.batch_size: (batch + 1) * self.batch_size]  # [batch_size , 1296]
                    batch_y = train_y[batch * self.batch_size: (batch + 1) * self.batch_size]  # [batch_size,]
                    noise = self.random_noise(self.batch_size)  # [batch_size, 128]

                    sess.run(g_train, feed_dict={self.Z: noise})
                    sess.run(d_train, feed_dict={self.X: batch_x, self.Z: noise})

                    gl, dl = sess.run([g_loss, d_loss], feed_dict={X: batch_x, Z: noise})

                # check every 20 epoch
                if (epoch + 1) % 20 == 0 or epoch == 1:
                    print("=======Epoch : ", epoch, " =======================================")
                    print("Generator performance: ", gl)
                    print("Discriminator performance : ", dl)

                # check real outcome every 10 epoch

                if epoch == 0 or (epoch + 1) % 10 == 0:
                    sample_noise = self.random_noise(10)

                    generated = sess.run(generated_fake_image, feed_dict={Z: sample_noise})

                    fig, ax = plt.subplots(1, 10, figsize=(10, 1))
                    for i in range(10):
                        ax[i].set_axis_off()
                        ax[i].imshow(np.reshape(generated[i], (28, 28)))

                    plt.savefig('goblin-gan-generated/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
                    plt.close(fig)

            print('Optimization complete!')


if __name__ == "__main__":
    g = GAN()
    g.train(g.root_directory)
