import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameters
num_epochs = 15
batch_size = 100
learning_rate = 0.001

num_classes = 10


# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

with tf.variable_scope('layer1') as scope:
    W1 = tf.Variable(tf.random_normal([784, 512]))
    b1 = tf.Variable(tf.random_normal([512]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    tf.summary.histogram("X", X)
    tf.summary.histogram("weights", W1)
    tf.summary.histogram("bias", b1)
    tf.summary.histogram("layer", L1)

with tf.variable_scope('layer2') as scope:
    W2 = tf.Variable(tf.random_normal([512, 512]))
    b2 = tf.Variable(tf.random_normal([512]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    tf.summary.histogram("weights", W2)
    tf.summary.histogram("bias", b2)
    tf.summary.histogram("layer", L2)

with tf.variable_scope('layer3') as scope:
    W3 = tf.Variable(tf.random_normal([512, 512]))
    b3 = tf.Variable(tf.random_normal([512]))
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

    tf.summary.histogram("weights", W3)
    tf.summary.histogram("bias", b3)
    tf.summary.histogram("layer", L3)

with tf.variable_scope('layer4') as scope:
    W4 = tf.Variable(tf.random_normal([512, 512]))
    b4 = tf.Variable(tf.random_normal([512]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

    tf.summary.histogram("weights", W4)
    tf.summary.histogram("bias", b4)
    tf.summary.histogram("layer", L4)

with tf.variable_scope('layer5') as scope:
    W5 = tf.Variable(tf.random_normal([512, num_classes]))
    b5 = tf.Variable(tf.random_normal([num_classes]))

    tf.summary.histogram("weights", W5)
    tf.summary.histogram("bias", b5)


hypothesis = tf.matmul(L4, W5) + b5
tf.summary.histogram("hypothesis", hypothesis)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

num_iterations = int(mnist.train.num_examples / batch_size)

global_step = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ### Summary & writer ###########################################
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('d:/temp/tensorflowlogs')
    writer.add_graph(sess.graph)
    ################################################################

    # 학습 수행
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}

            ### Summary & optimizer ########################################
            s, t = sess.run([summary, optimizer], feed_dict = feed_dict)
            writer.add_summary(s, global_step=global_step)
            ################################################################

            cost_val = sess.run(cost, feed_dict = feed_dict)
            avg_cost += cost_val / num_iterations

            global_step += 1

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    # 모델의 Accuracy 지표를 이용한 성능 측정
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}
        ),
    )

    # 학습된 모델을 이용한 예측
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print(
        "Prediction: ",
        sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}),
    )

