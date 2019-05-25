import tensorflow as tf
import matplotlib.pyplot as plt
import random

#tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class_cnt = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, class_cnt])

W = tf.Variable(tf.random_normal([784, class_cnt]))
b = tf.Variable(tf.random_normal([class_cnt]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
num_epochs = 15
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)

#### histogram & scalar ########################################
tf.summary.histogram("weights", W)
tf.summary.histogram("bias", b)
tf.summary.histogram("hypothesis", hypothesis)
tf.summary.scalar("loss", cost)
################################################################

global_step = 0

with tf.Session() as sess:\
    sess.run(tf.global_variables_initializer())

    ### Summary & writer ###########################################
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('d:/temp/tensorflowlogs')
    writer.add_graph(sess.graph)
    ################################################################

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

    # Test the model using test sets
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}
        ),
    )

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print(
        "Prediction: ",
        sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}),
    )


