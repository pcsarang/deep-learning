import tensorflow as tf

x = tf.constant(10, name = 'x')
y = tf.Variable(x + 5, name = 'y')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(y))