import tensorflow as tf

s = tf.constant(483)
v = tf.constant([1.1, 2.2, 3.3])
m = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
t = tf.constant([[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]])

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(tf.rank(s)))
    print(sess.run(tf.rank(v)))
    print(sess.run(tf.rank(m)))
    print(sess.run(tf.rank(t)))

sess.close()

