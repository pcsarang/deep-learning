import tensorflow as tf

mat1 = tf.constant([[3., 3.]])
mat2 = tf.constant([[2.], [2.]])
mat_mul = tf.matmul(mat1, mat2)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    result = sess.run(mat_mul)
    print(result)
    print(result.shape)
