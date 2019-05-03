import tensorflow as tf

st = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1.0, 1.0], dense_shape=[3, 4])

dense = tf.sparse_tensor_to_dense(st)

c = tf.constant([[1.0, 2.0, 3.0, 4.0]], dtype=tf.float32)
print(c.shape)
tp = tf.transpose(c)
print(tp.shape)
dense_matmul = tf.sparse_tensor_dense_matmul(st, tp)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    result = sess.run(dense_matmul)
    print("dense \n", sess.run(dense))
    print("transpose \n", sess.run(tp))
    print("result \n", result)


