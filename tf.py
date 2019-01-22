import tensorflow as tf

## Tensorflow Test ##
# a = tf.constant('HI')
# sess = tf.Session()
# print(sess.run(a))

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = a + b
mul = a * b
g = (a * b) + a + b
sess = tf.Session()

print("Add : %i " %sess.run(add, feed_dict={a:2, b:3}))
print("Mul : %i " %sess.run(mul, feed_dict={a:2, b:3}))
print("Graph : %i " %sess.run(g, feed_dict={a:2, b:3}))
