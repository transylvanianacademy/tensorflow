import tensorflow as tf

sess = tf.InteractiveSession()


a = [[31, 23,  4, 24, 27, 34],
[18,  3, 25,  0,  6, 35],
[28, 14, 33, 22, 20,  8],
[13, 30, 21, 19,  7,  9],
[16,  1, 26, 32,  2, 29],
[17, 12,  5, 11, 10, 15]]

b = tf.argmax(a, 1).eval()

print(list(b))