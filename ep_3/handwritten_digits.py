import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# locatia unde vom descarca datset-ul
DATA_DIR = ''
# numarul de iteratii efectuate pentru invatare
NUM_STEPS = 1000
# cantitatea de imagini folosita pentru fiecare ciclu de invatare
MINIBATCH_SIZE = 100

'''
descarcam MNIST dataset, in locatia specificata
pentru etichetele imaginilor folosin 'one_hot' encoding

'one_hot' inseamna ca eticheta pentru imaginile care contin cifra 1 vor arata [0,1,0,0,0,0,0,0,0,0]
pentru cifra 9 eticheta asociata imaginii va fi [0,0,0,0,0,0,0,0,0,1]
'''
mnist_data = input_data.read_data_sets(DATA_DIR, one_hot=True)

'''
'x' este namespace-ul folosit ca place holder pentru imaginile noastre ce vor fi incarcate la training

folosim None pentru ca nu stim cate imagini vom incarca

imaginile din MNIST dataset au o rezolutie de 28x28 pixeli
28x28 = 784 reprezinta numarul de pixeli din imaginile
'''
x = tf.placeholder(tf.float32, [None, 784])

# acum vom defini o variabila folosita pentru stocarea "weights"
W = tf.Variable(tf.zeros([784, 10]))

# valorile pe care le are poza incarcata
y_true = tf.placeholder(tf.float32, [None, 10])

# matrice cu valorile prezise de model
y_pred = tf.matmul(x, W)

'''
aceasta formula ne ajuta sa determinam 'distanta' dintre ce am prezis si ce stim ca este adevarat
Ex: daca stim ca imaginea este un 5, y_true o sa fie: [0,0,0,0,0, 1 ,0,0,0,0] iar
y_pred o sa fie o prezicere: [0.1,0.2,0.1,0.3,0.2, 0.9 ,0.1,0.3,0.2,0.1]
'''
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

'''
acuma definim marimea pasilor pe care sa ii facem, in cautarea formulei optime
de aproximare raspunsului adevarat
'''
gradient_descent_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

'''
'argmax' intoarce pozitia dintr-o lista a celui mai mare argument numeric 
[[31, 23,  4, 24, 27, 34],
[18,  3, 25,  0,  6, 35],
[28, 14, 33, 22, 20,  8],
[13, 30, 21, 19,  7,  9],
[16,  1, 26, 32,  2, 29],
[17, 12,  5, 11, 10, 15]]

array([5, 5, 2, 1, 3, 0])

'tf.equal' va intoarce o matrice care compara cele 2 rezultate calculate cu 'argmax'
'''
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true,1))

'''
folosim 'tf.cast' pentru a transforma noua matrice in datatype-ul ales de noi tf.float32

'tf.reduce_mean' calculeaza media de acuratete
'''
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(NUM_STEPS):
        # incarcam urmatorul set de imagini impreuna cu etichetele acestora
        batch_xs, batch_ys = mnist_data.train.next_batch(MINIBATCH_SIZE)

        # incepem procesul de invatare
        sess.run(gradient_descent_step, feed_dict={x:batch_xs, y_true:batch_ys})

    writer = tf.summary.FileWriter('C:\\temp\\tensor_log', sess.graph)

    # calculam acuratetea modelului nostru pe un set de imagini noi
    ans = sess.run(accuracy, feed_dict={x:mnist_data.test.images, y_true:mnist_data.test.labels})

    # printam rezultatul
    print("Accuracy: {:.4}%".format(ans*100))


# how to see the graph
'''
# create this folder 'C:\temp\tensor_log'
tensorboard --logdir==training:C:\temp\tensor_log --host=127.0.0.1
'''