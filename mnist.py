from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#x is a placeholder, it is an input
x = tf.placeholder("float", [None, 784])


#these are variables, they live outside of python
#usually we use variables for the modeling stuff
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#implements the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#y_ holds the correct answers, its an input too
y_ = tf.placeholder("float", [None, 10])

#cross-entropy measures how bad our current prediction is
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

#tensorflow function to do the training using grad descent and backprop
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#tensorflow initialize the variables
init = tf.initialize_all_variables()

#launch the model and initialize
sess = tf.Session()
sess.run(init)

#training!
for i in range(1000):
    #we assign the batches out of the training data
    batch_xs, batch_ys = mnist.train.next_batch(100)
    
    #train and assign xs and ys from batch data, the feed_dict feeds the inputs
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#evaluate model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

#take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#print it!
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

