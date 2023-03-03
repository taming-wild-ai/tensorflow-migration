"""Dead simple tutorial for defining and training a small feedforward neural
network (also known as a multilayer perceptron) for regression using TensorFlow 1.X.

Introduces basic TensorFlow concepts including the computational graph,
placeholder variables, and the TensorFlow Session.

Author: Ji-Sung Kim
Contact: hello (at) jisungkim.com
"""
import numpy as np
import tensorflow as tf

"""Summary

TensorFlow is a unique computation framework: it is used for defining
computational graphs and running these graphs. TensorFlow was originally
designed for constructing and manipulating deep neural networks.

TensorFlow is special in that it does not dynamically compute the values
of the outputs of operations. This is in contrast with standard Python which
dynamically computes the outputs of operations. In TensorFlow, we have to
instead  specify a static computational graph using tensors, and then explicitly
run the graph (through a `tf.Session()` object).

```typical Python
x = 5
y = x + 5
print(y)  # prints 10
```

```TensorFlow
import tensorflow as tf

x = tf.constant(5)  # a tensor
y = x + 5  # equivalent to tf.add(x, 5)
print(y)  # doesn't print the actual value because graph has not been run
sess = tf.Session()
print(sess.run(y))  # prints 10
```

In this tutorial, we define a simple feedforward neural network with TensorFlow.

This particular neural network takes in as input a placeholder tensor called
x_placeholder of shape (batch_size, dim_input) and outputs a tensor of
shape (batch_size, dim_output). The batch_size represents the number of
instances in a single batch of samples run through the graph. We often use
placeholders which represent empty tensors through which we can pass in
arbitrary data. Of course, the actual data fed in through the placeholders
must match the shape of the placeholders.

We then compute the mean squared error between the network outputs (estimated
target) and the true target values (which we feed in through the placeholder
`y_placeholder`). We minimize this error to train the neural network; training
involves adjusting the tunable parameters within the neural network model
(here, specifically the weight and bias variables inside the `dense` layers)
using gradient descent.
"""
dim_input = 3  # arbitrarily chosen for this example script
dim_output = 1

# Define the neural network which consists of two dense (fully connected)
# layers (which comprise simple matrix multiplication and addition operations).
# These "layers" are all TensorFlow operations which can be explicitly run.

# The input to the first layer is the input features (given via
# `x_placeholder`).
intermediate_layer = tf.keras.layers.Dense(12, input_shape=[None, dim_input])

# We pass the outputs of the first layer as inputs to the second, final layer
# which outputs the estimated target.
final_layer = tf.keras.layers.Dense(dim_output, input_shape=[None, 12])
estimated_targets = final_layer  # just a different name for clarity

# We define the `loss` (error) function which we minimize to train our neural
# network. The following loss operation is equivalent to calling the helper
# `tf.losses.mean_squared_error(y_placeholder, estimated_targets)` which also
# returns an operation.
@tf.function
def loss(examples, labels):
    return tf.square(tf.subtract(tf.cast(labels, tf.float32), estimated_targets(intermediate_layer(examples))))  # operation

# We use the Adam optimizer which is an object which provides functions
# to optimize (minimize) the loss using a variant of gradient descent.
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_op(examples, labels):
    with tf.GradientTape() as tape:
        return optimizer.minimize(
            loss(examples, labels),
            intermediate_layer.trainable_variables + final_layer.trainable_variables,
            tape)

# Repeatedly train the neural network for `num_epoch` times
num_epoch = 2000
batch_size = 500
for i in range(num_epoch):
  # Define input training data. `x_data` represents the training data features
  # which are 0 or 1; these are the input data to the neural network.
  # `y_data` represents the training data "true" targets; `y_data` is just
  # the outputs of the function y = 5 * sum(x) applied to the data batch.
  # We are trying to learn this function (mapping from x to y) with our
  # neural network. Neural networks are general function estimators.

  # generate random binary np.array with shape (batch_size, 3)
  x_data = np.random.randint(2, size=(batch_size, dim_input))
  # calculate targets from feature array
  y_data = 5 * np.sum(x_data, axis=-1).reshape((-1, 1))
  # reshape to match `y_placeholder` shape which has a last dimension of 1
  y_data = y_data.reshape((-1, 1))

  # Run the training operation defined earlier.
  train_op(x_data, y_data)


"""After we finished training our neural network (NN), we are going to use it
with new test data. "Using" the neural network is just running new values
through the computational graph that the NN represents. Again, we keep in mind
a neural network is just a function which transforms some inputs to outputs."""

# We get new test data, again using the random numpy generation function.
x_data_test = np.random.randint(2, size=(5, dim_input))

y_estimate_test = final_layer(intermediate_layer(x_data_test))

# Examine test data.
print('x_data_test')
print(x_data_test)
print()

# Are the estimates of the target from the NN close to what we expected?
print('y_estimate_test')
print(y_estimate_test)
print()

# We could also measure the error for the test_data but we would have to specify
# the true target values for the test data and then pass it through `y_placeholder`
# in the `feed_dict`. We could run the `loss` operation to compute the
# test error.

# This is left empty as an exercise to the reader.
