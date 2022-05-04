
# Import necessary packages
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Relation 2x - 1
#xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
#ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Relation 2x -2
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-4.0, -2.0, 0.0, 2.0, 4.0, 6.0], dtype=float)

l0 = Dense(units=1, input_shape=[1])

# Initialize the Sequential model
model = Sequential([l0])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

print(model.summary())
for epoch in range(0,6):

    # Fit model to data
    model.fit(xs, ys, epochs = epoch*100, verbose=0)

    # Expected 18
    print(model.predict([10.0]))

    # Output will be in format [array([[1.9999975]], dtype=float32), array([-1.9999933], dtype=float32)]
    # First value 1.99 resembles 2 and ssecond value resembles -1.99 resembles -2
    # Hence 2x - 2
    print("Weight: ",l0.get_weights())
