import cv2 as cv
import  numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model1 = tf.keras.models.Sequential()
model1.add(tf.keras.layers.Flatten(input_shape=(28,28))) # 28,28 is the number of pixels the training data should be of
model1.add(tf.keras.layers.Dense(units = 128, activation = tf.nn.relu))
model1.add(tf.keras.layers.Dense(units = 128, activation = tf.nn.relu))
model1.add(tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax))

model1.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

model1.fit(x_train, y_train, epochs=6) #epochs is the number of time it will train itself with the same data

loss, accuracy= model1.evaluate(x_test, y_test)
print('accuracy: ',accuracy)
print('loss: ', loss)

model1.save('digits.model1')

for x in range(1,10):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model1.predict(img)
    print(f'The result is probably: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
