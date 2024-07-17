from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

'''
1. MULTI-LAYER PERCEPTRON FOR LINEARLY NONSEPARABLE DATA 
In the MLP sample code, we demonstrated by changing the sigmoid 
activation function to ReLU, a two-layer MLP model can effectively classify 
the “XOR” linearly non separable sample points (shown below) with 100% 
accuracy. 
Please propose and train an alternative MLP model, whereby the two class 
labels are assigned as [1, 0] and [0, 1], and the hidden layers will keep the 
sigmoid activation, but you may be able to add more perceptrons per layer 
or add more layers. Please show which new MLP model you may !ind can 
effectively classify the sample points with 100% accuracy
'''

# Create data
linearSeparableFlag = False
x_bias = 6

def toy_2D_samples(x_bias ,linearSeparableFlag):
    label1 = np.array([[1, 0]])
    label2 = np.array([[0, 1]])

    if linearSeparableFlag:
        samples1 = np.random.multivariate_normal([5+x_bias, 0], [[1, 0],[0, 1]], 100)
        samples2 = np.random.multivariate_normal([-5+x_bias, 0], [[1, 0],[0, 1]], 100)

        samples = np.concatenate((samples1, samples2 ), axis =0)
    
        # Plot the data
        plt.plot(samples1[:, 0], samples1[:, 1], 'bo')
        plt.plot(samples2[:, 0], samples2[:, 1], 'rx')
        plt.show()

    else:
        samples1 = np.random.multivariate_normal([5+x_bias, 5], [[1, 0],[0, 1]], 50)
        samples2 = np.random.multivariate_normal([-5+x_bias, -5], [[1, 0],[0, 1]], 50)
        samples3 = np.random.multivariate_normal([-5+x_bias, 5], [[1, 0],[0, 1]], 50)
        samples4 = np.random.multivariate_normal([5+x_bias, -5], [[1, 0],[0, 1]], 50)

        samples = np.concatenate((samples1, samples2, samples3, samples4 ), axis =0)
    
        # Plot the data
        plt.plot(samples1[:, 0], samples1[:, 1], 'bo')
        plt.plot(samples2[:, 0], samples2[:, 1], 'bo')
        plt.plot(samples3[:, 0], samples3[:, 1], 'rx')
        plt.plot(samples4[:, 0], samples4[:, 1], 'rx')
        plt.show()

    label1 = np.array([[1, 0]])
    label2 = np.array([[0, 1]])
    labels1 = np.repeat(label1, 100, axis = 0)
    labels2 = np.repeat(label2, 100, axis = 0)
    labels = np.concatenate((labels1, labels2 ), axis =0)
    return samples, labels

samples, labels = toy_2D_samples(x_bias ,linearSeparableFlag)

# Split training and testing set

randomOrder = np.random.permutation(200)
trainingX = samples[randomOrder[0:100], :]
trainingY = labels[randomOrder[0:100], :]
testingX = samples[randomOrder[100:200], :]
testingY = labels[randomOrder[100:200], :]

model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='sigmoid', use_bias=True))
# model.add(Dense(4, input_shape=(2,), activation='relu', use_bias=True))
model.add(Dense(8, input_shape=(2,), activation='tanh', use_bias=True))
model.add(Dense(2, activation='softmax' ))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['binary_accuracy'])

model.fit(trainingX, trainingY, epochs=500, batch_size=10, verbose=1, validation_split=0.2)

# score = model.evaluate(testingX, testingY, verbose=0)
score = 0
for i in range(100):
    predict_x=model.predict(np.array([testingX[i,:]])) 
    print(predict_x)
    estimate=np.argmax(predict_x,axis=1)

    if testingY[i,estimate] == 1:
        score = score  + 1

    if estimate == 0:
        plt.plot(testingX[i, 0], testingX[i, 1], 'bo')
    else: 
        plt.plot(testingX[i, 0], testingX[i, 1], 'rx')

print('Test accuracy:', score/100)
plt.show()
