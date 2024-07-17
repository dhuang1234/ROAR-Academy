from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

'''
For the above problem, if we use ([1, 1], [-1, -1]) to identify the two blue 
clusters within the first group, and ([1, -1], [-1, 1]) to identify the two red 
clusters within the second group, then it is possible to implement a one-layer 
perceptron network with just two perceptrons to train on the output vector 
[out_1, out_2], and there is no additional hidden layers. Please implement a 
feasible solution to the above setup of the XOR classi!ication problem. 
'''

# Create data
linearSeparableFlag = False
x_bias = 6

def toy_2D_samples(x_bias ,linearSeparableFlag):
    if linearSeparableFlag:
        samples1 = np.random.multivariate_normal([5+x_bias, 0], [[1, 0],[0, 1]], 100)
        samples2 = np.random.multivariate_normal([-5+x_bias, 0], [[1, 0],[0, 1]], 100)

        samples = np.concatenate((samples1, samples2 ), axis =0)
    
        # Plot the data
        plt.plot(samples1[:, 0], samples1[:, 1], 'bo')
        plt.plot(samples2[:, 0], samples2[:, 1], 'rx')
        plt.show()

    else:
        samples1 = np.random.multivariate_normal([5+x_bias, 5], [[1, 0],[0, 1]], 400) # change back to 200
        samples2 = np.random.multivariate_normal([-5+x_bias, -5], [[1, 0],[0, 1]], 400)
        samples3 = np.random.multivariate_normal([-5+x_bias, 5], [[1, 0],[0, 1]], 400)
        samples4 = np.random.multivariate_normal([5+x_bias, -5], [[1, 0],[0, 1]], 400)

        samples = np.concatenate((samples1, samples2, samples3, samples4), axis =0)
    
        # Plot the data
        plt.plot(samples1[:, 0], samples1[:, 1], 'bo')
        plt.plot(samples2[:, 0], samples2[:, 1], 'bo')
        plt.plot(samples3[:, 0], samples3[:, 1], 'rx')
        plt.plot(samples4[:, 0], samples4[:, 1], 'rx')
        plt.show()

    label1 = np.array([[1, 1]])
    label11 = np.array([[-1, -1]]) # blues
    label2 = np.array([[-1, 1]]) # reds
    label22 = np.array([[1, -1]])

    labels1 = np.repeat(label1, 400, axis = 0)
    labels2 = np.repeat(label2, 400, axis = 0)
    labels11 = np.repeat(label11, 400, axis = 0)
    labels22 = np.repeat(label22, 400, axis = 0)

    labels = np.concatenate((labels1, labels11, labels2,labels22), axis =0)
    return samples, labels

samples, labels = toy_2D_samples(x_bias ,linearSeparableFlag)

# Split training and testing set
randomOrder = np.random.permutation(1600)
trainingX = samples[randomOrder[0:800], :]
trainingY = labels[randomOrder[0:800], :]
testingX = samples[randomOrder[800:1600], :]
testingY = labels[randomOrder[800:1600], :]

#define a matrix that has 0 for blue, 1 for red
ans = np.zeros((800,1))
for i in range(800):
    if (testingY[i,0]+testingY[i,1]==0):
        ans[i] = 1
    else:
        ans[i] = 0

model = Sequential()
model.add(Dense(2, activation='sigmoid', use_bias=True))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['binary_accuracy'])
model.fit(trainingX, trainingY, epochs=1000, batch_size=10, verbose=1, validation_split=0.2) # old was 0.2 val split
#old batchsize was 10

def showgraph():
    weights, biases = model.layers[0].get_weights()
    w1, w2 = weights[0,:]
    w3, w4 = weights[1,:]
    plt.xlim(-5, 15)
    plt.ylim(-8, 8)
    plt.autoscale(False)

    x_vals = np.linspace(-5, 15, 100)
    y_vals = np.linspace(-8, 8, 100)
    y_perceptron1 = -(w1 / w2) * x_vals - (biases[0] / w2)
    y_perceptron2 = -(w3 / w4) * y_vals - (biases[1] / w4)

    # Plotting the lines
    plt.plot(x_vals, y_perceptron1, label='Perceptron 1', color='blue')
    plt.plot(y_vals, y_perceptron2, label='Perceptron 2', color='red')
    plt.title('Decision Boundaries of Perceptron Network')
    plt.xlabel('X1')
    plt.ylabel('X2')

    # print out weight info
    print(f"P1 weights are {w1} and {w2}, bias is {biases[0]}")
    print(f"first line is y={-w1/w2}*x-{biases[0]/w2}")
    print(f"P1 weights are {w3} and {w4}, bias is {biases[1]}")
    print(f"second line is y={-w3/w4}*x-{biases[1]/w4}")

# score = model.evaluate(testingX, testingY, verbose=0)
score = 0
weights,bias= model.layers[0].get_weights()

#just to see the weights and biases
print(weights)
print(bias)

for i in range(800):
    predict_x=model.predict(np.array([testingX[i,:]])) 
    print("datapoint: ",testingX[i,:])
    print("prediction: " , predict_x)

    if predict_x[0,0].round() == predict_x[0,1].round():
        estimate = 0
    else: 
        estimate = 1
    

    # estimate=np.argmax(predict_x,axis=1) # 0 is blue, 1 is red

    if ans[i] == estimate:
        score = score  + 1

    if estimate == 0:
        plt.plot(testingX[i, 0], testingX[i, 1], 'bo')
    else: 
        plt.plot(testingX[i, 0], testingX[i, 1], 'rx')

showgraph()
print('Test accuracy:', score/800)
plt.show()
