import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plot
import time

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

train_X = train_X.astype(np.float32) / 255
train_Y = np.array(train_Y)
test_X = test_X.astype(np.float32) / 255
test_Y = np.array(test_Y)

train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    for i in range(y.shape[0]):
        one_hot[i, y[i]] = 1
    return one_hot

train_Y = one_hot_encode(train_Y, 10)
test_Y = one_hot_encode(test_Y, 10)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def compute_loss(X, y, W, b):
    m = X.shape[0]
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    loss = -np.sum(y * np.log(probs + 1e-9)) / m
    return loss

def compute_gradient(X, y, W, b):
    m = X.shape[0]
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    grad_W = np.dot(X.T, (probs - y)) / m
    grad_b = np.sum(probs - y,axis = 0) / m
    return grad_W, grad_b

def compute_hessian(X, y, W, b, regularization_param=1e-5):
    m, n = X.shape
    k = y.shape[1]
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    H = np.zeros((n * k, n * k))
    for i in range(100):
        print(i)
        p = probs[i].reshape(-1, 1)
        D = np.diagflat(p) - np.dot(p, p.T)
        X_i = X[i].reshape(-1, 1)
        H += np.kron(D, np.dot(X_i, X_i.T))
    H /= m
    H += regularization_param * np.eye(n * k)
    return H

def line_search(X, y, W, b, grad_W, grad_b, direction_W, direction_b, alpha=0.1, beta=0.8):
    t = 1.0
    initial_loss = compute_loss(X, y, W, b)

    while True:
        new_W = W - t * direction_W
        new_b = b - t * direction_b
        new_loss = compute_loss(X, y, new_W, new_b)

        if new_loss <= initial_loss - alpha * t * (np.sum(grad_W * direction_W) + np.sum(grad_b * direction_b)):
            break
        t *= beta
    return t

def newton(X, y, W, b, epochs=600):
    m, n = X.shape
    k = y.shape[1]
    for i in range(epochs):
        #print(i)
        grad_W, grad_b = compute_gradient(X, y, W, b)
        H = compute_hessian(X, y, W, b)
        H_inv = np.linalg.inv(H)

        grad_W_flat = grad_W.flatten()
        W_flat = W.flatten()

        direction_W_flat = H_inv.dot(grad_W_flat)
        direction_W = direction_W_flat.reshape(W.shape)

        step_size = line_search(X, y, W, b, grad_W, grad_b, direction_W, grad_b)

        W_flat -= step_size * direction_W_flat

        W = W_flat.reshape(W.shape)
        b -= step_size * grad_b

        loss = compute_loss(X, y, W, b)
        print(f"Iteration {i + 1}, Loss: {loss}")
    return W, b

#10000 epochs, 91.07% accuracy | 2000 epochs, 88.7% accuracy
def aor_hb_update(X, y, W, b, W_prev, b_prev, alpha, beta, gamma):
    m = X.shape[0]
    logits = np.dot(X, W) + b
    y_pred = softmax(logits)

    grad_W, grad_b = compute_gradient(X, y, W, b)

    W_new = W + beta * (W - W_prev) - alpha * grad_W
    b_new = b + beta * (b - b_prev) - alpha * grad_b

    W_new += gamma * (W_new - W)
    b_new += gamma * (b_new - b)

    return W_new, b_new

def train_aor_hb(X, y, W, b, epochs=2000, alpha=0.01, beta=0.9, gamma=0.1):
    W_prev = np.copy(W)
    b_prev = np.copy(b)
    losses = []
    elapsed_epochs = []

    for epoch in range(epochs):
        W, b = aor_hb_update(X, y, W, b, W_prev, b_prev, alpha, beta, gamma)
        W_prev, b_prev = W, b

        loss = compute_loss(X, y, W, b)
        losses.append(loss)
        elapsed_epochs.append(epoch)

        if epoch % 10 == 0:
            test_logits = np.dot(test_X, W) + b
            pred_val = softmax(test_logits)
            test_loss = compute_loss(test_X, test_Y, W, b)
            print(f'Epoch {epoch}, Loss: {loss}, Validation Loss: {test_loss}')

    show_error_graph(elapsed_epochs, losses, "AOR-HB")
    return W, b, losses

#2000 epochs, 89.97% accuracy
def aor_hb_update2(X, y, W, b, W_prev, b_prev, grad_W_prev, grad_b_prev, gamma, beta, step_size):
    m = X.shape[0]
    logits = np.dot(X, W) + b
    y_pred = softmax(logits)

    grad_W, grad_b = compute_gradient(X, y, W, b)

    W_new = W + beta * (W - W_prev) - gamma * (2 * grad_W - grad_W_prev)
    b_new = b + beta * (b - b_prev) - gamma * (2 * grad_b - grad_b_prev)

    #W_new += step_size * (W_new - W)
    #b_new += step_size * (b_new - b)

    grad_W_prev = grad_W
    grad_b_prev = grad_b

    return W_new, b_new

def train_aor_hb2(X, y, W, b, epochs=1000, L=1, mu=0.01, step_size=0.1):
    gamma = 1/(np.sqrt(mu)+np.sqrt(L))**2
    beta = L / (np.sqrt(mu) + np.sqrt(L))**2
    W_prev = np.copy(W)
    b_prev = np.copy(b)
    losses = []
    elapsed_epochs = []
    start = time.time()

    for epoch in range(epochs):
        W, b = aor_hb_update2(X, y, W, b, W_prev, b_prev, 0, 0, gamma, beta, step_size)
        W_prev, b_prev = W, b

        loss = compute_loss(X, y, W, b)
        losses.append(loss)
        elapsed_epochs.append(epoch)

        if epoch % 30 == 0:
            #test_logits = np.dot(test_X, W) + b
            #pred_val = softmax(test_logits)
            #test_loss = compute_loss(test_X, test_Y, W, b)
            #print(f'Epoch {epoch}, Loss: {loss}, Validation Loss: {test_loss}')
            print(f'Epoch {epoch}, Loss: {loss}')

    end = time.time()
    print('Elapsed Time: ', end - start, 's')
    show_error_graph(elapsed_epochs, losses, "AOR-HB")
    return W, b, losses

def show_error_graph(epoch, error, title):
    plot.figure(figsize=(9,4))
    plot.plot(epoch, error, "m-")
    plot.xlabel("Epoch Number")
    plot.ylabel("Erorr")
    plot.title(title)
    plot.show()

num_classes = train_Y.shape[1]
num_features = train_X.shape[1]

W = np.zeros((num_features, num_classes))
b = np.zeros(num_classes)

#W, b = newton(train_X, train_Y, W, b)
W,b, losses = train_aor_hb2(train_X, train_Y, W, b)
#W, b, losses = train_aor_hb2(train_X, train_Y, W, b)

def predict(X, W, b):
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    return np.argmax(probs, axis=1)

train_Y_predict = predict(train_X, W, b)
test_Y_predict = predict(test_X, W, b)

train_acc = np.mean(np.argmax(train_Y, axis=1) == train_Y_predict)
test_acc = np.mean(np.argmax(test_Y, axis=1) == test_Y_predict)

print(f"Train Accuracy: {train_acc}")
print(f"Test Accuracy: {test_acc}")
