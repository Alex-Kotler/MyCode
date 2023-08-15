import numpy as np
import matplotlib.pyplot as plt

print("PART 5 IS RUNNING")
# PART 5
class Perceptron:
    def __init__(self, n, m):
        # Initialize weights to small random values in [-0.05, 0.05]
        self.weights = np.random.uniform(-0.05, 0.05, (n + 1, m))

    def __str__(self):
        return f"A perceptron with {self.weights.shape[0] - 1} inputs and {self.weights.shape[1]} outputs."
    
    def test(self, input_vector):
        # Append a bias term (1) to the input_vector
        augmented_input = np.append(input_vector, 1)
        # Perform the dot product and compare with 0
        return np.dot(augmented_input, self.weights) > 0

    def train(self, input_patterns, target_patterns, niter=1000, learning_rate=0.1):
        for _ in range(niter):
            for input_vector, target_vector in zip(input_patterns, target_patterns):
                # Calculate output
                output = self.test(input_vector)
                # Adjust weights using the perceptron learning rule
                error = target_vector - output
                self.weights += learning_rate * np.outer(np.append(input_vector, 1), error)
                #print("Updated weights:\n", self.weights)

# Test the perceptron
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# OR Function
targets_or = np.array([[0], [1], [1], [1]])
perceptron_or = Perceptron(2, 1)
perceptron_or.train(inputs, targets_or)
print("OR Function:")
for i in inputs:
    print(f"For input {i}, output is:", perceptron_or.test(i))

# AND Function
targets_and = np.array([[0], [0], [0], [1]])
perceptron_and = Perceptron(2, 1)
perceptron_and.train(inputs, targets_and)
print("\nAND Function:")
for i in inputs:
    print(f"For input {i}, output is:", perceptron_and.test(i))


print("PART 6 IS RUNNING")
# PART 6
def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = []
    i = 0
    while i < len(lines):
        identifier = lines[i].strip()
        i += 1

        pixels = []
        for j in range(14):
            row = list(map(float, lines[i].split()))
            pixels.extend(row)
            i += 1

        target = list(map(int, lines[i].split()))
        i += 1

        data.append({
            "id": identifier,
            "pixels": pixels,
            "target": target
        })

    return data

def visualize_example(example):
    pixels = np.array(example['pixels']).reshape(14, 14)
    plt.imshow(pixels, cmap='gray')
    plt.title(example['id'])
    plt.show()

train_data = read_data('digits_train.txt')
test_data = read_data('digits_test.txt')
visualize_example(train_data[0])


print("PART 7 IS RUNNING")
# PART 7
def preprocess_data(data):
    X = []
    y = []
    for entry in data:
        X.append(entry["pixels"])
        y.append(1 if entry["target"][2] == 1 else 0)
    return np.array(X), np.array(y)

def train_perceptron(X, y, epochs=100000, learning_rate=0.01):
    weights = np.zeros(X.shape[1])
    bias = 0
    
    for epoch in range(epochs):
        for i in range(X.shape[0]):
            linear_output = np.dot(X[i], weights) + bias
            prediction = 1 if linear_output >= 0 else 0
            
            update = learning_rate * (y[i] - prediction)
            weights += update * X[i]
            bias += update

        if (epoch + 1) % 100 == 0:
            print(f"{epoch+1}/{epochs} iterations completed")
    
    return weights, bias

def evaluate_perceptron(X, y, weights, bias):
    predictions = 1 * (np.dot(X, weights) + bias >= 0)
    false_positives = sum((predictions == 1) & (y == 0))
    false_negatives = sum((predictions == 0) & (y == 1))
    
    print(f"{false_positives} false positives / {sum(y == 0)} = {false_positives/sum(y==0)*100:.2f}%")
    print(f"{false_negatives} misses / {sum(y == 1)} = {false_negatives/sum(y==1)*100:.2f}%")

X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)
weights, bias = train_perceptron(X_train, y_train)
evaluate_perceptron(X_test, y_test, weights, bias)
