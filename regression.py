import numpy as np

print("PART 1 IS RUNNING")
#PART 1
def least_squares(x, y):
    mean_x=np.average(x)
    mean_y=np.average(y)
    dis_x=x-mean_x*np.ones(len(x))
    dis_y=y-mean_y*np.ones(len(y))
    m = np.dot(dis_x,dis_y) / np.dot(dis_x, dis_x)
    b = mean_y-m*mean_x
    return m, b

def part1():
    data = np.loadtxt("assign1_data.txt", skiprows=1)
    x1 = data[:, 0]
    x2 = data[:, 1]
    y = data[:, 2]
    m1, b1 = least_squares(x1, y)
    m2, b2 = least_squares(x2, y)
    print(f"For y = m*x1 + b: m = {m1:.4f}, b = {b1:.4f}")
    print(f"For y = m*x2 + b: m = {m2:.4f}, b = {b2:.4f}")

part1()

print("PART 2 IS RUNNING")
#PART 2
def part2():
    data = np.loadtxt("assign1_data.txt", skiprows=1)
    x1 = data[:, 0]
    x2 = data[:, 1]
    y = data[:, 2]
    A = np.vstack([x1, x2, np.ones(len(x1))]).T
    w1, w2, b = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f"w1 = {w1}")
    print(f"w2 = {w2}")
    print(f"b = {b}")

part2()

print("PART 3 IS RUNNING")
#PART 3
def part3():
    data = np.loadtxt("assign1_data.txt", skiprows=1)
    x1 = data[:, 0]
    x2 = data[:, 1]
    y = data[:, 2]
    z = data[:,3]
    A = np.vstack([x1, x2, np.ones(len(x1))]).T
    w1, w2, b = np.linalg.lstsq(A, y, rcond=None)[0]
    predictions = w1 * x1 + w2 * x2 + b
    classifications = (predictions > 0).astype(int)
    success_rate = np.mean(classifications == z) * 100
    print(f"Success rate: {success_rate:.2f}%")

part3()

print("PART 4 IS RUNNING")
#PART 4
def train_test_split(data, split_percentage):
    split_index = int(len(data) * split_percentage / 100)
    return data[:split_index], data[split_index:]

def get_classification_accuracy(preds, actual):
    classifications = (preds > 0).astype(int)
    return np.mean(classifications == actual) * 100

data = np.loadtxt("assign1_data.txt", skiprows=1)
x1 = data[:, 0]
x2 = data[:, 1]
y = data[:, 2]
z = data[:,3]
results = {}

for split in [25, 50, 75]:
    train_x1, test_x1 = train_test_split(x1, split)
    train_x2, test_x2 = train_test_split(x2, split)
    train_y, test_y = train_test_split(z, split)
    A_train = np.vstack([train_x1, train_x2, np.ones(len(train_x1))]).T
    w1, w2, b = np.linalg.lstsq(A_train, train_y, rcond=None)[0]
    predictions = w1 * test_x1 + w2 * test_x2 + b
    accuracy = get_classification_accuracy(predictions, test_y)
    results[f"Train on {split}%"] = accuracy

baseline_predictions = np.zeros_like(z)
baseline_accuracy = get_classification_accuracy(baseline_predictions, z)
results["Baseline (w1=w2=b=0)"] = baseline_accuracy

for key, value in results.items():
    print(f"{key}: {value:.2f}%")
