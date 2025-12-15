import csv
import random

def split_data(csv_path: str, train_size: float):
    assert 0.0 < train_size < 1.0, "Incorrect Input"
    data = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            row.pop("Id")
            data.append(row)
    random.shuffle(data)
    split_index = int(len(data) * train_size)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

def get_score(x: list, w: list, b: float):
    n = len(x)
    score = 0
    for i in range(n):
        score += x[i] * w[i]
    return score + b

def get_sign(z: float):
    if z >= 0:
        return 1
    else:
        return -1

def update_weights(x: list, w: list, b: float, error: int, eta: float):
    n = len(x)
    for i in range(n):
        w[i] += eta * error * x[i]
    b += eta * error
    return w, b

def main(n: int, train_data: list, test_data: list, first_label: str, second_label: str):

        # Training
        w = [random.uniform(-0.1, 0.1) for _ in range(n)]
        b = 0
        eta = 1
        epochs = 0
        while True:
            epochs += 1
            errors = 0
            for row in train_data:
                values = list(row.values())
                x = [float(value) for value in values[:n]]
                label = values[n]
                if label == first_label:
                    y = 1
                elif label == second_label:
                    y = -1
                y_hat = get_sign(get_score(x, w, b))
                error = y - y_hat
                if error != 0:
                    w, b = update_weights(x, w, b, error, eta)
                    errors += 1
            if errors == 0:
                break

        # Testing
        total, correct, incorrect = 0, 0, 0
        for row in test_data:
            values = list(row.values())
            x = [float(value) for value in values[:n]]
            label = values[n]
            if label == first_label:
                y = 1
            elif label == second_label:
                y = -1
            y_hat = get_sign(get_score(x, w, b))
            total += 1
            if y == y_hat:
                correct += 1
            else:
                incorrect += 1
        return f"""Epochs: {epochs}
Total: {total}
Correct: {correct}
Incorrect: {incorrect}
Accuracy: {correct / total}"""

if __name__ == "__main__":
    train_data, test_data = split_data("Iris.csv", 0.8)
    print(main(4, train_data, test_data, "Iris-setosa", "Iris-versicolor"))