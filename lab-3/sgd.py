import ucimlrepo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time
from ucimlrepo import fetch_ucirepo

ARITHMETIC_COUNTER = 0
GRAD_CALCS = 0


def plus(a, b):
    global ARITHMETIC_COUNTER
    ARITHMETIC_COUNTER += 1
    return a + b


def mul(a, b):
    global ARITHMETIC_COUNTER
    ARITHMETIC_COUNTER += 1
    return a * b


def div(a, b):
    global ARITHMETIC_COUNTER
    ARITHMETIC_COUNTER += 1
    return a / b


def scal(a, b):
    res = 0
    for i in range(0, len(a)):
        res = plus(res, mul(a[i], b[i]))
    return res


wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets

X = X.dropna()
y = y.loc[X.index]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

X_train, X_test, y_train, y_test = train_test_split(
    X_b, y.values, test_size=0.2, random_state=42
)


def sgd_regressor(X, y, batch_size=1, n_epochs=100, learning_rate=0.01,
                  lr_schedule='constant', decay_rate=0.01, decay_steps=1000,
                  regularization='none', alpha=0.0001, l1_ratio=0.5):
    global ARITHMETIC_COUNTER
    global GRAD_CALCS
    ARITHMETIC_COUNTER = 0
    GRAD_CALCS = 0

    n_samples, n_features = X.shape
    theta = np.random.randn(n_features).tolist()
    mse_history = []

    # Initialize learning rate schedule parameters
    initial_lr = learning_rate
    step_counter = 0

    def grad_calc(w, x, y, current_bs):
        global GRAD_CALCS
        GRAD_CALCS += 1
        wx = scal(w, x)
        error = plus(wx, -y)
        factor = mul(div(2, current_bs), error)
        grad = [0] * len(x)
        for i in range(len(x)):
            grad[i] = mul(factor, x[i])
        return grad

    def sum_vectors(a, b):
        res = [0] * len(a)
        for i in range(len(a)):
            res[i] = plus(a[i], b[i])
        return res

    def scale_vector(a, scalar):
        return [mul(scalar, x) for x in a]

    def subtract_vectors(a, b):
        return [plus(a[i], -b[i]) for i in range(len(a))]

    def compute_regularization_gradient(theta, alpha, l1_ratio, regularization):
        grad = [0] * len(theta)
        if regularization == 'none':
            return grad

        for j in range(1, len(theta)):  # Skip bias term
            if regularization == 'l2':
                grad[j] = mul(alpha, theta[j])
            elif regularization == 'l1':
                sign = 1 if theta[j] > 0 else -1 if theta[j] < 0 else 0
                grad[j] = mul(alpha, sign)
            elif regularization == 'elasticnet':
                sign = 1 if theta[j] > 0 else -1 if theta[j] < 0 else 0
                l1_term = mul(mul(alpha, l1_ratio), sign)
                l2_term = mul(mul(alpha, plus(1, -l1_ratio)), theta[j])
                grad[j] = plus(l1_term, l2_term)
        return grad

    def get_current_lr(initial_lr, step, schedule, decay_rate, decay_steps):
        if schedule == 'constant':
            return initial_lr
        elif schedule == 'time-based':
            return div(initial_lr, plus(1, mul(decay_rate, step)))
        elif schedule == 'step':
            exponent = div(step, decay_steps)
            return mul(initial_lr, pow(plus(1, -decay_rate), exponent))
        elif schedule == 'exponential':
            return mul(initial_lr, pow(plus(1, -decay_rate), step))
        return initial_lr

    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(n_samples)
        X_shuffled = X[shuffled_indices].tolist()
        y_shuffled = y[shuffled_indices].tolist()

        n_batches = n_samples // batch_size
        if n_samples % batch_size != 0:
            n_batches += 1

        start = 0 * batch_size
        end = min(start + batch_size, n_samples)
        current_batch_size = end - start
        xi = X_shuffled[start:end]
        yi = [item[0] for item in y_shuffled[start:end]]

        total_grad = [0] * n_features
        for i in range(current_batch_size):
            sample_grad = grad_calc(theta, xi[i], yi[i], current_batch_size)
            total_grad = sum_vectors(total_grad, sample_grad)

        # Add regularization gradient
        reg_grad = compute_regularization_gradient(theta, alpha, l1_ratio, regularization)
        for j in range(n_features):
            total_grad[j] = plus(total_grad[j], reg_grad[j])

        # Update learning rate based on schedule
        current_lr = get_current_lr(initial_lr, step_counter, lr_schedule, decay_rate, decay_steps)
        step_counter += 1

        # Update weights
        theta = subtract_vectors(theta, scale_vector(total_grad, current_lr))

        theta_np = np.array(theta)
        y_pred = X.dot(theta_np)
        mse = mean_squared_error(y, y_pred)
        mse_history.append(mse)
        if mse < 1e-2:
            break

    print(f"Arithmetic operations for batch size {batch_size}: {ARITHMETIC_COUNTER}")
    print(f"Grad calcs for {batch_size}: {GRAD_CALCS}")
    return theta_np, mse_history


# Experiment configurations
configurations = [
    {'name': 'Constant LR, No Reg', 'lr_schedule': 'constant', 'regularization': 'none'},
    {'name': 'Time-based Decay', 'lr_schedule': 'time-based', 'regularization': 'none'},
    {'name': 'L2 Regularization', 'lr_schedule': 'constant', 'regularization': 'l2'},
    {'name': 'ElasticNet', 'lr_schedule': 'constant', 'regularization': 'elasticnet'},
]

batch_sizes = [1, 8, 32, 128, 512, 1024, len(X_train)]
n_epochs = 100
learning_rate = 0.01
results = {cfg['name']: {'batch_size': [], 'time': [], 'final_mse': [], 'mse_history': []} for cfg in configurations}

for config in configurations:
    print(f"\nTraining with configuration: {config['name']}")
    for batch_size in batch_sizes:
        print(f"  Batch size: {batch_size}")
        start_time = time.time()

        theta, mse_history = sgd_regressor(
            X_train,
            y_train,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            lr_schedule=config['lr_schedule'],
            decay_rate=0.01,
            decay_steps=100,
            regularization=config['regularization'],
            alpha=0.001,
            l1_ratio=0.5
        )

        exec_time = time.time() - start_time
        y_test_pred = X_test.dot(theta)
        test_mse = mean_squared_error(y_test, y_test_pred)

        results[config['name']]['batch_size'].append(batch_size)
        results[config['name']]['time'].append(exec_time)
        results[config['name']]['final_mse'].append(test_mse)
        results[config['name']]['mse_history'].append(mse_history)

# Plotting results
plt.figure(figsize=(18, 12))

# Time vs Batch Size
plt.subplot(2, 2, 1)
for config_name, data in results.items():
    plt.plot(data['batch_size'], data['time'], 'o-', label=config_name)
plt.xscale('log')
plt.xlabel('Batch Size')
plt.ylabel('Time (seconds)')
plt.title('Training Time vs Batch Size')
plt.legend()

# Test MSE vs Batch Size
plt.subplot(2, 2, 2)
for config_name, data in results.items():
    plt.plot(data['batch_size'], data['final_mse'], 'o-', label=config_name)
plt.xscale('log')
plt.xlabel('Batch Size')
plt.ylabel('Test MSE')
plt.title('Test Error vs Batch Size')
plt.legend()

# Convergence Speed
plt.subplot(2, 2, 3)
for config_name, data in results.items():
    for i, bs in enumerate(batch_sizes):
        if bs in [1, 32]:  # Plot only selected batch sizes for clarity
            plt.plot(data['mse_history'][i], label=f"{config_name} (bs={bs})")
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.title('Convergence Speed')

# Error vs Wall-clock Time
plt.subplot(2, 2, 4)
for config_name, data in results.items():
    for i, (bs, mse_history, exec_time) in enumerate(zip(
            data['batch_size'],
            data['mse_history'],
            data['time']
    )):
        if bs in [1, 32]:  # Plot only selected batch sizes
            relative_time = np.linspace(0, exec_time, len(mse_history))
            plt.plot(relative_time, mse_history, label=f"{config_name} (bs={bs})")
plt.xlabel('Time (s)')
plt.ylabel('MSE')
plt.legend()
plt.title('Error vs Wall-clock Time')

plt.tight_layout()
plt.show()
