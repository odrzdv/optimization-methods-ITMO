import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features.copy()
y = wine_quality.data.targets.copy()

# Fill NaN values safely
X.fillna(X.mean(), inplace=True)
y = y.values.ravel()

# Нормализация и разделение данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Конвертация в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Конфигурации экспериментов
configurations = {
    "SGD": {"momentum": 0.0},
    "SGD-Momentum": {"momentum": 0.9}
}
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
epochs = 50
learning_rate = 0.01

# Словарь для результатов
results = {
    config_name: {
        'batch_size': [],
        'time': [],
        'final_mse': [],
        'mse_history': []
    }
    for config_name in configurations
}

# Обучение для каждой конфигурации
for config_name, params in configurations.items():
    momentum = params["momentum"]
    for batch_size in batch_sizes:
        # Инициализация модели и функции потерь
        model = nn.Linear(X_train.shape[1], 1)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        # Создание DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Обучение
        start_time = time.time()
        mse_history = []

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Оценка на тестовом наборе
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_tensor)
                mse = criterion(test_preds, y_test_tensor).item()
                mse_history.append(mse)

        exec_time = time.time() - start_time
        final_mse = mse_history[-1]

        # Сохранение результатов
        results[config_name]['batch_size'].append(batch_size)
        results[config_name]['time'].append(exec_time)
        results[config_name]['final_mse'].append(final_mse)
        results[config_name]['mse_history'].append(mse_history)

# Построение графиков
plt.figure(figsize=(18, 12))

# 1. Time vs Batch Size
plt.subplot(2, 2, 1)
for config_name, data in results.items():
    plt.plot(data['batch_size'], data['time'], 'o-', label=config_name)
plt.xscale('log')
plt.xlabel('Batch Size')
plt.ylabel('Time (seconds)')
plt.title('Training Time vs Batch Size')
plt.legend()

# 2. Test MSE vs Batch Size
plt.subplot(2, 2, 2)
for config_name, data in results.items():
    plt.plot(data['batch_size'], data['final_mse'], 'o-', label=config_name)
plt.xscale('log')
plt.xlabel('Batch Size')
plt.ylabel('Test MSE')
plt.title('Test Error vs Batch Size')
plt.legend()

# 3. Convergence Speed (MSE vs Epochs)
plt.subplot(2, 2, 3)
for config_name, data in results.items():
    for i, bs in enumerate(batch_sizes):
        if bs in [1, 32]:  # Только выбранные размеры батча
            plt.plot(data['mse_history'][i], label=f"{config_name} (bs={bs})")
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.title('Convergence Speed')

# 4. Error vs Wall-clock Time
plt.subplot(2, 2, 4)
for config_name, data in results.items():
    for i, (bs, mse_history, exec_time) in enumerate(zip(
            data['batch_size'],
            data['mse_history'],
            data['time']
    )):
        if bs in [1, 32]:  # Только выбранные размеры батча
            relative_time = np.linspace(0, exec_time, len(mse_history))
            plt.plot(relative_time, mse_history, label=f"{config_name} (bs={bs})")
plt.xlabel('Time (s)')
plt.ylabel('MSE')
plt.legend()
plt.title('Error vs Wall-clock Time')

plt.tight_layout()
plt.show()