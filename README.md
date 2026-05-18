# Итоговый проект: Letter Classifier (EMNIST Letters) - `Александра Бужор`

Распознавание рукописных букв английского алфавита (A–Z) с помощью многослойного перцептрона.  
Реализовано параллельно на **PyTorch** и **TensorFlow/Keras** с полным циклом: данные → обучение → анализ ошибок → инференс.

## Структура репозитория

```
.
├── Alexandra_Bujor_Letter_Classifier.ipynb   # основной ноутбук
├── data/                                     # датасет EMNIST (скачивается автоматически)
├── test_images/                              # тестовые изображения для инференса
├── mlp_baseline_torch.pth                    # веса PyTorch Baseline
├── mlp_dropout_torch.pth                     # веса PyTorch + Dropout
├── mlp_baseline_tf.h5                        # веса TensorFlow Baseline
├── mlp_dropout_tf.h5                         # веса TensorFlow + Dropout
├── class_distribution.png                    # распределение классов (EDA)
├── examples_per_class.png                    # примеры изображений A–Z
├── baseline_comparison.png                   # сравнение loss/accuracy двух фреймворков
├── dropout_comparison.png                    # baseline vs dropout
├── confusion_matrices.png                    # матрицы ошибок
├── per_class_accuracy.png                    # точность по каждой букве
├── error_examples.png                        # визуализация ошибочных предсказаний
└── inference_results.png                     # результаты инференса на 5 буквах
```

## Датасет

**EMNIST Letters** загружен через `tensorflow_datasets` (зеркало Google CDN).

| Параметр | Значение |
|---|---|
| Обучающая выборка (train) | 79 920 изображений (90%) |
| Валидационная выборка (val) | 8 880 изображений (10%) |
| Тестовая выборка (test) | 14 800 изображений |
| Всего | 103 600 изображений |
| Размер изображения | 28 × 28 пикселей, grayscale |
| Размер вектора признаков | 784 (после flatten) |
| Классов | 26 (A – Z) |
| Баланс классов | равномерный (min 3 365 / max 3 453, дисбаланс 1.03×) |
| Диапазон пикселей | [0.0, 1.0] |
| Среднее / std пикселей | 0.1722 / 0.3310 |

## Архитектура модели (MLP)

Одинаковая для PyTorch и TensorFlow:

```
Вход (784) → Dense(512, ReLU) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(26, Softmax)
```

Опциональный **Dropout(p=0.3)** после каждого скрытого слоя (бонусная часть).

| Слой | Вход | Выход | Активация |
|---|---|---|---|
| 1 | 784 | 512 | ReLU |
| 2 | 512 | 256 | ReLU |
| 3 | 256 | 128 | ReLU |
| 4 | 128 | 26 | Softmax |

Обучаемых параметров: **569 498**

## Результаты обучения (10 эпох)

| Модель | Test Accuracy | Val Accuracy | Overfitting Gap |
|---|---|---|---|
| PyTorch Baseline MLP | **0.8936** | 0.9102 | +0.0426 |
| PyTorch MLP + Dropout(0.3) | 0.8921 | 0.9074 | −0.0116 |
| TensorFlow Baseline MLP | 0.8926 | 0.9046 | +0.0488 |
| TensorFlow MLP + Dropout(0.3) | **0.8952** | 0.9079 | −0.0142 |

> Отрицательный Overfitting Gap у Dropout-моделей означает отсутствие переобучения.

**Лучшие модели:**
- PyTorch → **Baseline** (Test Acc = 0.8936)
- TensorFlow → **MLP + Dropout** (Test Acc = 0.8952)

**Среднее время на эпоху:** PyTorch — 5.5 с, TensorFlow — 8.7 с (CPU, без GPU)

## Анализ ошибок

### Топ-5 проблемных пар (PyTorch Baseline)

| Ранг | Истинная | Предсказана | Ошибок |
|---|---|---|---|
| 1 | L | I | 216 |
| 2 | I | L | 130 |
| 3 | G | Q | 122 |
| 4 | Q | G | 62 |
| 5 | D | O | 34 |

### Топ-5 проблемных пар (TensorFlow + Dropout)

| Ранг | Истинная | Предсказана | Ошибок |
|---|---|---|---|
| 1 | L | I | 179 |
| 2 | I | L | 160 |
| 3 | G | Q | 121 |

> Паттерны ошибок обеих моделей совпадают — путаются визуально похожие буквы: **I/L**, **G/Q**, **D/O**.

### Per-class accuracy (PyTorch Baseline)

**Лучшие буквы:** M (97.0%), O (96.8%), S (96.5%), B (94.6%), P (94.6%)

Всего ошибок на тестовой выборке: **1 574 из 14 800**

## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/<your-username>/letter-classifier-emnist.git
cd letter-classifier-emnist
```

### 2. Создание окружения

```bash
conda create -n ml_env python=3.11
conda activate ml_env
```

### 3. Установка зависимостей

```bash
pip install torch==2.2.2 torchvision==0.17.2 \
            tensorflow==2.16.2 tensorflow-datasets \
            matplotlib numpy==1.26.4 scikit-learn==1.8.0 pillow
```

### 4. Запуск ноутбука

```bash
jupyter notebook Alexandra_Bujor_Letter_Classifier.ipynb
```

Запускайте ячейки последовательно сверху вниз. При первом запуске данные скачаются автоматически через `tensorflow_datasets` (~130 MB, CDN Google).

> **Примечание:** официальный сервер NIST (`biometrics.nist.gov`) периодически недоступен или работает очень медленно. Ноутбук автоматически переключается на `tensorflow_datasets` в случае ошибки.

## Инференс на своём изображении

```python
# Используйте функцию из ноутбука:
letter, confidence = predict_letter("my_letter.png", best_model_torch, framework="pytorch")
print(f"Предсказание: {letter}, уверенность: {confidence:.2%}")
```

Функция автоматически:
- конвертирует изображение в grayscale 28×28
- инвертирует цвета при светлом фоне (mean > 127)
- нормализует пиксели к [0, 1]

## Воспроизводимость

Random seed зафиксирован во всех фреймворках:

```python
SEED = 42
torch.manual_seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
```

## Стек технологий

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange?logo=pytorch)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-FF6F00?logo=tensorflow)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?logo=numpy)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikit-learn)

## Этапы проекта

- **Data Pipeline** — загрузка через tfds/torchvision, предобработка, конвертация между фреймворками
- **EDA** — распределение 26 классов (дисбаланс 1.03×), визуализация 130 примеров (5 на букву)
- **Baseline MLP** — обучение 10 эпох, сравнение PyTorch vs TensorFlow по accuracy, loss и скорости
- **Dropout** — регуляризация p=0.3, анализ Overfitting Gap *(бонус)*
- **Анализ ошибок** — Confusion Matrix 26×26, топ-5 проблемных пар, 1 574 ошибки на 14 800 примерах
- **Инференс** — функция `predict_letter()`, тест на 5 рукописных буквах (O, Q, I, L, G)