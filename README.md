# Machine Learning Models Collection

A comprehensive collection of machine learning algorithms implemented from scratch in Python. This repository contains implementations of various supervised learning, unsupervised learning, and optimization algorithms with practical examples and datasets.

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Utility Functions](#utility-functions)
- [Examples](#examples)
- [Requirements](#requirements)
- [Contributing](#contributing)

## 🔍 Overview

This project demonstrates the implementation of fundamental machine learning algorithms without relying on external ML libraries like scikit-learn. Each algorithm is implemented with clear, educational code that helps understand the underlying mathematical concepts and computational processes.

### Key Features
- ✅ From-scratch implementations
- ✅ Multiple algorithm variants (e.g., Gradient Descent vs OLS)
- ✅ Real-world dataset examples
- ✅ Performance metrics and evaluation
- ✅ Visualization capabilities
- ✅ Modular and reusable code structure

## 🤖 Algorithms Implemented

### Supervised Learning

#### 1. **Simple Linear Regression**
- **Location**: `Simple Linear Regression/`
- **Implementation**: Two approaches available
  - Gradient Descent optimization
  - Ordinary Least Squares (OLS) analytical solution
- **Use Case**: Predicting salary based on years of experience
- **Dataset**: `Salary_Data.csv` (30+ records)

#### 2. **Multiple Linear Regression**
- **Location**: `Multiple Linear Regression/`
- **Implementation**: Supports multiple features
  - Gradient Descent optimization
  - Ordinary Least Squares (OLS) analytical solution
- **Use Case**: Sales prediction based on advertising spend across multiple channels
- **Dataset**: `Advertising and Sales.csv` (4500+ records)

#### 3. **Logistic Regression**
- **Location**: `Logistic Regression/`
- **Implementation**: Binary classification using sigmoid function
- **Optimization**: Gradient Descent with logistic cost function
- **Use Case**: Diabetes prediction based on medical indicators
- **Dataset**: `diabetes2.csv` (768 records, 8 features)

#### 4. **K-Nearest Neighbors (KNN)**
- **Location**: `K-Nearest Neighbors/`
- **Implementation**: Distance-based classification
- **Use Case**: Breast cancer diagnosis (malignant/benign classification)
- **Dataset**: `KNNAlgorithmDataset.csv` (569 records, 30+ features)

### Unsupervised Learning

#### 5. **K-Means Clustering**
- **Location**: `K-Means Clustering/`
- **Implementation**: Iterative centroid-based clustering
- **Features**: 
  - Random centroid initialization
  - Convergence tolerance control
  - Maximum iterations limit
- **Use Case**: Data segmentation and pattern discovery

#### 6. **DBSCAN Clustering**
- **Location**: `DBSCAN/`
- **Implementation**: Density-based clustering algorithm
- **Features**:
  - Noise point detection
  - Arbitrary cluster shapes
  - Automatic cluster count determination
- **Parameters**: Radius (ε) and minimum neighbors

### Time Series & Forecasting

#### 7. **Double Exponential Smoothing**
- **Location**: `Double Exponential Smoothing/`
- **Implementation**: Holt's method for trend analysis
- **Features**:
  - Level and trend smoothing parameters (α, β)
  - Configurable initial values
  - Parameter validation
- **Use Case**: Time series forecasting with trend

### Optimization & Scheduling

#### 8. **Johnson's Rule**
- **Location**: `Johnson's Rule/`
- **Implementation**: Two-machine scheduling optimization
- **Purpose**: Minimize total completion time for job scheduling
- **Use Case**: Production scheduling and workflow optimization

## 📁 Project Structure

```
Machine Learning Models/
├── README.md
├── utils/                          # Shared utility functions
│   ├── utils.py                   # Data preprocessing utilities
│   ├── metrics.py                 # Performance evaluation metrics
│   └── __pycache__/
├── Simple Linear Regression/
│   ├── simple_linear_regression.py
│   ├── main.py
│   ├── Salary_Data.csv
│   └── __pycache__/
├── Multiple Linear Regression/
│   ├── multiple_linear_regression.py
│   ├── main.py
│   ├── Advertising and Sales.csv
│   └── __pycache__/
├── Logistic Regression/
│   ├── logistic_regression.py
│   ├── main.py
│   ├── diabetes2.csv
│   └── __pycache__/
├── K-Nearest Neighbors/
│   ├── k_nearest_neighbors.py
│   ├── main.py
│   ├── KNNAlgorithmDataset.csv
│   └── __pycache__/
├── K-Means Clustering/
│   ├── k_means_clustering.py
│   ├── main.py
│   └── __pycache__/
├── DBSCAN/
│   ├── dbscan.py
│   ├── main.py
│   └── __pycache__/
├── Double Exponential Smoothing/
│   ├── double_exponential_smoothing.py
│   ├── main.py
│   └── __pycache__/
└── Johnson's Rule/
    ├── johnsons_rule.py
    ├── main.py
    └── __pycache__/
```

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Machine Learning Models"
   ```

2. **Install required dependencies**:
   ```bash
   pip install numpy pandas matplotlib
   ```

3. **Verify installation** by running any example:
   ```bash
   cd "Simple Linear Regression"
   python main.py
   ```

## 📖 Usage

Each algorithm can be used independently. Here are basic usage patterns:

### Simple Linear Regression Example
```python
from simple_linear_regression import SimpleLinearRegression
import pandas as pd

# Load data
df = pd.read_csv("Salary_Data.csv")
X = df.iloc[:, 0].values  # Years of experience
y = df.iloc[:, 1].values  # Salary

# Create and train model
model = SimpleLinearRegression(epochs=1000, learning_rate=0.01)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```

### K-Means Clustering Example
```python
from k_means_clustering import KMeansClustering
import numpy as np

# Generate or load data
X = np.random.randn(100, 2)

# Create and fit model
model = KMeansClustering(k=3)
clusters = model.predict(X)

# Access centroids
centroids = model.centroids
```

### Logistic Regression Example
```python
from logistic_regression import LogisticRegression
import pandas as pd

# Load data
df = pd.read_csv("diabetes2.csv")
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Target

# Create and train model
model = LogisticRegression(epochs=1000, learning_rate=0.01)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```

## 📊 Datasets

| Algorithm | Dataset | Records | Features | Target |
|-----------|---------|---------|----------|---------|
| Simple Linear Regression | Salary_Data.csv | 30 | 1 (Years Experience) | Salary |
| Multiple Linear Regression | Advertising and Sales.csv | 4,573 | 4 (TV, Radio, Social Media, Influencer) | Sales |
| Logistic Regression | diabetes2.csv | 768 | 8 (Medical indicators) | Diabetes (Binary) |
| K-Nearest Neighbors | KNNAlgorithmDataset.csv | 569 | 30 (Cell measurements) | Diagnosis (M/B) |

## 🛠 Utility Functions

The `utils/` directory contains shared functionality:

### `utils.py`
- **`normalize(data)`**: Min-max normalization
- **`sigmoid(x)`**: Sigmoid activation function
- **`train_test_split(X, y, test_size=0.2)`**: Data splitting utility

### `metrics.py`
- **`R_squared(y, y_pred)`**: Coefficient of determination
- **`MAPE(y, y_pred)`**: Mean Absolute Percentage Error

## 🎯 Examples

Each algorithm folder contains a `main.py` file demonstrating:
- Data loading and preprocessing
- Model initialization and training
- Performance evaluation
- Results visualization (where applicable)

To run examples:
```bash
# Navigate to any algorithm folder
cd "K-Means Clustering"
python main.py

# Or run from root directory
python "Simple Linear Regression/main.py"
```

## 📋 Requirements

- **Python**: 3.7+
- **NumPy**: For numerical computations
- **Pandas**: For data manipulation
- **Matplotlib**: For visualization

Install with:
```bash
pip install numpy pandas matplotlib
```

## 🔍 Algorithm Details

### Gradient Descent Implementation
Most supervised learning algorithms implement gradient descent optimization:
- Configurable learning rate and epochs
- Convergence through iterative parameter updates
- Support for both single and multiple features

### Distance Metrics
Clustering algorithms use Euclidean distance:
```python
distance = np.sqrt(((point1 - point2) ** 2).sum())
```

### Performance Metrics
- **Regression**: R², MAPE
- **Classification**: Accuracy (via prediction comparison)
- **Clustering**: Visual inspection and silhouette analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/algorithm-name`)
3. Implement your algorithm following the existing structure
4. Add example usage in `main.py`
5. Include appropriate dataset (if applicable)
6. Update this README
7. Submit a pull request

### Guidelines for New Algorithms
- Follow the existing class structure pattern
- Include both `fit()` and `predict()` methods where applicable
- Add comprehensive docstrings
- Provide example usage
- Include performance metrics

## 📝 Notes

- All implementations are educational and optimized for clarity over performance
- Algorithms use basic Python/NumPy operations without ML libraries
- Real-world applications should consider using optimized libraries like scikit-learn
- Each implementation includes parameter validation and error handling

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Author**: Machine Learning Implementation Collection  
**Last Updated**: October 2025

For questions or suggestions, please open an issue or contribute to the project!
