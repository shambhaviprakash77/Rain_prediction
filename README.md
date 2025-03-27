# Weather Prediction using Neural Networks

## Overview
This project utilizes a **Neural Network Model** to predict whether it will rain the next day based on historical weather data. The dataset used is `weatherAUS.csv`, which contains weather observations from various locations in Australia.

The model preprocesses the data, handles missing values, encodes categorical features, applies feature scaling, and trains a **deep learning model** using **Keras and TensorFlow**.

## Features
- Data preprocessing and visualization
- Encoding categorical data
- Handling missing values
- Feature scaling and outlier removal
- Neural network-based prediction model
- Model evaluation using confusion matrices and classification reports

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- TensorFlow/Keras

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/shambhavi77/weather-prediction.git
   cd weather-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the script:
   ```sh
   python weather_prediction.py
   ```

## Data Preprocessing
- **Datetime Encoding:** The `Date` column is converted to a cyclic representation using sine and cosine transformations.
- **Handling Missing Values:** Categorical variables are filled with the mode, and numeric variables are filled with the median.
- **Feature Scaling:** StandardScaler is used to normalize numerical features.
- **Outlier Removal:** Outliers in multiple features such as temperature, rainfall, humidity, and wind speed are removed based on statistical thresholds.

## Neural Network Model
- **Layers:**
  - Input Layer (26 neurons)
  - 2 Hidden Layers (32 neurons each, ReLU activation)
  - 1 Hidden Layer (16 neurons, ReLU activation, Dropout 25%)
  - 1 Hidden Layer (8 neurons, ReLU activation, Dropout 50%)
  - Output Layer (Sigmoid activation)
- **Optimization:** Adam Optimizer
- **Loss Function:** Binary Cross-Entropy
- **Early Stopping:** Implemented to prevent overfitting

## Model Training and Evaluation
The model is trained for 150 epochs with batch size = 32 and an 80-20 train-test split.



### Classification Report
```sh
Precision: 0.84
Recall: 0.79
F1 Score: 0.81
Accuracy: 82%
```

## Author
**Shambhavi Prakash**  
GitHub: shambhaviprakash77

