# 🩺 Diabetes Prediction App

A comprehensive machine learning project for diabetes prediction using multiple algorithms, deployed as an interactive web application on Hugging Face Spaces.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Ashvitta07/Diabetes-Prediction)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-red.svg)](https://mlflow.org/)

## 📊 Project Overview

This project implements a complete machine learning pipeline for diabetes prediction, featuring advanced data preprocessing, comprehensive model comparison, and production-ready deployment. The system achieves **87% accuracy** using a Random Forest classifier.

## 🚀 Live Demo

Try the interactive diabetes prediction app: [**Diabetes Prediction App**](https://huggingface.co/spaces/Ashvitta07/Diabetes-Prediction)

## 🔄 Project Flow

### 1. **Data Preprocessing Pipeline**
- **Data Loading**: Diabetes dataset with 8 features and 1 target variable
- **Missing Value Handling**: Replaces zero values with column means
- **Outlier Treatment**: IQR-based outlier capping for robust predictions
- **Data Transformation**: Yeo-Johnson Power Transformation for distribution normalization
- **Feature Scaling**: StandardScaler for consistent feature ranges

### 2. **Class Imbalance Resolution**
- **SMOTE Implementation**: RandomOverSampler to balance diabetes/non-diabetes cases
- **Balanced Training**: Ensures equal representation of both classes

### 3. **Model Comparison & Selection**
**Models Tested:**
- Logistic Regression
- Naive Bayes  
- Random Forest ⭐ **(Best: 87% accuracy)**
- Gradient Boosting
- AdaBoost
- XGBoost

### 4. **MLflow Integration**
- **Experiment Tracking**: Systematic model experimentation and logging
- **Model Versioning**: Automatic model artifact management
- **Reproducibility**: Complete experiment history and parameter tracking

### 5. **Production Deployment**
- **Gradio Interface**: Interactive web application
- **Real-time Prediction**: Live preprocessing and prediction
- **Gender-aware Output**: Personalized results based on user input

## 🎯 Unique Features

### **Advanced Data Quality Management**
- **Multi-step Preprocessing**: Systematic handling of missing values, outliers, and distributions
- **Power Transformation**: Yeo-Johnson transformation for optimal data distribution
- **Persistent Preprocessing**: Saved transformation objects for consistent inference

### **Professional Model Management**
- **MLflow Integration**: Enterprise-grade experiment tracking and model versioning
- **Automated Selection**: Intelligent best model identification and persistence
- **Comprehensive Comparison**: 6-algorithm performance evaluation

### **Production-Ready Architecture**
- **End-to-end Pipeline**: Seamless training-to-deployment workflow
- **Consistent Preprocessing**: Identical data processing for training and inference
- **User-friendly Interface**: Intuitive Gradio app with clear input validation

### **Data Science Best Practices**
- **Visualization Suite**: Comprehensive EDA with boxplots, histograms, and correlation analysis
- **Class Balance**: SMOTE implementation for imbalanced dataset handling
- **Robust Validation**: Proper train-test split with reproducible results

## 📁 Project Structure

```
Diabetes/
├── diabetes.ipynb          # Main analysis notebook
├── app.py                  # Gradio web application
├── diabetes.csv            # Dataset
├── best_model.pkl          # Best performing model
├── scaler.pkl             # StandardScaler object
├── Transformation.pkl     # PowerTransformer object
├── requirements.txt       # Dependencies
├── mlruns/               # MLflow experiment tracking
└── README.md             # Project documentation
```

## 🛠️ Technical Stack

- **Python 3.8+**
- **Scikit-learn**: Machine learning algorithms
- **MLflow**: Experiment tracking and model management
- **Gradio**: Web application interface
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Imbalanced-learn**: Class imbalance handling

## 📈 Model Performance

| Model | Accuracy |
|-------|----------|
| Random Forest | **87.0%** |
| Gradient Boosting | 79.5% |
| Naive Bayes | 75.0% |
| Logistic Regression | 74.0% |
| AdaBoost | 73.0% |
| XGBoost | 73.0% |

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://huggingface.co/spaces/Ashvitta07/Diabetes-Prediction
cd Diabetes-Prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the notebook**
```bash
jupyter notebook diabetes.ipynb
```

4. **Launch the web app**
```bash
python app.py
```

### Using the Web App

1. Visit the [live demo](https://huggingface.co/spaces/Ashvitta07/Diabetes-Prediction)
2. Fill in the patient details:
   - Gender
   - Pregnancies
   - Glucose level
   - Blood Pressure
   - Skin Thickness
   - Insulin level
   - BMI
   - Diabetes Pedigree Function
   - Age
3. Click "Submit" to get the prediction

## 📊 Dataset Information

The dataset contains 768 samples with 8 features:
- **Pregnancies**: Number of pregnancies
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure
- **SkinThickness**: Triceps skin fold thickness
- **Insulin**: 2-Hour serum insulin
- **BMI**: Body mass index
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years

## 🔬 Methodology

1. **Exploratory Data Analysis**: Comprehensive visualization and statistical analysis
2. **Data Preprocessing**: Missing value imputation, outlier treatment, and normalization
3. **Feature Engineering**: Power transformation and standardization
4. **Model Training**: Multiple algorithm comparison with cross-validation
5. **Model Selection**: Best model identification based on accuracy metrics
6. **Deployment**: Production-ready web application with consistent preprocessing

## 📝 Key Insights

- **Random Forest** emerged as the best performing algorithm with 87% accuracy
- **Data quality** improvements significantly enhanced model performance
- **Class balancing** using SMOTE improved prediction reliability
- **Power transformation** normalized feature distributions effectively

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Ashvitta07**
- GitHub: [@Ashvitta07](https://github.com/Ashvitta07)
- Hugging Face: [@Ashvitta07](https://huggingface.co/Ashvitta07)

## 🙏 Acknowledgments

- Dataset: Pima Indians Diabetes Database
- MLflow for experiment tracking
- Gradio for web application framework
- Scikit-learn for machine learning algorithms

---

⭐ **Star this repository if you found it helpful!**