# AI-Powered-Disease-Prediction-Using-Streamlit
AI-Powered Disease Prediction Using Streamlit" is  that harnesses the power of machine learning algorithms to predict potential diseases like Diabetes, parkinsons, heart issues based on users' health data.

# AI-Powered Disease Prediction Using Streamlit

This project is an AI-powered disease prediction application built using **Streamlit** and various data analytics techniques such as **NumPy** and machine learning. The application predicts three major diseases based on user-input health data: **Diabetes**, **Parkinson's Disease**, and **Heart Disease**.

## Features:
- **Disease Prediction:** The app predicts the likelihood of developing **Diabetes**, **Parkinson's Disease**, or **Heart Disease** based on health-related data like symptoms and medical history.
- **Interactive UI:** The application uses **Streamlit** to provide a simple and interactive user interface for easy data input and prediction viewing.
- **Data Analytics:** The app utilizes advanced **data analytics** techniques, including statistical analysis and machine learning algorithms, to make accurate predictions.
- **Model Integration:** The machine learning models used for prediction are trained on well-known datasets such as the **Pima Indians Diabetes Dataset**, **Parkinson’s Disease Dataset**, and **Heart Disease Dataset**.

## Technologies Used:
- **Streamlit:** For building the interactive web app interface.
- **NumPy:** For numerical operations and data manipulation.
- **Pandas:** For data handling and manipulation.
- **Scikit-learn:** For building and evaluating machine learning models.
- **Matplotlib/Seaborn:** For data visualization and presenting insights.

## Installation:

### Prerequisites:
To run this project locally, ensure you have the following installed:
- Python 3.x
- pip (Python package installer)

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Powered-Disease-Prediction.git
   cd AI-Powered-Disease-Prediction
   ```
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Visit `http://localhost:8501` in your browser to access the app.

## Usage:

1. **Input your health data:** Enter relevant health information, including symptoms and medical history.
2. **Get predictions:** After entering the data, the app will use AI models to predict the likelihood of having **Diabetes**, **Parkinson’s Disease**, or **Heart Disease**.
3. **View results:** The app displays the prediction and a confidence score based on your data.

## Datasets:
- **Diabetes Dataset:** Pima Indians Diabetes Database (available at UCI Machine Learning Repository).
- **Parkinson’s Disease Dataset:** Parkinson's Disease Classification dataset (available at UCI Machine Learning Repository).
- **Heart Disease Dataset:** Heart Disease dataset (available at UCI Machine Learning Repository).

## Machine Learning Models:
- **Logistic Regression:** For binary classification of disease vs. no disease.
- **Random Forest Classifier:** For multi-class disease classification.
- **Support Vector Machines (SVM):** For classifying patterns in disease data.

## Acknowledgments:
- **UCI Machine Learning Repository** for providing the datasets used in this project.
- **Streamlit** for providing an easy-to-use framework to deploy the web application.
- **Scikit-learn**, **NumPy**, **Pandas**, and other libraries for providing essential tools for data analysis and model development.

