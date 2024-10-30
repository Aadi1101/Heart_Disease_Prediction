# Heart Disease Prediction using Machine Learning

This project aims to predict the likelihood of heart disease based on various health metrics using machine learning. The project includes a web application powered by Flask for easy user interaction and visualization of prediction results.


## Table of Contents
- [Project Description](#project-description)
- [Prerequisites](#prerequisites)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Input Parameters](#input-parameters)
- [API Endpoints](#api-endpoints)
- [ML Model](#ml-model)
- [Model Tuning and Preprocessing](#model-tuning-and-preprocessing)
- [Future Enhancements](#future-enhancements)
- [References or Documentation Links](#references-or-documentation-links)
- [Contributing](#contributing)
- [License](#license)

## Project Description
Heart disease is a leading cause of death worldwide. Early diagnosis is crucial for effective management and treatment. This project uses machine learning to predict the likelihood of heart disease based on a series of health metrics, including age, cholesterol levels, blood pressure, and more. The model was trained on historical patient data and integrated into a Flask application for ease of use.

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)

## Technologies Used
The following technologies are used in this project:

- **Programming Language**: Python 3.7+
- **Framework**: Flask (for building the web application)
- **Machine Learning Libraries**:
  - **scikit-learn**
  - **catboost**
  - **xgboost**
  - **GridSearchCV** (for hyperparameter tuning)
  - **StandardScaler** (for data normalization)
- **Data Processing Libraries**:
  - **numpy**
  - **pandas**
  - **seaborn**
  - **openpyxl**
- **Docker**: Containerization with Docker
- **Serialization**: dill (for serializing and deserializing ML models)
- **Notebook**: Jupyter Notebook (for experimentation)

## Installation

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Aadi1101/Heart_Disease_Prediction.git
   cd Heart_Disease_Prediction
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use:
   venv\Scripts\activate
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Flask app:**

   ```bash
   python app.py
   ```
The Flask app will start at http://127.0.0.1:5000/.

## Usage
### Web Application
1. Once the Flask server is running, open your web browser and navigate to http://127.0.0.1:5000/ to access the Heart Disease detection page.

2. Enter the required health parameters (e.g., Age,Sex,CP,TrestBps,Chol), and submit the form to get a prediction.

Alternatively, use Docker:

Pull the Docker image:
```bash
docker pull gogetama/heartdiseaseprediction
```
Run the Docker container:
```bash
docker run -p 5000:5000 gogetama/heartdiseaseprediction
```
The Flask app will start at http://127.0.0.1:5000/.

### API Usage
Alternatively, you can use the API for predictions by sending a POST request to /predict.

Example:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"data":"age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal"}'
```
The system will return a prediction on whether Heart Disease is predicted or not.

## Input Parameters
The system expects the following input parameters (comma-separated or via JSON):

- Age: Age of the individual.
- Sex: Gender of the individual.
- Cp: Chest pain type.
- Trestbps: Resting blood pressure (in mm Hg).
- Chol: Serum cholesterol in mg/dl.
- Fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
- Restecg: Resting electrocardiographic results.
- Thalach: Maximum heart rate achieved.
- Exang: Exercise induced angina (1 = yes; 0 = no).
- Oldpeak: ST depression induced by exercise relative to rest.
- Slope: The slope of the peak exercise ST segment.
- Ca: Number of major vessels (0–3) colored by fluoroscopy.
- Thal: Thalassemia type.

## API Endpoints
1. ```/``` (Home)
- Method: GET
- Description: Displays the homepage where users can input their health data for Heart Disease prediction.
2. ```/predict``` (Prediction)
- Method: POST
- Description: Takes input data via JSON and returns a Heart Disease prediction result.
- Request Payload:
```json
{
  "data": "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal"
}
```
- Response: A prediction on whether the individual has heart disease.

## ML Model
- Algorithms:
    - CatBoost: A gradient boosting algorithm used for building the model.
    - XGBoost: Another gradient boosting algorithm for comparison.
    - Model Training: The model was trained using health and hormone metrics relevant to PCOS diagnosis.
    - Model Serialization: The model is saved in a model.pkl file, and dill is used for model serialization.
## Model Tuning and Preprocessing
- Preprocessing:

    - The input data was normalized using StandardScaler to ensure that all features were on the same scale, which is important for algorithms like XGBoost and CatBoost.

- Hyperparameter Tuning:

    - GridSearchCV was used for hyperparameter optimization to find the best combination of parameters for the machine learning models. The grid search was applied to both CatBoost and XGBoost models to improve accuracy.
    Example:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [4, 6, 8],
    'n_estimators': [100, 200]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```
This process was essential in selecting the optimal model parameters and improving prediction performance.

## Future Enhancements
- Implement a more detailed user interface for better user experience.
- Integrate additional health metrics for improved prediction accuracy.
- Add user authentication to allow for personalized user experiences.

## References or Documentation Links
- [Flask Documentation](https://flask.palletsprojects.com/en/3.0.x/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Catboost Documentation](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier)
- [Xgboost Documentation](https://xgboost.readthedocs.io/en/stable/python/index.html#)

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Aadi1101/Heart_Disease_Prediction?tab=MIT-1-ov-file) file for details.