# IMDb Movie Rating Predictor

This project uses a Random Forest Regressor to predict IMDb movie ratings based on various features such as director, genre, box office gross, and critic scores. It includes a full pipeline for data cleaning, model training, performance auditing, and visual analysis.

## Features

- **Automated Data Cleaning:** Handles missing values, formatting (e.g., Gross and Runtime strings), and consolidates rare categories.
- **Machine Learning Pipeline:** Utilizes Scikit-learn's `Pipeline` with `ColumnTransformer` for seamless preprocessing and model application.
- **Detailed Auditing:** Generates comprehensive reports on model accuracy, identifying "Bullseyes" and "Biggest Misses."
- **Visual Insights:** Produces 6 distinct charts to visualize model performance and feature importance.

## Dataset

The data used in this project is the **IMDb Top 1000 Movies** dataset, which can be found on Kaggle:
[IMDb Dataset of Top 1000 Movies and TV Shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)

## Project Structure

- `train.py`: The entry point for training the model. Saves the trained model as `data/movie_rating_model.pkl`.
- `predict.py`: Loads the trained model to perform an audit on the test dataset. Generates `data/detailed_predictions.csv` and `data/ai_feature_drivers.csv`.
- `visualize.py`: Generates PNG charts in the `graphs/` directory based on prediction results.
- `data/`: Directory containing the source dataset, trained models, and generated prediction CSVs.
- `graphs/`: Directory containing generated visualization charts.
- `requirements.txt`: Python dependencies.

## Installation

1. Clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Model
Run the training script to clean the data and train the Random Forest model.
```bash
python train.py
```
This will output the Mean Absolute Error (MAE) and save `data/movie_rating_model.pkl`.

### 2. Run Predictions & Audit
Analyze the model's performance on a test split of the data.
```bash
python predict.py
```
This generates:
- `data/detailed_predictions.csv`: Row-by-row comparison of actual vs. predicted ratings.
- `data/ai_feature_drivers.csv`: A list of features ranked by their influence on the model.
- A summary report in the terminal.

### 3. Generate Visualizations
Create high-resolution charts to analyze errors and decision factors.
```bash
python visualize.py
```
Charts are saved to the `graphs/` folder:
1. **Actual vs. Predicted**: Overall accuracy overview.
2. **Error Distribution**: Shows if the AI tends to over or under-predict.
3. **Top 10 Decision Factors**: What features the AI relies on most.
4. **Error vs. Runtime**: Identifies if movie length affects prediction accuracy.
5. **Accuracy Tiers**: Pie chart breakdown of precision.
6. **Correlation Heatmap**: Interactions between different numeric features.

## Model Details

- **Algorithm:** Random Forest Regressor (200 estimators, max depth 10).
- **Features Used:**
    - **Categorical:** Director, Genre.
    - **Numeric:** Gross, No_of_Votes, Meta_score, Runtime.
- **Preprocessing:** Standard scaling for numeric data and One-Hot Encoding for categorical data.

<!-- ## Sample Visualizations
*(Generated in the `graphs/` directory)*
- `1_actual_vs_predicted.png`
- `3_top_decision_factors.png`
- `5_accuracy_pie_chart.png` -->
