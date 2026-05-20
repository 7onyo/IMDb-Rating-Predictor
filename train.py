# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib  


print("Loading data")
df = pd.read_csv('./data/imdb_top_1000.xls')


print("Cleaning data")
df['Gross'] = df['Gross'].astype(str).str.replace(',', '')
df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce') 

df['Runtime'] = df['Runtime'].astype(str).str.replace(' min', '')
df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce') 

df['Meta_score'] = df['Meta_score'].fillna(df['Meta_score'].median())
df['Gross'] = df['Gross'].fillna(df['Gross'].median())
df = df.dropna(subset=['IMDB_Rating'])


director_counts = df['Director'].value_counts()
df['Director'] = df['Director'].apply(lambda x: 'Other' if director_counts.get(x, 0) < 3 else x)



features = ['Director', 'Genre', 'Gross', 'No_of_Votes', 'Meta_score', 'Runtime']
X = df[features]
y = df['IMDB_Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



numeric_features = ['Gross', 'No_of_Votes', 'Meta_score', 'Runtime']
categorical_features = ['Director', 'Genre']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) 
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
])


print("Training model")
pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model trained successfully (MAE: {mae:.2f})")


print("Saving model")
joblib.dump(pipeline, './data/movie_rating_model.pkl')
print("Saved as './data/movie_rating_model.pkl'")