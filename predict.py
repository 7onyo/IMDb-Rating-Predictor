import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

def audit_all_test_data():
    print("Loading model")
    try:
        pipeline = joblib.load('movie_rating_model.pkl')
    except FileNotFoundError:
        print("Error: Could not find 'movie_rating_model.pkl'. Run train.py first")
        return

    print("Loading + Cleaning Test Dataset")
    df = pd.read_csv('imdb_top_1000.xls')

    df['Gross'] = df['Gross'].astype(str).str.replace(',', '')
    df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce') 
    df['Runtime'] = df['Runtime'].astype(str).str.replace(' min', '')
    df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce') 
    
    df['Meta_score'] = df['Meta_score'].fillna(df['Meta_score'].median())
    df['Gross'] = df['Gross'].fillna(df['Gross'].median())
    df = df.dropna(subset=['IMDB_Rating'])

    director_counts = df['Director'].value_counts()
    df['Director'] = df['Director'].apply(lambda x: 'Other' if director_counts.get(x, 0) < 3 else x)

    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    features = ['Director', 'Genre', 'Gross', 'No_of_Votes', 'Meta_score', 'Runtime']
    X_test = test_df[features]
    y_test = test_df['IMDB_Rating']
    titles = test_df['Series_Title'] 


    print(f"PREDICTION: ALL {len(test_df)} TEST MOVIES")
    print()
    print(f"{'MOVIE TITLE':<45} | {'ACTUAL':<6} | {'AI GUESS':<8} | {'DIFFERENCE'}")
    print("-" * 80)

    predictions = pipeline.predict(X_test)

    differences = []
    for i in range(len(test_df)):
        title = str(titles.iloc[i])[:42] 
        actual = y_test.iloc[i]
        guess = round(predictions[i], 2)
        diff = round(abs(actual - guess), 2)
        differences.append(diff)
        
        print(f"{title:<45} |  {actual:<4} |  {guess:<6} |  {diff}")

    print()
    avg_diff = sum(differences) / len(differences)
    print(f"Total Movies Audited: {len(test_df)}")
    print(f"Average Error Margin (MAE): {avg_diff:.2f} points off")


if __name__ == "__main__":
    audit_all_test_data()