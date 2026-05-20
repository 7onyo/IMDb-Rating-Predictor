import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

def export_feature_importance(pipeline):
    """Extracts what the AI cares about most and saves it to a file."""
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']

    numeric_features = ['Gross', 'No_of_Votes', 'Meta_score', 'Runtime']
    categorical_features = ['Director', 'Genre']
    
    encoded_features = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(encoded_features)

    importances = regressor.feature_importances_
    importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
    
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    filename = './data/ai_feature_drivers.csv'
    importance_df.to_csv(filename, index=False)
    print(f"AI's top decision drivers saved to '{filename}'")
    
    return importance_df


def print_summary_report(results_df, importance_df):
    """Generates a detailed statistical breakdown of the AI's performance."""
    print()
    print("AI PREDICTION SUMMARY REPORT")
    print()

    total = len(results_df)
    mae = results_df['Difference'].mean()

    bullseye = len(results_df[results_df['Difference'] <= 0.1])
    close = len(results_df[(results_df['Difference'] > 0.1) & (results_df['Difference'] <= 0.3)])
    fair = len(results_df[(results_df['Difference'] > 0.3) & (results_df['Difference'] <= 0.5)])
    missed = len(results_df[results_df['Difference'] > 0.5])

    worst_idx = results_df['Difference'].idxmax()
    worst_movie = results_df.loc[worst_idx]

    best_idx = results_df['Difference'].idxmin()
    best_movie = results_df.loc[best_idx]

    avg_actual = results_df['Actual Rating'].mean()
    avg_guess = results_df['AI Guess'].mean()

    print(f"Total Movies Audited: {total}")
    print(f"Overall Error Margin (MAE): {mae:.2f} points off\n")

    print("--- ACCURACY BREAKDOWN ---")
    print(f"Bullseye (Off by 0.0 to 0.10):    {bullseye:<3} movies ({bullseye/total*100:.1f}%)")
    print(f"Close (Off by 0.11 to 0.30): {close:<3} movies ({close/total*100:.1f}%)")
    print(f"Acceptable (Off by 0.31 to 0.50): {fair:<3} movies ({fair/total*100:.1f}%)")
    print(f"Missed    (Off by over 0.50):    {missed:<3} movies ({missed/total*100:.1f}%)\n")

    print("--- TOP 5 DRIVERS (What the AI cares about) ---")
    for i in range(5):
        feature = importance_df.iloc[i]['Feature']
        weight = importance_df.iloc[i]['Importance'] * 100
        print(f" {i+1}. {feature}: {weight:.1f}%")
    print()

    print("--- STATS ---")
    print(f"The Best Guess:  '{best_movie['Title']}'")
    print(f"    Actual: {best_movie['Actual Rating']} | AI Guessed: {best_movie['AI Guess']} | Off by: {best_movie['Difference']}")
    
    print(f"The Biggest Miss: '{worst_movie['Title']}'")
    print(f"    Actual: {worst_movie['Actual Rating']} | AI Guessed: {worst_movie['AI Guess']} | Off by: {worst_movie['Difference']}")
    print(f"Average Actual Rating: {avg_actual:.2f} | Average AI Guess: {avg_guess:.2f}")
    

def audit_all_test_data():
    print("Loading model")
    try:
        pipeline = joblib.load('./data/movie_rating_model.pkl')
    except FileNotFoundError:
        print("Error: Could not find './data/movie_rating_model.pkl'. Run train.py first")
        return

    print("Loading and preparing original dataset")
    df = pd.read_csv('./data/imdb_top_1000.xls')

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


    print("Making predictions on the test split")
    predictions = pipeline.predict(X_test)

    results_data = []

    for i in range(len(test_df)):
        actual = float(y_test.iloc[i])
        guess = round(float(predictions[i]), 2)
        diff = round(abs(actual - guess), 2)
        
        results_data.append({
            'Title': str(test_df.iloc[i]['Series_Title']),
            'Actual Rating': actual,
            'AI Guess': guess,
            'Difference': diff,
            'Director': test_df.iloc[i]['Director'],
            'Genre': test_df.iloc[i]['Genre'],
            'Runtime (min)': test_df.iloc[i]['Runtime'],
            'Critic Score': test_df.iloc[i]['Meta_score'],
            'Total Votes': test_df.iloc[i]['No_of_Votes'],
            'Box Office Gross': test_df.iloc[i]['Gross']
        })

    results_df = pd.DataFrame(results_data)
    print("Saving data to CSV")
    
    movies_filename = './data/detailed_predictions.csv'
    results_df.to_csv(movies_filename, index=False)
    print(f"Detailed movie predictions saved to '{movies_filename}'")
    
    importance_df = export_feature_importance(pipeline)

    print_summary_report(results_df, importance_df)

if __name__ == "__main__":
    audit_all_test_data()