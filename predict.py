from numpy import diff
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

def print_summary_report(results_df):
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


    avg_actual = results_df['Actual'].mean()
    avg_guess = results_df['Guess'].mean()

    print(f"Total Movies Audited: {total}")
    print(f"Overall Error Margin (MAE): {mae:.2f} points off\n")

    print("--- ACCURACY BREAKDOWN ---")
    print(f"Bullseye (Off by 0.0 to 0.10):    {bullseye:<3} movies ({bullseye/total*100:.1f}%)")
    print(f"Very Close (Off by 0.11 to 0.30): {close:<3} movies ({close/total*100:.1f}%)")
    print(f"Acceptable (Off by 0.31 to 0.50): {fair:<3} movies ({fair/total*100:.1f}%)")
    print(f"Way Off    (Off by over 0.50):    {missed:<3} movies ({missed/total*100:.1f}%)\n")

    print("--- INTERESTING STATS ---")
    print(f"The Biggest Miss: '{worst_movie['Title']}'")
    print(f"    Actual: {worst_movie['Actual']} | AI Guessed: {worst_movie['Guess']} | Off by: {worst_movie['Difference']}")
    print(f"Average Actual Rating: {avg_actual:.2f} | Average AI Guess: {avg_guess:.2f}")

def audit_all_test_data():
    print("Loading model")
    try:
        pipeline = joblib.load('./data/movie_rating_model.pkl')
    except FileNotFoundError:
        print("Error: Could not find './data/movie_rating_model.pkl'. Run train.py first")
        return

    print("Loading + Cleaning Test Dataset")
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
    titles = test_df['Series_Title'] 


    print(f"PREDICTION: ALL {len(test_df)} TEST MOVIES")
    print()
    print(f"{'MOVIE TITLE':<45} | {'ACTUAL':<6} | {'AI GUESS':<8} | {'DIFFERENCE'}")
    print("-" * 80)

    predictions = pipeline.predict(X_test)

    results_data = []
    for i in range(len(test_df)):
        title_full = str(titles.iloc[i])
        title_short = title_full[:42] 
        actual = float(y_test.iloc[i])
        guess = round(float(predictions[i]), 2)
        diff = round(abs(actual - guess), 2)
        
        results_data.append({'Title': title_full, 'Actual': actual, 'Guess': guess, 'Difference': diff})

        print(f"{title_short:<45} |  {actual:<4} |  {guess:<6} |  {diff}")

    results_df = pd.DataFrame(results_data)

    output_filename = './data/prediction_results.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"\nFull prediction results saved to '{output_filename}'")


    print_summary_report(results_df)

if __name__ == "__main__":
    audit_all_test_data()