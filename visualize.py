import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CHART 1: Actual vs. Predicted (Scatter Plot)
def scatter_plot(predictions_df, folder_name):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
            data=predictions_df, 
            x='Actual Rating', 
            y='AI Guess', 
            alpha=0.7, 
            color='#2ecc71',
            edgecolor='black',
            s=60
        )
    min_val = min(predictions_df['Actual Rating'].min(), predictions_df['AI Guess'].min())
    max_val = max(predictions_df['Actual Rating'].max(), predictions_df['AI Guess'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction Line')
    
    plt.title('1. Accuracy Overview: AI Guess vs Reality', fontsize=13, fontweight='bold')
    plt.xlabel('Real-World IMDb Rating')
    plt.ylabel('AI Predicted Rating')
    plt.legend()
    plt.savefig(os.path.join(folder_name, "1_actual_vs_predicted.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: 1_actual_vs_predicted.png")



# CHART 2: The Error Distribution (Histogram)
def histogram(predictions_df, folder_name):
    plt.figure(figsize=(8, 6))
    sns.histplot(data=predictions_df, x='Raw Error', bins=20, kde=True, color='#3498db')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error (Bullseye)')

    plt.title('2. Error Distribution: Over vs Under Predicting', fontsize=13, fontweight='bold')
    plt.xlabel('Error Margin (AI Guess - Actual Rating)')
    plt.ylabel('Number of Movies')
    plt.legend()
    plt.savefig(os.path.join(folder_name, "2_error_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: 2_error_distribution.png")




# CHART 3: Top 10 Decision Factors (Horizontal Bar)
def decision_factors_chart(features_df, folder_name):
    plt.figure(figsize=(10, 6))
    top_features = features_df.head(10)
    sns.barplot(
        data=top_features, 
        x='Importance', 
        y='Feature', 
        palette='magma',
        hue='Feature',
        legend=False
    )
    plt.title("3. The AI's Brain: Top 10 Decision Drivers", fontsize=13, fontweight='bold')
    plt.xlabel('Importance Weight (%)')
    plt.ylabel('Feature Columns')
    plt.savefig(os.path.join(folder_name, "3_top_decision_factors.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: 3_top_decision_factors.png")

def main():
    print("Loading data from CSV files")
    
    if not os.path.exists('./data/detailed_predictions.csv') or not os.path.exists('./data/ai_feature_drivers.csv'):
        print("Error: Could not find your CSV files. Run predict.py first")
        return

    predictions_df = pd.read_csv('./data/detailed_predictions.csv')
    features_df = pd.read_csv('./data/ai_feature_drivers.csv')

    predictions_df['Raw Error'] = predictions_df['AI Guess'] - predictions_df['Actual Rating']

    folder_name = "graphs"
    os.makedirs(folder_name, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    print("Generating charts")
    print(f"Saving standard images inside '{folder_name}/'\n")

    scatter_plot(predictions_df, folder_name)
    histogram(predictions_df, folder_name)
    decision_factors_chart(features_df, folder_name)

    print(f"\nDONE")


if __name__ == "__main__":
    main()
    