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



# CHART 4: Error vs. Runtime (Regression Plot)
def regression_plot(predictions_df, folder_name):
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=predictions_df, 
        x='Runtime (min)', 
        y='Difference', 
        scatter_kws={'alpha':0.5, 'color':'#e67e22'}, 
        line_kws={'color':'red', 'linewidth':2}
    )
    plt.title('4. Blind Spot Check: Error vs. Movie Length', fontsize=13, fontweight='bold')
    plt.xlabel('Runtime (Minutes)')
    plt.ylabel('Absolute Error (Distance from Reality)')
    plt.savefig(os.path.join(folder_name, "4_error_vs_runtime.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: 4_error_vs_runtime.png")



# NEW CHART 5: Accuracy Tiers Distribution (Pie Chart)
def pie_chart(predictions_df, folder_name):
   
    plt.figure(figsize=(8, 8))
    
    # Bucket data into bands (Removed emojis from names)
    def categorise_diff(d):
        if d <= 0.1: return 'Bullseye (<= 0.1)'
        elif d <= 0.3: return 'Close (0.11 - 0.3)'
        elif d <= 0.5: return 'Acceptable (0.31 - 0.5)'
        else: return 'Missed (> 0.5)'
    
    predictions_df['Accuracy Class'] = predictions_df['Difference'].apply(categorise_diff)
    pie_data = predictions_df['Accuracy Class'].value_counts()
    
    # Map consistent high-contrast colors (Updated keys to match text-only labels)
    color_map = {
        'Bullseye (<= 0.1)': '#2ecc71',
        'Close (0.11 - 0.3)': '#3498db',
        'Acceptable (0.31 - 0.5)': '#f1c40f',
        'Missed (> 0.5)': '#e74c3c'
    }
    plot_colors = [color_map[idx] for idx in pie_data.index]

    plt.pie(
        pie_data, 
        labels=pie_data.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=plot_colors,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    plt.title('5. Precision Breakdown: Percentage of Target Hits', fontsize=13, fontweight='bold')
    plt.savefig(os.path.join(folder_name, "5_accuracy_pie_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: 5_accuracy_pie_chart.png")



# NEW CHART 6: Feature Correlation Matrix (Heatmap)
def heatmap(predictions_df, folder_name):

    plt.figure(figsize=(10, 8))
    
    # Isolate relevant purely numeric variables to map underlying patterns
    numeric_cols = ['Actual Rating', 'AI Guess', 'Difference', 'Runtime (min)', 'Critic Score', 'Total Votes', 'Box Office Gross']
    correlation_matrix = predictions_df[numeric_cols].corr()

    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=0.5,
        vmin=-1, vmax=1
    )
    plt.title('6. Structural Correlation Matrix: Feature Interactions', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(folder_name, "6_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: 6_correlation_heatmap.png")



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
    regression_plot(predictions_df, folder_name)
    pie_chart(predictions_df, folder_name)
    heatmap(predictions_df, folder_name)

    print(f"\nDONE")


if __name__ == "__main__":
    main()
    