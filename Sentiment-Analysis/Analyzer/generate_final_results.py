from datetime import datetime
import pandas as pd

results_dir = '../Analyzer'
results_path = f'{results_dir}/output/sentiment_results.csv'
df = pd.read_csv(results_path)

sentiment_counts = df['sentiment_category'].value_counts()
positive_pct = sentiment_counts.get('positive', 0) / len(df) * 100
neutral_pct = sentiment_counts.get('neutral', 0) / len(df) * 100
negative_pct = sentiment_counts.get('negative', 0) / len(df) * 100

readme = f"""# Reddit Comments Sentiment Analysis on Ozempic

## Project Overview
This package contains the results of a sentiment analysis performed on Reddit comments about Ozempic and similar GLP-1 agonists for type 2 diabetes and weight management. The analysis was completed on {datetime.now().strftime('%Y-%m-%d')}.

## Contents

### Data Files
- `preprocessed_comments.csv`: The cleaned and preprocessed Reddit comments
- `sentiment_results.csv`: Complete sentiment analysis results with polarity and subjectivity scores

### Reports
- `report/ozempic_sentiment_analysis_report.md`: Comprehensive analysis report with detailed insights
- `report/key_findings_summary.md`: Summary of key findings for quick reference

### Visualizations
- Word clouds for positive, neutral, and negative comments
- Sentiment distribution charts
- Polarity and subjectivity analysis
- Common words analysis
- Sentiment by comment score and depth

## Key Findings Summary
- Overall sentiment is predominantly neutral to positive ({positive_pct:.1f}% positive, {neutral_pct:.1f}% neutral, {negative_pct:.1f}% negative)
- Users report significant benefits:
  - Substantial weight loss (45-90 pounds)
  - Improved A1C levels
  - Reduced insulin requirements
  - Additional health benefits like reduced inflammation
- Main concerns include:
  - Gastrointestinal side effects
  - Insurance coverage and cost issues
  - Hypoglycemia in some diabetic patients

## Methodology
The analysis used TextBlob for sentiment analysis, with polarity scores ranging from -1 (negative) to +1 (positive) and subjectivity scores from 0 (objective) to 1 (subjective).

For detailed insights, please refer to the comprehensive report in the report directory.
"""

with open(f'{results_dir}/README.md', 'w') as f:
    f.write(readme)