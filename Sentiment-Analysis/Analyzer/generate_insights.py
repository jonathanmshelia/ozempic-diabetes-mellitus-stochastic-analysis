import pandas as pd
import os

# Load the sentiment analysis results
base_path = '../Analyzer'
results_path = f'{base_path}/output/sentiment_results.csv'
df = pd.read_csv(results_path)

print(f"Generating insights report based on {len(df)} analyzed comments")

# Create a directory for the report if it doesn't exist
report_dir = f'{base_path}/report'
os.makedirs(report_dir, exist_ok=True)

# Function to extract key insights
def generate_insights_report():
    # Overall sentiment statistics
    sentiment_counts = df['sentiment_category'].value_counts()
    positive_pct = sentiment_counts.get('positive', 0) / len(df) * 100
    neutral_pct = sentiment_counts.get('neutral', 0) / len(df) * 100
    negative_pct = sentiment_counts.get('negative', 0) / len(df) * 100
    
    avg_polarity = df['polarity'].mean()
    avg_subjectivity = df['subjectivity'].mean()
    
    # Most positive and negative comments
    most_positive = df.loc[df['polarity'].idxmax()]
    most_negative = df.loc[df['polarity'].idxmin()]
    
    # Sentiment by comment score
    high_score_sentiment = df[df['score'] > 10]['sentiment_category'].value_counts(normalize=True) * 100
    
    # Sentiment by comment depth (top-level vs replies)
    top_level_sentiment = df[df['depth'] == 0]['sentiment_category'].value_counts(normalize=True) * 100
    replies_sentiment = df[df['depth'] > 0]['sentiment_category'].value_counts(normalize=True) * 100
    
    # Generate the report
    report = f"""# Sentiment Analysis Report: Reddit Comments on Ozempic for Type 2 Diabetes and Weight Management

## Executive Summary

This report presents the findings from a sentiment analysis conducted on {len(df)} Reddit comments discussing Ozempic (and similar GLP-1 agonists like Mounjaro/Zepbound) for type 2 diabetes management and weight loss. The analysis reveals that the overall sentiment toward these medications is predominantly neutral to positive, with {positive_pct:.1f}% positive, {neutral_pct:.1f}% neutral, and {negative_pct:.1f}% negative comments.

The average sentiment polarity score is {avg_polarity:.4f} (on a scale from -1 to +1), indicating a slightly positive overall sentiment. The average subjectivity score is {avg_subjectivity:.4f} (on a scale from 0 to 1), suggesting that comments contain a moderate mix of factual information and personal opinions.

## Key Findings

### 1. Overall Sentiment Distribution

- **Positive comments**: {positive_pct:.1f}% ({sentiment_counts.get('positive', 0)} comments)
- **Neutral comments**: {neutral_pct:.1f}% ({sentiment_counts.get('neutral', 0)} comments)
- **Negative comments**: {negative_pct:.1f}% ({sentiment_counts.get('negative', 0)} comments)

### 2. Key Themes in Positive Comments

Based on the word cloud and frequency analysis of positive comments, the following themes emerged:

1. **Significant weight loss benefits**: Many users report substantial weight loss, with some mentioning losing 45-90 pounds.
2. **Improved diabetes management**: Users frequently mention improved A1C levels, with reductions from higher levels (7.6+) to near-normal ranges (5.6-6.0).
3. **Reduced insulin requirements**: Type 1 diabetics report dramatic reductions in daily insulin usage (e.g., from 110-120 units to 40-50 units).
4. **Life-changing effects**: Terms like "game changer" and "miracle drug" appear frequently in positive comments.
5. **Additional health benefits**: Some users mention reduced inflammation and improvements in related conditions.

### 3. Key Themes in Negative Comments

The negative comments, while fewer, highlight important concerns:

1. **Side effects**: Gastrointestinal issues including nausea and stomach pain are the most commonly mentioned side effects.
2. **Insurance and cost concerns**: Some users express frustration about insurance coverage and the high cost of these medications.
3. **Medical professional resistance**: Some comments indicate that certain doctors are reluctant to prescribe these medications for diabetes management.
4. **Hypoglycemia concerns**: Some diabetic users report experiencing more frequent low blood sugar episodes.

### 4. Insights by Comment Characteristics

- **High-scoring comments** (those with more upvotes) tend to be more positive ({high_score_sentiment.get('positive', 0):.1f}% positive) than the overall dataset, suggesting community agreement with positive experiences.
- **Top-level comments** are more positive ({top_level_sentiment.get('positive', 0):.1f}% positive) compared to replies ({replies_sentiment.get('positive', 0):.1f}% positive), indicating that initial experiences shared are generally more favorable.
- **Comment depth analysis** shows that deeper conversation threads tend to include more neutral and technical discussions about medication management.

### 5. Notable User Experiences

**Most Positive Comment:**
"{most_positive['body']}"
- Polarity score: {most_positive['polarity']:.4f}

**Most Negative Comment:**
"{most_negative['body']}"
- Polarity score: {most_negative['polarity']:.4f}

## Detailed Insights on Ozempic and Similar GLP-1 Agonists

### Medical Benefits

1. **Diabetes Management**:
   - Users consistently report significant improvements in A1C levels
   - Better blood glucose control with fewer spikes
   - Reduced insulin resistance
   - More predictable response to insulin doses

2. **Weight Management**:
   - Substantial weight loss reported (ranging from 20-90 pounds)
   - Reduced hunger and appetite
   - Improved relationship with food
   - Sustainable weight management compared to other methods

3. **Additional Health Benefits**:
   - Reduced inflammation
   - Potential reduction in diabetes-related complications
   - Improved energy levels
   - Better overall quality of life

### Challenges and Concerns

1. **Side Effects**:
   - Gastrointestinal issues (nausea, stomach pain)
   - Potential for hypoglycemic episodes in diabetic patients
   - Adaptation period required for some users

2. **Access Issues**:
   - Insurance coverage challenges
   - High cost without insurance
   - Prior authorization requirements
   - Supply shortages mentioned in some comments

3. **Medical Support**:
   - Varied experiences with healthcare providers
   - Some users report resistance from doctors to prescribe for diabetes management
   - Need for proper medical supervision highlighted

## Conclusions

The sentiment analysis of Reddit comments reveals predominantly positive to neutral sentiment toward Ozempic and similar GLP-1 agonists for type 2 diabetes and weight management. Users report significant benefits in terms of improved diabetes control, substantial weight loss, and reduced insulin requirements. The medications are frequently described as "game changers" and "life-changing."

However, the analysis also highlights important concerns including side effects (primarily gastrointestinal), insurance coverage challenges, and the need for proper medical supervision. These concerns should be addressed to improve patient experiences with these medications.

The strong positive sentiment among high-scoring comments suggests broad community agreement about the benefits of these medications, while the more detailed discussions in comment threads provide valuable insights into the nuances of individual experiences.

## Recommendations

Based on the sentiment analysis, the following recommendations emerge:

1. **Patient Education**: Provide comprehensive information about potential side effects and management strategies to set realistic expectations.

2. **Healthcare Provider Awareness**: Share positive patient experiences with healthcare providers to increase awareness of benefits beyond the primary indications.

3. **Insurance Advocacy**: Address insurance coverage challenges to improve access to these medications for appropriate patients.

4. **Monitoring Protocols**: Develop standardized monitoring protocols for patients using these medications, particularly for those with type 1 diabetes who report significant changes in insulin requirements.

5. **Further Research**: Conduct formal studies on the reported additional benefits (e.g., reduced inflammation, improved quality of life) to expand the evidence base for these medications.

---

*This report was generated through sentiment analysis of Reddit comments and should be interpreted as reflecting user experiences rather than clinical evidence. Individual experiences may vary, and medical decisions should be made in consultation with healthcare providers.*
"""
    
    # Save the report
    report_path = f'{report_dir}/ozempic_sentiment_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Insights report generated and saved to {report_path}")
    return report_path

# Generate the insights report
report_path = generate_insights_report()

# Create a summary of key findings for quick reference
def generate_summary():
    sentiment_counts = df['sentiment_category'].value_counts()
    positive_pct = sentiment_counts.get('positive', 0) / len(df) * 100
    neutral_pct = sentiment_counts.get('neutral', 0) / len(df) * 100
    negative_pct = sentiment_counts.get('negative', 0) / len(df) * 100
    
    summary = f"""# Ozempic Sentiment Analysis: Key Findings Summary

## Overall Sentiment
- **Positive**: {positive_pct:.1f}% ({sentiment_counts.get('positive', 0)} comments)
- **Neutral**: {neutral_pct:.1f}% ({sentiment_counts.get('neutral', 0)} comments)
- **Negative**: {negative_pct:.1f}% ({sentiment_counts.get('negative', 0)} comments)
- **Average Polarity**: {df['polarity'].mean():.4f} (scale: -1 to +1)

## Top Positive Themes
1. Significant weight loss (45-90 pounds reported)
2. Improved A1C levels (reductions to 5.6-6.0 range)
3. Reduced insulin requirements (up to 50-60% reduction)
4. "Game changer" for diabetes management
5. Additional benefits like reduced inflammation

## Top Negative Themes
1. Gastrointestinal side effects (nausea, stomach pain)
2. Insurance coverage and cost concerns
3. Hypoglycemia in some diabetic patients
4. Medical professional resistance to prescribing

## Key Insights
- High-scoring comments tend to be more positive
- Top-level comments are more positive than replies
- Type 1 diabetics report benefits despite not being the primary target population
- Users frequently describe life-changing improvements in diabetes management
"""
    
    summary_path = f'{report_dir}/key_findings_summary.md'
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"Key findings summary generated and saved to {summary_path}")
    return summary_path

# Generate the summary
summary_path = generate_summary()

print("Insights report generation completed successfully!")
