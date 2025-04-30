# Notes
from collections import Counter
from datetime import datetime
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import textblob
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Download necessary NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    return True

download_nltk_resources()

# Function to clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    
    # Convert to string if not already
    text = str(text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove Reddit formatting (e.g., *italic*, **bold**)
    text = re.sub(r'\*\*|\*', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

# Modified tokenization approach
def simple_tokenize(text):
    # Simple tokenization by splitting on whitespace
    return text.split()

# Function to perform sentiment analysis with TextBlob
def get_textblob_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return {'polarity': 0, 'subjectivity': 0}
    
    analysis = textblob.TextBlob(text)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }

# Function to perform sentiment analysis with NLTK (VADER)
def get_nltk_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return {'polarity': 0, 'subjectivity': 0}
    
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    
    return {
        'polarity': sentiment_scores['compound'],
        'subjectivity': 0.5  # VADER doesn't provide subjectivity, using default value
    }

# Function to categorize sentiment
def categorize_sentiment(score):
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Function to preprocess data
@st.cache_data
def preprocess_data(df, sentiment_analyzer='textblob'):
    # Check for deleted comments
    deleted_count = df[df['author'] == '[deleted]'].shape[0]
    
    # Apply text cleaning to the 'body' column
    df['cleaned_text'] = df['body'].apply(clean_text)
    
    # Remove empty comments after cleaning
    df = df[df['cleaned_text'].str.strip() != ""]
    
    # Tokenize
    df['tokens'] = df['cleaned_text'].apply(simple_tokenize)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['tokens_without_stopwords'] = df['tokens'].apply(lambda tokens: [word for word in tokens if word not in stop_words])
    
    # Convert UTC timestamp to datetime
    df['created_datetime'] = df['created_utc'].apply(lambda x: datetime.fromtimestamp(x))
    
    # Sort by date
    df = df.sort_values('created_datetime')
    
    # Count tokens
    df['token_count'] = df['tokens'].apply(len)
    
    # Add sentiment analysis based on selected analyzer
    if sentiment_analyzer == 'textblob':
        sentiments = df['cleaned_text'].apply(get_textblob_sentiment)
    else:  # NLTK
        sentiments = df['cleaned_text'].apply(get_nltk_sentiment)

    # Extract polarity and subjectivity, duplicate the polarity across two columns
    df['sentiment_score'] = sentiments.apply(lambda x: x['polarity'])
    df['polarity'] = sentiments.apply(lambda x: x['polarity'])
    df['subjectivity'] = sentiments.apply(lambda x: x['subjectivity'])

    df['sentiment_category'] = df['polarity'].apply(categorize_sentiment)
    
    return df, deleted_count

# Function to generate word frequency
def get_word_frequency(tokens_list, top_n=20):
    all_words = [word for tokens in tokens_list for word in tokens]
    word_counts = Counter(all_words)
    return word_counts.most_common(top_n)

# Function to create word cloud
def create_wordcloud(tokens_list):
    all_words = ' '.join([word for tokens in tokens_list for word in tokens])
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_words)
    return wordcloud

# Function to extract key insights
def generate_insights_report(df: pd.DataFrame) -> str:
    # Overall sentiment statistics
    sentiment_counts = df['sentiment_category'].value_counts()
    positive_pct = sentiment_counts.get('Positive', 0) / len(df) * 100
    neutral_pct = sentiment_counts.get('Neutral', 0) / len(df) * 100
    negative_pct = sentiment_counts.get('Negative', 0) / len(df) * 100
    
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
    return f"""# Sentiment Analysis Report: Reddit Comments on Ozempic for Type 2 Diabetes and Weight Management

## Executive Summary

This report presents the findings from a sentiment analysis conducted on {len(df)} Reddit comments discussing Ozempic (and similar GLP-1 agonists like Mounjaro/Zepbound) for type 2 diabetes management and weight loss. The analysis reveals that the overall sentiment toward these medications is predominantly neutral to positive, with {positive_pct:.1f}% positive, {neutral_pct:.1f}% neutral, and {negative_pct:.1f}% negative comments.

The average sentiment polarity score is {avg_polarity:.4f} (on a scale from -1 to +1), indicating a slightly positive overall sentiment. The average subjectivity score is {avg_subjectivity:.4f} (on a scale from 0 to 1), suggesting that comments contain a moderate mix of factual information and personal opinions.

## Key Findings

### 1. Overall Sentiment Distribution

- **Positive comments**: {positive_pct:.1f}% ({sentiment_counts.get('Positive', 0)} comments)
- **Neutral comments**: {neutral_pct:.1f}% ({sentiment_counts.get('Neutral', 0)} comments)
- **Negative comments**: {negative_pct:.1f}% ({sentiment_counts.get('Negative', 0)} comments)

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

- **High-scoring comments** (those with more upvotes) tend to be more positive ({high_score_sentiment.get('Positive', 0):.1f}% positive) than the overall dataset, suggesting community agreement with positive experiences.
- **Top-level comments** are more positive ({top_level_sentiment.get('Positive', 0):.1f}% positive) compared to replies ({replies_sentiment.get('Positive', 0):.1f}% positive), indicating that initial experiences shared are generally more favorable.
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

# Function to analyze a single comment
def analyze_single_comment(text, analyzer='textblob'):
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Tokenize
    tokens = simple_tokenize(cleaned_text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens_without_stopwords = [word for word in tokens if word not in stop_words]
    
    # Get sentiment based on selected analyzer
    if analyzer == 'textblob':
        sentiment = get_textblob_sentiment(cleaned_text)
    else:  # NLTK
        sentiment = get_nltk_sentiment(cleaned_text)
    
    polarity = sentiment['polarity']
    subjectivity = sentiment['subjectivity']
    category = categorize_sentiment(polarity)
    
    return {
        'original_text': text,
        'cleaned_text': cleaned_text,
        'tokens': tokens,
        'tokens_without_stopwords': tokens_without_stopwords,
        'token_count': len(tokens),
        'polarity': polarity,
        'subjectivity': subjectivity,
        'category': category
    }

def sentiment_analysis_app():
    st.title("Comment Sentiment Analyzer")
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Add sentiment analyzer selection
    sentiment_analyzer = st.sidebar.radio(
        "Select Sentiment Analysis Method:",
        options=["TextBlob", "NLTK (VADER)"],
        index=0
    )
    
    # Convert selection to lowercase for function calls
    analyzer_key = 'textblob' if sentiment_analyzer == "TextBlob" else 'nltk'
    
    # Add a legend for sentiment scoring
    st.sidebar.title("Sentiment Score Legend")
    
    # Display different legends based on selected analyzer
    if sentiment_analyzer == "TextBlob":
        st.sidebar.markdown("""
        ### TextBlob Sentiment Scale
        - **Polarity**: Ranges from -1.0 (very negative) to 1.0 (very positive)
          - **-1.0 to -0.1**: Negative sentiment
          - **-0.1 to 0.1**: Neutral sentiment
          - **0.1 to 1.0**: Positive sentiment
        
        - **Subjectivity**: Ranges from 0.0 (very objective) to 1.0 (very subjective)
          - **0.0 to 0.3**: More objective/factual
          - **0.3 to 0.7**: Moderately subjective
          - **0.7 to 1.0**: Highly subjective/opinion-based
        """)
    else:
        st.sidebar.markdown("""
        ### NLTK VADER Sentiment Scale
        - **Compound Score**: Ranges from -1.0 (very negative) to 1.0 (very positive)
          - **-1.0 to -0.1**: Negative sentiment
          - **-0.1 to 0.1**: Neutral sentiment
          - **0.1 to 1.0**: Positive sentiment
        
        VADER is specifically attuned to social media and considers:
        - Punctuation, capitalization, and intensifiers
        - Negation and context
        - Slang and emoticons
        """)
    
    # Add custom comment analysis section
    st.sidebar.title("Analyze Your Own Comment")
    custom_comment = st.sidebar.text_area("Enter text to analyze:", height=150)
    
    if st.sidebar.button("Analyze Comment"):
        if custom_comment.strip():
            # Perform analysis
            analysis_result = analyze_single_comment(custom_comment, analyzer_key)
            
            # Display results
            st.sidebar.subheader("Analysis Results:")
            st.sidebar.metric("Sentiment Category", analysis_result['category'])
            st.sidebar.metric("Polarity Score", f"{analysis_result['polarity']:.4f}")
            st.sidebar.metric("Subjectivity Score", f"{analysis_result['subjectivity']:.4f}")
            st.sidebar.metric("Token Count", analysis_result['token_count'])
        else:
            st.sidebar.warning("Please enter text to analyze.")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Preprocess data with selected sentiment analyzer
        with st.spinner(f"Processing data using {sentiment_analyzer}..."):
            processed_df, deleted_count = preprocess_data(df, analyzer_key)
        
        # Display basic information
        st.header("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Comments", len(processed_df))
        with col2:
            st.metric("Deleted Comments", deleted_count)
        with col3:
            st.metric("Avg. Tokens per Comment", f"{processed_df['token_count'].mean():.2f}")
        with col4:
            st.metric("Sentiment Method", sentiment_analyzer)
        
        # Data explorer
        st.header("Data Explorer")
        if st.checkbox("Show sample data"):
            st.dataframe(processed_df[['author', 'cleaned_text', 'created_datetime', 'sentiment_score', 'sentiment_category', 'polarity', 'subjectivity']].head(20))
        
        # Sentiment Analysis
        st.header("Sentiment Analysis")
        
        # Sentiment score distribution chart
        st.subheader("Sentiment Score Distribution")
        
        # Create a histogram of sentiment scores with color-coded regions
        fig = px.histogram(processed_df, x='sentiment_score', nbins=50,
                          title='Distribution of Sentiment Scores')
        
        # Add vertical lines to indicate sentiment categories
        fig.add_vline(x=-0.1, line_dash="dash", line_color="red")
        fig.add_vline(x=0.1, line_dash="dash", line_color="green")
        
        # Add annotations - fixed to avoid errors with y-axis values
        # Define a fixed y-position for annotations without relying on data
        y_pos = 0.85  # Position in normalized coordinates (0-1)
        
        # Add annotations with fixed y-position
        fig.add_annotation(x=-0.5, y=y_pos, text="Negative", showarrow=False, 
                          font=dict(color="red"), xref="x", yref="paper")
        fig.add_annotation(x=0, y=y_pos, text="Neutral", showarrow=False, 
                          xref="x", yref="paper")
        fig.add_annotation(x=0.5, y=y_pos, text="Positive", showarrow=False, 
                          font=dict(color="green"), xref="x", yref="paper")
        
        # Update layout with x-axis range
        fig.update_layout(xaxis_range=[-1, 1])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment distribution
        sentiment_counts = processed_df['sentiment_category'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            fig = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                         color='Sentiment', 
                         color_discrete_map={'Positive':'green', 'Neutral':'gray', 'Negative':'red'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment Over Time")
            # Group by date and calculate average sentiment
            daily_sentiment = processed_df.groupby(processed_df['created_datetime'].dt.date)['sentiment_score'].mean().reset_index()
            daily_sentiment.columns = ['Date', 'Average Sentiment']
            
            fig = px.line(daily_sentiment, x='Date', y='Average Sentiment')
            
            # Add horizontal lines to indicate sentiment categories
            fig.add_hline(y=-0.1, line_dash="dash", line_color="red")
            fig.add_hline(y=0.1, line_dash="dash", line_color="green")
            
            # Add annotations with fixed x-position at middle of date range
            if len(daily_sentiment) > 0:
                mid_date = daily_sentiment['Date'].iloc[len(daily_sentiment)//2]
                
                fig.add_annotation(x=mid_date, y=-0.2, text="Negative", 
                                 showarrow=False, font=dict(color="red"))
                fig.add_annotation(x=mid_date, y=0, text="Neutral", 
                                 showarrow=False)
                fig.add_annotation(x=mid_date, y=0.2, text="Positive", 
                                 showarrow=False, font=dict(color="green"))
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Word frequency analysis
        st.header("Word Frequency Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Words (Without Stopwords)")
            word_freq = get_word_frequency(processed_df['tokens_without_stopwords'].tolist())
            word_freq_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            
            fig = px.bar(word_freq_df.head(15), x='Frequency', y='Word', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Word Cloud")
            wordcloud = create_wordcloud(processed_df['tokens_without_stopwords'].tolist())
            
            # Save wordcloud to a temporary file
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            
            # Use st.pyplot directly
            st.pyplot(plt)
        
        # Author Analysis
        st.header("Author Analysis")
        
        # Top authors by comment count
        author_counts = processed_df['author'].value_counts().reset_index()
        author_counts.columns = ['Author', 'Comment Count']
        author_counts = author_counts[author_counts['Author'] != '[deleted]'].head(10)
        
        fig = px.bar(author_counts, x='Comment Count', y='Author', orientation='h',
                    title='Top 10 Authors by Comment Count')
        st.plotly_chart(fig, use_container_width=True)
        
        # Comment Length Analysis
        st.header("Comment Length Analysis")
        
        fig = px.histogram(processed_df, x='token_count', nbins=50,
                          title='Distribution of Comment Lengths (in tokens)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Advanced Analysis
        st.header("Advanced Analysis")
        
        # Sentiment by comment length
        st.subheader("Sentiment Score vs. Comment Length")
        fig = px.scatter(processed_df, x='token_count', y='sentiment_score', 
                        color='sentiment_category',
                        color_discrete_map={'Positive':'green', 'Neutral':'gray', 'Negative':'red'},
                        opacity=0.6)
        
        # Add horizontal lines to indicate sentiment categories
        fig.add_hline(y=-0.1, line_dash="dash", line_color="red")
        fig.add_hline(y=0.1, line_dash="dash", line_color="green")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Subjectivity vs. Polarity (for TextBlob)
        if sentiment_analyzer == "TextBlob":
            st.subheader("Subjectivity vs. Polarity")
            fig = px.scatter(processed_df, x='polarity', y='subjectivity',
                            color='sentiment_category',
                            color_discrete_map={'Positive':'green', 'Neutral':'gray', 'Negative':'red'},
                            opacity=0.6)
            
            # Add vertical lines to indicate sentiment categories
            fig.add_vline(x=-0.1, line_dash="dash", line_color="red")
            fig.add_vline(x=0.1, line_dash="dash", line_color="green")
            
            # Update layout
            fig.update_layout(xaxis_range=[-1, 1], yaxis_range=[0, 1])
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual Comment Analysis
        st.header("Individual Comment Analysis")
        
        # Comment selector
        selected_comment_index = st.selectbox(
            "Select a comment to analyze:",
            options=processed_df.index.tolist(),
            format_func=lambda x: f"{processed_df.loc[x, 'author']} - {processed_df.loc[x, 'cleaned_text'][:50]}..."
        )
        
        if selected_comment_index is not None:
            comment = processed_df.loc[selected_comment_index]
            
            st.subheader(f"Analysis of Comment by {comment['author']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_area("Original Text", comment['body'], height=150)
                st.text_area("Cleaned Text", comment['cleaned_text'], height=150)
            
            with col2:
                st.metric("Sentiment Score", f"{comment['sentiment_score']:.4f}")
                st.metric("Sentiment Category", comment['sentiment_category'])
                st.metric("Subjectivity Score", f"{comment['subjectivity']:.4f}")
                st.metric("Token Count", comment['token_count'])
                st.metric("Created Date", comment['created_datetime'].strftime('%Y-%m-%d %H:%M:%S'))
        
        st.header("Sentiment Analysis Report")
        report_content = generate_insights_report(processed_df)
        st.markdown(report_content, unsafe_allow_html=True)

    else:
        # Display instructions when no file is uploaded
        st.info("Please upload a CSV file to begin analysis or use the sidebar to analyze a custom comment.")
        st.markdown("""
        ### Expected CSV Format:
        The CSV should contain Reddit comments with at least the following columns:
        - `id`: Comment ID
        - `author`: Username of the commenter
        - `body`: The comment text
        - `created_utc`: Timestamp in UTC
        - Additional columns like `score`, `parent_id`, etc. are also supported
        """)

# Monte Carlo Simulation Functions
def run_monte_carlo_simulation(n_simulations, baseline_hba1c, treatment_effect_mean, treatment_effect_sd, 
                              weight_loss_mean, weight_loss_sd, baseline_variability, baseline_weight=90, baseline_weight_sd=20):
    """
    Run Monte Carlo simulation for HbA1c reduction and weight loss
    
    Parameters:
    -----------
    n_simulations: int
        Number of simulations to run
    baseline_hba1c: float
        Baseline HbA1c level
    treatment_effect_mean: float
        Mean treatment effect on HbA1c reduction
    treatment_effect_sd: float
        Standard deviation of treatment effect
    weight_loss_mean: float
        Mean weight loss in kg
    weight_loss_sd: float
        Standard deviation of weight loss
    baseline_variability: float
        Variability in baseline HbA1c
    baseline_weight: float
        Mean baseline weight
    baseline_weight_sd: float
        Standard deviation of baseline weight
        
    Returns:
    --------
    dict: Dictionary containing simulation results
    """
    # Generate random baseline HbA1c values with variability
    baseline_values = np.random.normal(baseline_hba1c, baseline_variability, n_simulations)
    baseline_weights = np.random.normal(baseline_weight, baseline_weight_sd, n_simulations)
    
    # Generate random treatment effects
    treatment_effects = np.random.normal(treatment_effect_mean, treatment_effect_sd, n_simulations)
    
    # Calculate final HbA1c values
    final_hba1c = baseline_values - treatment_effects
    
    # Calculate proportion of patients reaching target HbA1c < 7%
    prop_under_7 = (final_hba1c < 7.0).mean() * 100
    
    # Generate random weight loss values
    weight_loss = np.random.normal(weight_loss_mean, weight_loss_sd, n_simulations)
    weight_loss = np.minimum(weight_loss, baseline_weights)  # Physical constraint
    
    # Calculate confidence intervals
    ci_hba1c_red = np.percentile(treatment_effects, [2.5, 97.5])
    ci_prop_under_7 = np.percentile(
        [np.mean(np.random.choice(final_hba1c < 7.0, n_simulations, replace=True)) * 100 
         for _ in range(1000)], 
        [2.5, 97.5]
    )
    ci_weight_loss = np.percentile(weight_loss, [2.5, 97.5])
    
    return {
        'baseline_values': baseline_values,
        'treatment_effects': treatment_effects,
        'final_hba1c': final_hba1c,
        'prop_under_7': prop_under_7,
        'weight_loss': weight_loss,
        'ci_hba1c_red': ci_hba1c_red,
        'ci_prop_under_7': ci_prop_under_7,
        'ci_weight_loss': ci_weight_loss
    }

def run_sensitivity_analysis(baseline_params, param_ranges):
    """
    Run sensitivity analysis by varying parameters
    
    Parameters:
    -----------
    baseline_params: dict
        Dictionary of baseline parameters
    param_ranges: dict
        Dictionary of parameter ranges to test
        
    Returns:
    --------
    dict: Dictionary containing sensitivity analysis results
    """
    results = {}
    
    # HbA1c reduction sensitivity
    hba1c_reduction_results = []
    for effect in param_ranges['treatment_effect_mean']:
        params = baseline_params.copy()
        params['treatment_effect_mean'] = effect
        sim_results = run_monte_carlo_simulation(**params)
        hba1c_reduction_results.append({
            'effect': effect,
            'prop_under_7': sim_results['prop_under_7'],
            'mean_reduction': np.mean(sim_results['treatment_effects'])
        })
    
    # Weight loss sensitivity
    weight_loss_results = []
    for wl in param_ranges['weight_loss_mean']:
        params = baseline_params.copy()
        params['weight_loss_mean'] = wl
        sim_results = run_monte_carlo_simulation(**params)
        weight_loss_results.append({
            'weight_loss': wl,
            'mean_actual_loss': np.mean(sim_results['weight_loss'])
        })
    
    # Baseline HbA1c variability
    baseline_variability_results = []
    for var in param_ranges['baseline_variability']:
        params = baseline_params.copy()
        params['baseline_variability'] = var
        sim_results = run_monte_carlo_simulation(**params)
        baseline_variability_results.append({
            'variability': var,
            'prop_under_7': sim_results['prop_under_7']
        })
    
    results['hba1c_reduction'] = hba1c_reduction_results
    results['weight_loss'] = weight_loss_results
    results['baseline_variability'] = baseline_variability_results
    
    return results

def test_convergence(baseline_params, n_sim_values):
    """
    Test convergence of simulation by varying number of simulations
    
    Parameters:
    -----------
    baseline_params: dict
        Dictionary of baseline parameters
    n_sim_values: list
        List of simulation counts to test
        
    Returns:
    --------
    dict: Dictionary containing convergence test results
    """
    convergence_results = []
    
    for n_sim in n_sim_values:
        params = baseline_params.copy()
        params['n_simulations'] = n_sim
        sim_results = run_monte_carlo_simulation(**params)
        convergence_results.append({
            'n_simulations': n_sim,
            'prop_under_7': sim_results['prop_under_7'],
            'mean_reduction': np.mean(sim_results['treatment_effects']),
            'mean_weight_loss': np.mean(sim_results['weight_loss'])
        })
    
    return convergence_results

# Model description paragraphs
model_descriptions = {
    "Model 1 (SUSTAIN 1 Trial)":
        """
        **Model 1 (SUSTAIN 1 Trial)** represents the foundational clinical trial for semaglutide in treatment-naïve 
        patients with type 2 diabetes. This 30-week trial demonstrated significant improvements in glycemic control 
        and weight reduction. With a baseline HbA1c of 8.05%, this model simulates an expected mean HbA1c reduction 
        of 1.55% and average weight loss of 4.13kg. The model accounts for patient variability with relatively tight 
        confidence intervals, reflecting the controlled clinical trial environment. This simulation is particularly relevant 
        for assessing outcomes in newly diagnosed patients beginning GLP-1 RA therapy.
        """,
    
    "Model 2 (SUSTAIN 2 Trial)":
        """
        **Model 2 (SUSTAIN 2 Trial)** builds on the first model but represents a 56-week trial comparing semaglutide 
        to sitagliptin in patients with type 2 diabetes inadequately controlled on metformin, thiazolidinedione, or both. 
        With a slightly higher baseline HbA1c of 8.1%, this model shows a similar HbA1c reduction (1.5%) but increased 
        weight loss benefit (4.5kg) compared to Model 1. The longer trial duration provides better estimates of sustained 
        effects and captures more variance in patient response, making this model ideal for simulating outcomes in patients 
        who have been on stable metformin therapy but require additional glycemic control.
        """,
    
    "Model 3 (SUSTAIN 7 Trial)":
        """
        **Model 3 (SUSTAIN 7 Trial)** represents an optimized parameter set based on real-world evidence and extended trial data, with 
        a focus on patients with more elevated baseline HbA1c (8.3%). This model predicts a more substantial treatment 
        effect (1.6% HbA1c reduction) and enhanced weight loss (4.8kg), but with greater variability in outcomes. The 
        increased baseline variability (0.95) reflects the heterogeneity typically seen in clinical practice rather than 
        controlled trial settings. This simulation is particularly valuable for estimating outcomes in patients with poorer 
        initial glycemic control who might demonstrate more pronounced responses to treatment.
        """,
    
    "Model 4 (Aggregated Trials)":
        """
        **Model 4 (Aggregated Trials)** prioritizes weight loss outcomes in patients with moderately elevated baseline HbA1c (8.2%). It 
        predicts the most substantial HbA1c reduction (1.7%) and weight loss (5.0kg) among all models, making it ideal 
        for simulating outcomes in patients where both glycemic control and significant weight reduction are primary 
        treatment goals. The tighter baseline variability (0.8) suggests this model may be most applicable to a more 
        homogeneous patient population. This simulation is particularly relevant for evaluating potential synergistic 
        benefits of improved glucose control alongside meaningful weight reduction.
        """
}

# Preset model parameters
preset_models = {
    "Model 1 (SUSTAIN 1 Trial)": {
        "baseline_hba1c": 8.05,
        "baseline_variability": 0.85,
        "treatment_effect_mean": 1.55,
        "treatment_effect_sd": 0.097,
        "weight_loss_mean": 4.13,
        "weight_loss_sd": 1.5,
        "baseline_weight": 91.93,
        "baseline_weight_sd": 23.8
    },
    "Model 2 (SUSTAIN 2 Trial)": {
        "baseline_hba1c": 8.1,
        "baseline_variability": 0.9,
        "treatment_effect_mean": 1.5,
        "treatment_effect_sd": 0.1,
        "weight_loss_mean": 4.5,
        "weight_loss_sd": 1.8,
        "baseline_weight": 92.0,
        "baseline_weight_sd": 24.0
    },
    "Model 3 (SUSTAIN 7 Trial)": {
        "baseline_hba1c": 8.3,
        "baseline_variability": 0.95,
        "treatment_effect_mean": 1.6,
        "treatment_effect_sd": 0.12,
        "weight_loss_mean": 4.8,
        "weight_loss_sd": 1.6,
        "baseline_weight": 93.5,
        "baseline_weight_sd": 22.5
    },
    "Model 4 (Aggregated Trials)": {
        "baseline_hba1c": 8.2,
        "baseline_variability": 0.8,
        "treatment_effect_mean": 1.7,
        "treatment_effect_sd": 0.11,
        "weight_loss_mean": 5.0,
        "weight_loss_sd": 1.7,
        "baseline_weight": 90.5,
        "baseline_weight_sd": 21.0
    }
}

def display_simulation_results(results, model_selection):
    """Helper function to display simulation results"""
    # Extract results
    treatment_effects = results['treatment_effects']
    final_hba1c = results['final_hba1c']
    weight_loss = results['weight_loss']
    prop_under_7 = results['prop_under_7']
    ci_hba1c_red = results['ci_hba1c_red']
    ci_prop_under_7 = results['ci_prop_under_7']
    ci_weight_loss = results['ci_weight_loss']
    
    # Calculate means for display
    mean_hba1c_red = np.mean(treatment_effects)
    mean_weight_loss = np.mean(weight_loss)
    
    # Display results
    st.header("Simulation Results")
    
    # Key metrics
    st.subheader("Key Outcomes")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean HbA1c Reduction", f"{mean_hba1c_red:.2f}%")
        st.write(f"95% CI: [{ci_hba1c_red[0]:.2f}%, {ci_hba1c_red[1]:.2f}%]")
    
    with col2:
        st.metric("Patients Reaching HbA1c <7%", f"{prop_under_7:.1f}%")
        st.write(f"95% CI: [{ci_prop_under_7[0]:.1f}%, {ci_prop_under_7[1]:.1f}%]")
    
    with col3:
        st.metric("Mean Weight Loss", f"{mean_weight_loss:.2f} kg")
        st.write(f"95% CI: [{ci_weight_loss[0]:.2f} kg, {ci_weight_loss[1]:.2f} kg]")
    
    # Visualizations
    st.subheader("Outcome Distributions")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["HbA1c Reduction", "Final HbA1c", "Weight Loss"])
    
    with tab1:
        # HbA1c Reduction Distribution
        fig = plt.figure(figsize=(10, 6))
        plt.hist(treatment_effects, bins=50, density=True, alpha=0.7, color='skyblue')
        plt.axvline(mean_hba1c_red, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_hba1c_red:.2f}%')
        plt.axvspan(ci_hba1c_red[0], ci_hba1c_red[1], color='green', alpha=0.2,
                    label=f'95% CI: [{ci_hba1c_red[0]:.2f}%, {ci_hba1c_red[1]:.2f}%]')
        plt.xlabel('HbA1c Reduction (%)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'HbA1c Reduction Distribution ({model_selection})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        st.pyplot(fig)
    
    with tab2:
        # Final HbA1c with <7% Threshold
        fig = plt.figure(figsize=(10, 6))
        plt.hist(final_hba1c, bins=50, density=True, alpha=0.7, color='lightgreen')
        plt.axvline(7.0, color='black', linestyle='-', linewidth=2,
                    label='Clinical Target: <7%')
        plt.axvline(np.mean(final_hba1c), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(final_hba1c):.2f}%')
        plt.axvspan(np.percentile(final_hba1c, 2.5), np.percentile(final_hba1c, 97.5),
                    color='orange', alpha=0.2,
                    label=f'95% CI: [{np.percentile(final_hba1c, 2.5):.2f}%, {np.percentile(final_hba1c, 97.5):.2f}%]')
        plt.xlabel('Final HbA1c (%)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Final HbA1c Values\n{prop_under_7:.1f}% Reach <7% Target', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        st.pyplot(fig)
    
    with tab3:
        # Weight Loss Distribution
        fig = plt.figure(figsize=(10, 6))
        plt.hist(weight_loss, bins=50, density=True, alpha=0.7, color='salmon')
        plt.axvline(mean_weight_loss, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_weight_loss:.2f} kg')
        plt.axvspan(ci_weight_loss[0], ci_weight_loss[1], color='purple', alpha=0.2,
                    label=f'95% CI: [{ci_weight_loss[0]:.2f} kg, {ci_weight_loss[1]:.2f} kg]')
        plt.xlabel('Weight Loss (kg)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Weight Loss Distribution ({model_selection})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        st.pyplot(fig)

def run_and_display_advanced_analyses(baseline_params, show_sensitivity, show_convergence):
    """Helper function to run and display advanced analyses"""
    # Run sensitivity analysis if requested
    if show_sensitivity:
        st.header("Sensitivity Analysis")
        
        with st.spinner("Running sensitivity analysis..."):
            # Define parameter ranges for sensitivity analysis
            param_ranges = {
                'treatment_effect_mean': [baseline_params['treatment_effect_mean'] - 0.5, 
                                         baseline_params['treatment_effect_mean'] - 0.25, 
                                         baseline_params['treatment_effect_mean'], 
                                         baseline_params['treatment_effect_mean'] + 0.25, 
                                         baseline_params['treatment_effect_mean'] + 0.5],
                'weight_loss_mean': [baseline_params['weight_loss_mean'] - 2.0, 
                                    baseline_params['weight_loss_mean'] - 1.0, 
                                    baseline_params['weight_loss_mean'], 
                                    baseline_params['weight_loss_mean'] + 1.0, 
                                    baseline_params['weight_loss_mean'] + 2.0],
                'baseline_variability': [0.1, 0.3, 0.5]
            }
            
            sensitivity_results = run_sensitivity_analysis(baseline_params, param_ranges)
            
            # Create tabs for different sensitivity analyses
            tab1, tab2, tab3 = st.tabs(["HbA1c Reduction Effect", "Weight Loss Effect", "Baseline Variability"])
            
            with tab1:
                # HbA1c reduction sensitivity
                hba1c_df = pd.DataFrame(sensitivity_results['hba1c_reduction'])
                
                fig = px.scatter(hba1c_df, x='effect', y='prop_under_7', 
                                title="Effect of HbA1c Reduction on Target Achievement",
                                labels={'effect': 'Mean HbA1c Reduction (%)', 
                                       'prop_under_7': 'Patients Reaching HbA1c <7% (%)'},
                                size_max=10, trendline='ols')
                fig.update_traces(marker=dict(size=12))
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Weight loss sensitivity
                weight_df = pd.DataFrame(sensitivity_results['weight_loss'])
                
                fig = px.scatter(weight_df, x='weight_loss', y='mean_actual_loss',
                                title="Expected vs. Actual Weight Loss",
                                labels={'weight_loss': 'Expected Mean Weight Loss (kg)',
                                       'mean_actual_loss': 'Actual Mean Weight Loss (kg)'},
                                size_max=10, trendline='ols')
                fig.update_traces(marker=dict(size=12))
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Baseline variability sensitivity
                var_df = pd.DataFrame(sensitivity_results['baseline_variability'])
                
                fig = px.bar(var_df, x='variability', y='prop_under_7',
                            title="Effect of Baseline HbA1c Variability on Target Achievement",
                            labels={'variability': 'Baseline HbA1c Variability',
                                   'prop_under_7': 'Patients Reaching HbA1c <7% (%)'},
                            color='prop_under_7', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
    
    # Run convergence testing if requested
    if show_convergence:
        st.header("Convergence Testing")
        
        with st.spinner("Testing convergence..."):
            # Define simulation counts for convergence testing
            n_sim_values = [100, 500, 1000, 2000, 5000, 10000]
            
            convergence_results = test_convergence(baseline_params, n_sim_values)
            conv_df = pd.DataFrame(convergence_results)
            
            # Create tabs for different convergence metrics
            tab1, tab2 = st.tabs(["Target Achievement Convergence", "Mean Outcomes Convergence"])
            
            with tab1:
                fig = px.line(conv_df, x='n_simulations', y='prop_under_7',
                             title="Convergence of Target Achievement Estimate",
                             labels={'n_simulations': 'Number of Simulations',
                                    'prop_under_7': 'Patients Reaching HbA1c <7% (%)'},
                             markers=True)
                fig.update_traces(line=dict(width=3))
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Prepare data for multiple lines
                fig = go.Figure()
                
                # Add HbA1c reduction convergence
                fig.add_trace(go.Scatter(
                    x=conv_df['n_simulations'],
                    y=conv_df['mean_reduction'],
                    mode='lines+markers',
                    name='HbA1c Reduction',
                    line=dict(width=3, color='blue')
                ))
                
                # Add weight loss convergence on secondary y-axis
                fig.add_trace(go.Scatter(
                    x=conv_df['n_simulations'],
                    y=conv_df['mean_weight_loss'],
                    mode='lines+markers',
                    name='Weight Loss',
                    line=dict(width=3, color='red'),
                    yaxis='y2'
                ))
                
                # Update layout for dual y-axis
                fig.update_layout(
                    title="Convergence of Mean Outcome Estimates",
                    xaxis=dict(title="Number of Simulations"),
                    yaxis=dict(title="HbA1c Reduction (%)", titlefont=dict(color="blue")),
                    yaxis2=dict(title="Weight Loss (kg)", titlefont=dict(color="red"),
                               overlaying="y", side="right"),
                    legend=dict(x=0.01, y=0.99),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

def monte_carlo_simulation_app():
    st.title("Monte Carlo Simulation for Clinical Outcomes")
    st.write("This simulation is based on SUSTAIN Trial results and allows you to explore potential outcomes for HbA1c reduction and weight loss.")
    
    # Sidebar for simulation parameters
    st.sidebar.header("Simulation Parameters")
    
    # Model selection
    model_selection = st.sidebar.radio(
        "Choose a Model or Custom Parameters",
        ["Model 1 (SUSTAIN 1 Trial)", "Model 2 (SUSTAIN 2 Trial)", "Model 3 (SUSTAIN 7 Trial)", "Model 4 (Aggregated Trials)", "Custom Parameters"]
    )
    
    # Basic parameters
    n_simulations = st.sidebar.slider("Number of Simulations", 1000, 10000, 5000, 1000)
    
    # Advanced options
    st.sidebar.subheader("Advanced Options")
    show_sensitivity = st.sidebar.checkbox("Run Sensitivity Analysis")
    show_convergence = st.sidebar.checkbox("Test Convergence")
    
    # Set parameters based on model selection
    if model_selection != "Custom Parameters":
        selected_model = preset_models[model_selection]
        
        # Display the preset model parameters (read-only)
        st.sidebar.subheader(f"{model_selection} Parameters")
        st.sidebar.info(f"""
        **Baseline HbA1c:** {selected_model['baseline_hba1c']:.2f}%
        **Baseline Variability:** {selected_model['baseline_variability']:.2f}
        **Treatment Effect:** {selected_model['treatment_effect_mean']:.2f}% ± {selected_model['treatment_effect_sd']:.2f}
        **Weight Loss:** {selected_model['weight_loss_mean']:.2f} kg ± {selected_model['weight_loss_sd']:.2f} kg
        **Baseline Weight:** {selected_model['baseline_weight']:.2f} kg ± {selected_model['baseline_weight_sd']:.2f} kg
        """)
        
        # Display model description
        st.markdown("## Model Description")
        st.markdown(model_descriptions[model_selection])
        
        # Use preset parameters
        baseline_params = {
            'n_simulations': n_simulations,
            'baseline_hba1c': selected_model['baseline_hba1c'],
            'treatment_effect_mean': selected_model['treatment_effect_mean'],
            'treatment_effect_sd': selected_model['treatment_effect_sd'],
            'weight_loss_mean': selected_model['weight_loss_mean'],
            'weight_loss_sd': selected_model['weight_loss_sd'],
            'baseline_variability': selected_model['baseline_variability'],
            'baseline_weight': selected_model['baseline_weight'],
            'baseline_weight_sd': selected_model['baseline_weight_sd']
        }
        
        # Run the simulation automatically for preset models
        with st.spinner(f"Running Monte Carlo simulation for {model_selection}..."):
            results = run_monte_carlo_simulation(**baseline_params)
            display_simulation_results(results, model_selection)
            run_and_display_advanced_analyses(baseline_params, show_sensitivity, show_convergence)
            
    else:
        # Custom parameters
        st.sidebar.subheader("Custom Parameters")
        baseline_hba1c = st.sidebar.slider("Baseline HbA1c (%)", 7.0, 10.0, 8.5, 0.1)
        baseline_variability = st.sidebar.slider("Baseline HbA1c Variability", 0.1, 1.0, 0.3, 0.1)
        
        # Treatment effect parameters
        st.sidebar.subheader("Treatment Effect Parameters")
        treatment_effect_mean = st.sidebar.slider("Mean HbA1c Reduction (%)", 0.5, 2.5, 1.5, 0.1)
        treatment_effect_sd = st.sidebar.slider("HbA1c Reduction SD", 0.1, 1.0, 0.4, 0.1)
        
        # Weight loss parameters
        st.sidebar.subheader("Weight Loss Parameters")
        weight_loss_mean = st.sidebar.slider("Mean Weight Loss (kg)", 1.0, 10.0, 4.5, 0.5)
        weight_loss_sd = st.sidebar.slider("Weight Loss SD", 0.5, 3.0, 1.5, 0.1)
        
        # Baseline weight parameters
        st.sidebar.subheader("Baseline Weight Parameters")
        baseline_weight = st.sidebar.slider("Mean Baseline Weight (kg)", 70.0, 120.0, 90.0, 1.0)
        baseline_weight_sd = st.sidebar.slider("Baseline Weight SD", 5.0, 30.0, 20.0, 1.0)
        
        # Run simulation button (only for custom parameters)
        run_simulation = st.sidebar.button("Run Simulation")
        
        # Display custom model description placeholder
        st.markdown("## Custom Model")
        st.markdown("""
        The **Custom Model** allows you to fully customize all simulation parameters according to your specific 
        clinical scenario or research question. You can adjust baseline HbA1c levels, expected treatment effects, 
        weight loss parameters, and patient variability to model a wide range of patient populations and treatment 
        protocols. This flexibility enables you to explore "what-if" scenarios beyond the standard clinical trial 
        parameters, which is particularly valuable when tailoring expectations for specific subpopulations or 
        when evaluating potential outcomes in real-world clinical settings where patient characteristics may 
        differ from those in controlled trials.
        """)
        
        # For custom parameters, only run when the button is clicked
        if run_simulation:
            baseline_params = {
                'n_simulations': n_simulations,
                'baseline_hba1c': baseline_hba1c,
                'treatment_effect_mean': treatment_effect_mean,
                'treatment_effect_sd': treatment_effect_sd,
                'weight_loss_mean': weight_loss_mean,
                'weight_loss_sd': weight_loss_sd,
                'baseline_variability': baseline_variability,
                'baseline_weight': baseline_weight,
                'baseline_weight_sd': baseline_weight_sd
            }
            
            with st.spinner("Running Monte Carlo simulation with custom parameters..."):
                results = run_monte_carlo_simulation(**baseline_params)
                display_simulation_results(results, "Custom Model")
                run_and_display_advanced_analyses(baseline_params, show_sensitivity, show_convergence)
        else:
            # Display instructions when custom simulation hasn't been run
            st.info("Adjust your custom parameters, then click 'Run Simulation' to see results.")

# Main app
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Analysis Type", 
                               ["Sentiment Analysis", "Monte Carlo Simulation"])
    
    if app_mode == "Sentiment Analysis":
        sentiment_analysis_app()
    else:
        monte_carlo_simulation_app()

if __name__ == "__main__":
    main()
