import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm

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

# Function to perform sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Return polarity score (-1 to 1)
    return analysis.sentiment.polarity

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
def preprocess_data(df):
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
    
    # Add sentiment analysis
    df['sentiment_score'] = df['cleaned_text'].apply(analyze_sentiment)
    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
    
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
    "Model 3": {
        "baseline_hba1c": 8.3,
        "baseline_variability": 0.95,
        "treatment_effect_mean": 1.6,
        "treatment_effect_sd": 0.12,
        "weight_loss_mean": 4.8,
        "weight_loss_sd": 1.6,
        "baseline_weight": 93.5,
        "baseline_weight_sd": 22.5
    },
    "Model 4": {
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

def sentiment_analysis_app():
    st.title("Comment Sentiment Analyzer")
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Preprocess data
        with st.spinner("Processing data..."):
            processed_df, deleted_count = preprocess_data(df)
        
        # Display basic information
        st.header("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Comments", len(processed_df))
        with col2:
            st.metric("Deleted Comments", deleted_count)
        with col3:
            st.metric("Avg. Tokens per Comment", f"{processed_df['token_count'].mean():.2f}")
        
        # Data explorer
        st.header("Data Explorer")
        if st.checkbox("Show sample data"):
            st.dataframe(processed_df[['author', 'cleaned_text', 'created_datetime', 'sentiment_score', 'sentiment_category']].head(10))
        
        # Sentiment Analysis
        st.header("Sentiment Analysis")
        
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
                st.metric("Token Count", comment['token_count'])
                st.metric("Created Date", comment['created_datetime'].strftime('%Y-%m-%d %H:%M:%S'))
    
    else:
        # Display instructions when no file is uploaded
        st.info("Please upload a CSV file to begin analysis.")
        st.markdown("""
        ### Expected CSV Format:
        The CSV should contain Reddit comments with at least the following columns:
        - `id`: Comment ID
        - `author`: Username of the commenter
        - `body`: The comment text
        - `created_utc`: Timestamp in UTC
        - Additional columns like `score`, `parent_id`, etc. are also supported
        """)

def monte_carlo_simulation_app():
    st.title("Monte Carlo Simulation for Clinical Outcomes")
    st.write("This simulation is based on SUSTAIN Trial results and allows you to explore potential outcomes for HbA1c reduction and weight loss.")
    
    # Sidebar for simulation parameters
    st.sidebar.header("Simulation Parameters")
    
    # Model selection
    model_selection = st.sidebar.radio(
        "Choose a Model or Custom Parameters",
        ["Model 1 (SUSTAIN 1 Trial)", "Model 2 (SUSTAIN 2 Trial)", "Model 3", "Model 4", "Custom Parameters"]
    )
    
    # Basic parameters
    n_simulations = st.sidebar.slider("Number of Simulations", 1000, 10000, 5000, 1000)
    
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
        
        # Use preset parameters
        baseline_hba1c = selected_model['baseline_hba1c']
        baseline_variability = selected_model['baseline_variability']
        treatment_effect_mean = selected_model['treatment_effect_mean']
        treatment_effect_sd = selected_model['treatment_effect_sd']
        weight_loss_mean = selected_model['weight_loss_mean']
        weight_loss_sd = selected_model['weight_loss_sd']
        baseline_weight = selected_model['baseline_weight']
        baseline_weight_sd = selected_model['baseline_weight_sd']
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
    
    # Run simulation button
    run_simulation = st.sidebar.button("Run Simulation")
    
    # Advanced options
    st.sidebar.subheader("Advanced Options")
    show_sensitivity = st.sidebar.checkbox("Run Sensitivity Analysis")
    show_convergence = st.sidebar.checkbox("Test Convergence")
    
    if run_simulation:
        with st.spinner("Running Monte Carlo simulation..."):
            # Run main simulation
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
            
            results = run_monte_carlo_simulation(**baseline_params)
            
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
            
            # Run sensitivity analysis if requested
            if show_sensitivity:
                st.header("Sensitivity Analysis")
                
                with st.spinner("Running sensitivity analysis..."):
                    # Define parameter ranges for sensitivity analysis
                    param_ranges = {
                        'treatment_effect_mean': [treatment_effect_mean - 0.5, treatment_effect_mean - 0.25, 
                                                 treatment_effect_mean, treatment_effect_mean + 0.25, 
                                                 treatment_effect_mean + 0.5],
                        'weight_loss_mean': [weight_loss_mean - 2.0, weight_loss_mean - 1.0, 
                                            weight_loss_mean, weight_loss_mean + 1.0, 
                                            weight_loss_mean + 2.0],
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
    
    else:
        # Display instructions when simulation hasn't been run
        st.info("Select a preset model or customize parameters, then click 'Run Simulation' to see results.")
        st.markdown("""
        ### About This Simulation
        
        This Monte Carlo simulation models clinical outcomes based on SUSTAIN Trial results. It allows you to:
        
        1. **Choose from Four Preset Models**:
           - Model 1 (SUSTAIN 1 Trial): The original trial parameters
           - Model 2 (SUSTAIN 2 Trial): Parameters from the second trial
           - Model 3: Alternative parameter set with higher treatment effect
           - Model 4: Alternative parameter set with higher weight loss
        
        2. **Or Customize Your Own Parameters**:
           - Baseline HbA1c and variability
           - Treatment effect size and variability
           - Weight loss parameters
           - Baseline weight parameters
        
        3. **Analyze Key Outcomes**:
           - HbA1c Reduction distribution
           - Percentage of patients reaching target HbA1c <7%
           - Weight loss distribution
        
        4. **Run Advanced Analyses** (optional):
           - Sensitivity Analysis: Test how changes in parameters affect outcomes
           - Convergence Testing: Verify the stability of simulation results with different sample sizes
        
        Select a model or customize parameters, then click "Run Simulation" to see the results.
        """)

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
