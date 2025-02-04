# Social Media Usage and Emotional Well-being Analysis 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FLAML](https://img.shields.io/badge/FLAML-1.2.2-orange.svg)](https://microsoft.github.io/FLAML/)

## Project Overview 📊

Machine learning analysis of social media usage patterns and their correlation with emotional well-being, using AutoML to predict emotional states from engagement metrics. Analysis spans major social platforms with 11,000+ user interactions.

## Data Overview 📈

### Dataset Characteristics
- Records: 11,000+ user interactions
- Features: 10 engagement metrics
- Target: 6 emotional states (Happiness, Sadness, Anger, Anxiety, Boredom, Neutral)
- Platforms: Facebook, Instagram, Twitter, LinkedIn, Snapchat, WhatsApp, Telegram

### Feature Categories
- Usage Metrics: Daily time, post frequency, message volume
- Engagement Metrics: Likes, comments, shares
- User Demographics: Age, gender, platform preference

## Key Findings 🔍

### Platform Analysis
Visual-focused platforms (Instagram, Snapchat) show 35% happiness rates but significant anxiety (25%), driven by comparison behavior. Text-heavy platforms (Twitter, LinkedIn) exhibit higher negative emotion rates (32% anger, 28% sadness). Users spending >120 minutes daily on Twitter show 2.3x higher negative emotions versus <60-minute users.

### Usage Patterns
Moderate users (90 minutes/day) show optimal emotional states with 42% reporting happiness. Heavy users (180+ minutes/day) experience 1.8x more emotional volatility and 35% anxiety rates. Active engagement through direct messages correlates with 2.1x higher positive emotions versus passive scrolling.

### Age Demographics
Younger users (18-24) average 160 minutes daily, primarily on Instagram, with 38% reporting anxiety. Middle-aged users (35-44) spend 70 minutes daily, mainly on Facebook, showing lower anxiety (18%) and higher happiness (32%).

## Methodology 🔬

### Data Processing Pipeline
```python
def process_data(df):
    # Clean and validate
    df = clean_dataframe(df)
    
    # Engineer features
    df = add_engagement_metrics(df)
    df = create_time_features(df)
    
    return df
```

### Model Architecture
```python
settings = {
    "time_budget": 60,
    "metric": 'roc_auc_ovo',
    "task": 'classification'
}

automl = AutoML()
automl.fit(X_train, y_train, **settings)
```

## Performance Metrics 📊

### Classification Results
```
Metric              Score
Accuracy           0.842
ROC AUC           0.912
Macro F1          0.839
CV Accuracy       0.835 ±0.018
```

### Feature Importance
```
Feature                  Impact
Daily Usage Time        1.000
Platform Type           0.876
Engagement Style        0.754
Age Group              0.721
```

## Installation & Usage 💻

### Requirements
```
python>=3.8
flaml==1.2.2
pandas==1.5.3
scikit-learn==1.0.2
matplotlib==3.7.1
seaborn==0.12.2
```
## Citation 📚
```bibtex
@misc{bulut2024social,
  author = {BULUT, Emirhan},
  title = {Social Media Usage and Emotional Well-Being Dataset},
  year = {2024},
  publisher = {Kaggle}
}
```
