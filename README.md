# Ad Click Prediction — End-to-End Classification Study

> **Can we predict whether an internet user will click on an advertisement, and turn that prediction into actionable targeting strategy?**

This project builds a production-oriented binary classification pipeline on an advertising dataset. It goes beyond fitting a model: the goal is to answer a real business question, understand *why* users click, and surface recommendations a marketing team can act on.

---

## The Business Problem

Digital advertising operates on probability. Every time an ad is served, the platform is making a bet: *will this user engage?* Serving ads to users who won't click wastes budget. Missing users who would have clicked loses revenue. A well-calibrated click prediction model sits at the center of this trade-off — and the difference between a 0.5 decision threshold and an optimized one can be measured directly in campaign P&L.

**Dataset:** 1,000 users with demographic, behavioral, and contextual features. Binary target: `Clicked on Ad` (1 = clicked, 0 = did not click).

---

## What I Did

### 1. Exploratory Data Analysis
Rather than treating EDA as a formality, I used it to build hypotheses that drove every downstream decision.

- **Behavioral profiling by click status** — KDE plots and box plots revealed that heavy internet users who spend more time on site are *less* likely to click. This counter-intuitive finding (power users are ad-immune) became the central insight of the project.
- **Time-of-day analysis** — extracted from the raw timestamp field, click rates peak during late-night (10pm–2am) and early morning windows, dropping during standard business hours.
- **KMeans segmentation** — clustered users into three behavioral archetypes: *High Clickers* (casual, older, low internet usage), *Ad-Immune Power Users* (young, high usage, rarely click), and *Moderate Engagers* (middle ground). Click rates differ dramatically across segments (~85% vs ~10%).

### 2. Feature Engineering
Expanded from 2 raw features to 20 engineered features across four categories:

| Category | Features | Rationale |
|---|---|---|
| Temporal | Hour, day of week, `is_weekend`, `is_business_hours`, `is_late_night` | Click rates vary ~20% across hours |
| Behavioral ratios | Site concentration ratio, age × income, internet usage per age | Captures interaction effects raw features miss |
| NLP | Ad topic word count, character count, sentiment polarity, subjectivity | Ad creative characteristics extracted via TextBlob |
| Categorical encoding | Country-level click rate, city-level click rate | Smoothed target encoding on 237-country, 969-city high-cardinality fields |

The `site_concentration_ratio` feature (time on site ÷ total internet usage) turned out to be one of the strongest engineered signals — users whose site visit represents a focused, intentional session are more likely to convert.

### 3. Multi-Model Benchmarking
Evaluated 5 models under 5-fold stratified cross-validation — not a single train/test split — so comparisons are statistically reliable:

| Model | CV AUC (mean ± std) |
|---|---|
| Majority Classifier (baseline) | 0.50 |
| Logistic Regression (L2) | ~0.96 |
| Logistic Regression (L1) | ~0.96 |
| Random Forest | ~0.97 |
| XGBoost | **~0.98** |

All models inside sklearn `Pipeline` objects to prevent data leakage from preprocessing steps.

### 4. Evaluation Beyond Accuracy
Moved past accuracy and confusion matrices to metrics that matter in practice:

- **ROC-AUC and Precision-Recall curves** for both shortlisted models
- **Calibration analysis** (reliability diagram + Brier score) — Logistic Regression is better calibrated than XGBoost, which matters when probability outputs feed directly into bid pricing
- **Threshold sweep** across the full 0–1 range for both F1-maximizing and business-value-maximizing operating points

### 5. Business Cost Optimization
Built an expected-value framework to find the optimal classification threshold:

```
Expected Value = (TP × revenue_per_click) − (FP × impression_cost) − (FN × opportunity_cost)
```

The optimal threshold shifts meaningfully from the default 0.5 when false positives and false negatives carry different costs — which they always do in real ad campaigns.

### 6. SHAP Interpretability
Used SHAP (SHapley Additive exPlanations) to explain model behavior at both global and individual prediction levels:

- **Beeswarm plot** — global feature impact distribution across the test set
- **Dependence plots** — how each top feature's value maps to its SHAP contribution
- **Waterfall plots** — individual prediction explanations for a high-probability and low-probability clicker

SHAP confirmed the EDA hypotheses: daily internet usage and time on site dominate, age and income contribute meaningfully, and the engineered concentration ratio ranks above several raw features.

### 7. Inference Function
Packaged the full pipeline (feature engineering + model + threshold) into a single `predict_ad_click()` function — the kind of artifact that gets deployed, not just demonstrated.

---

## Key Findings

**1. High engagement ≠ high click probability.** Users who spend the most time online and on the site are the least likely to click ads. Targeting should prioritize reach into lower-engagement demographics, not doubling down on power users.

**2. Evening and late-night delivery outperforms business hours.** Click probability is ~20% higher during the 9pm–2am window. Shifting delivery weighting toward these hours is a low-effort, measurable optimization.

**3. Three user archetypes explain most of the variance.** Rather than treating every user as an individual, the segmentation shows three clearly distinct behavioral clusters. Suppressing the power-user segment from campaigns reduces wasted impressions without meaningful click volume loss.

**4. Threshold selection is a business decision.** The gap between the F1-optimal threshold and the expected-value-optimal threshold represents real revenue. This decision belongs jointly with marketing and finance, not just the data team.

**5. Calibration matters as much as AUC.** XGBoost achieves higher AUC, but Logistic Regression's well-calibrated probabilities make it preferable when outputs feed into bid pricing logic — an overconfident model that scores 0.9 when the true probability is 0.6 causes systematic overbidding.

---

## Repository Structure

```
├── advertising.csv                          # Source dataset
├── Ad_Click_Prediction_Industry_Grade.ipynb # Full analysis notebook (executed)
└── README.md
```

---

## Technical Stack

- **Python 3.x**
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — modeling, pipelines, evaluation
- `xgboost` — gradient boosting
- `shap` — model interpretability
- `textblob` — NLP feature extraction
- `matplotlib`, `seaborn` — visualization

---

## How to Run

```bash
git clone https://github.com/your-username/ad-click-prediction
cd ad-click-prediction
pip install -r requirements.txt
jupyter notebook Ad_Click_Prediction_Industry_Grade.ipynb
```

---

## If This Were Production

The notebook covers model training, evaluation, and inference. A production deployment would add:

1. **Feature store** — user behavioral features served in <5ms at inference time
2. **Real-time scoring API** — the `predict_ad_click()` function wrapped in a FastAPI endpoint
3. **Drift monitoring** — Population Stability Index (PSI) on the top 5 features, alerting on distribution shift
4. **Periodic retraining** — user behavior changes seasonally; the pipeline should refresh on a defined schedule
5. **A/B framework** — threshold changes validated against a holdout before full deployment

---

*Built by Mylie*  
*[Portfolio](https://mylienow.vercel.app) · [LinkedIn](https://linkedin.com/in/your-handle)*
# AdClick_Prediction_CaseStudy
