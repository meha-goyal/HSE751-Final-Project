# HSE751 Final Project

**Predicting 30-Day Hospital Readmission in Diabetic Patients**

**Colab notebook:** https://drive.google.com/file/d/1A24Q9D3D9CXmwkLGQ1cP2Pucnbua_x7G/view?usp=sharing

## Overview

This project builds an automated analysis pipeline that predicts whether a hospitalized diabetic patient will be readmitted within 30 days of discharge. The notebook uploads and validates the data, profiles and cleans it, runs exploratory and inferential analyses, trains five supervised classification models, and selects the best model using a primary metric (PR-AUC) that is fixed before training.

## Repository Contents

| File | Description |
|---|---|
| `README.md` | Project description, instructions, and dependencies |
| `Meha_Goyal_HSE751_FinalProject.ipynb` | Full analysis notebook with saved outputs |

## Data Selection

I am using the **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the UC Irvine Machine Learning Repository: <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>

The dataset comes from the Health Facts database (Cerner), which includes ten years of inpatient diabetic encounters at 130 US hospitals and integrated delivery networks (Strack et al., 2014). It has 101,766 encounters and 47 predictor variables. The original target variable, `readmitted`, has the values `<30`, `>30`, and `NO`. The binary target used here is 1 if `readmitted` is `<30` and 0 otherwise.

I chose this dataset because my capstone project will be on heart failure readmission prediction, so exploring a different chronic condition and its readmission factors may be helpful for my overall goal. The dataset has many clinical features across demographics and lab results, and it is large and well documented.

The predictors cover demographics (`race`, `gender`, `age`, `weight`), admission and discharge information (`admission_type_id`, `admission_source_id`, `discharge_disposition_id`, etc.), hospital stay length (`time_in_hospital`), prior-year visits, diagnoses, lab results, and diabetes medications and treatment changes. Together they give a comprehensive view of the patient's inpatient care and can provide some predictive support for readmission risk.


### Settings

All settings that change the analysis are in the first code cell of Section 1:

| Setting | Default | Meaning |
|---|---|---|
| `RANDOM_STATE` | 123 | Random seed, for reproducible results |
| `TEST_SIZE` | 0.20 | Share of patients held out for testing |
| `PRIMARY_METRIC` | `"PR-AUC"` | Metric used to choose the best model (`PR-AUC`, `ROC-AUC`, `F1`, `Recall`, `Precision`, `Accuracy`) |
| `TOP_K` | 0.10 | Alert budget: share of patients that can be followed up (used for Recall@top 10%) |
| `MISSING_THRESHOLD` | 0.50 | Columns with a larger share of missing values are dropped automatically |
| `SVM_MAX_ROWS` | 20,000 | Above this many training rows, a linear SVM replaces the slower RBF-SVM |

Invalid settings and invalid input files stop the notebook with a clear error message.

## Dependencies

All packages are preinstalled in Google Colab except `ucimlrepo`, which is only needed for the fallback download.

| Package | Version tested | Minimum |
|---|---|---|
| pandas | 3.0 | 2.0 |
| numpy | 2.4 | 1.24 |
| scipy | 1.17 | 1.10 |
| scikit-learn | 1.8 | 1.2 |
| matplotlib | 3.10 | 3.7 |
| seaborn | 0.13 | 0.12 |
| ucimlrepo | latest | only for the UCI fallback download |

## Pipeline Structure

1. **Data input and validation:** upload from the user's computer, automated profiling (size, duplicates, constant and missing columns), and validation checks with a success confirmation.
2. **Processing and exploration:** descriptive statistics, visualizations, data quality assessment, and a reusable cleaning and feature engineering function.
3. **Analytical pipeline:** inferential statistics (Welch's t-test, Mann-Whitney U, chi-square test with effect sizes), formulas in LaTeX, and five classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM).
4. **Evaluation and interpretation:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC and Recall@top 10%; best-model selection by the predefined primary metric; written interpretation.

## Data Quality Assessment

Several variables have missing values: `weight` (96.9% missing), `medical_specialty` (49.1%), `payer_code` (39.6%), and `race` (2.2%). Since `weight` is almost entirely missing, it is dropped automatically (any column with more than 50% missing values is removed). For the others, I create an `Unknown` category rather than dropping rows. In the lab-result columns, `None` means the test was not performed, so it is kept as its own category.

Patients who died or went to hospice (2,423 encounters) are excluded because they cannot be readmitted. The dataset also contains repeated patients (71,518 unique patients across 101,766 encounters), so I keep one encounter per patient (`patient_nbr`). After cleaning, 69,987 patients remain.

The integer-coded categories (`admission_type_id`, `discharge_disposition_id`, `admission_source_id`) are converted to categorical variables, and rare levels are grouped together during encoding. High-cardinality ICD-9 diagnosis codes are mapped to broad diagnostic groups (circulatory, respiratory, diabetes, digestive, injury, musculoskeletal, genitourinary, neoplasms, other). I also created `total_prior_visits`, `n_meds_prescribed`, and `n_meds_changed`.

To handle the class imbalance, I use class weights and stratified train/test splits.

Challenges with this dataset include the class imbalance (about 9% positive after cleaning), which makes accuracy meaningless since always predicting "not readmitted" scores about 91%, and the large amount of missing data, which may itself be informative. Also, only readmission to the same system within 30 days is observed, so readmissions elsewhere are not available.

## Project Overview

**Prediction problem:** at the time of discharge, predict whether a hospitalized diabetic patient will be readmitted to the hospital within 30 days, using demographics, admission and discharge information, prior-year utilization, laboratory indicators, diagnoses, and medication management.

**Target variable:** `readmitted`, converted to a binary target: 1 if `<30` days, 0 otherwise.

**Primary evaluation metric:** PR-AUC (average precision), chosen before training because it is threshold-free and focused on the rare positive class, so it is more informative than ROC-AUC under imbalance. Accuracy, precision, recall, F1, ROC-AUC, and Recall@top 10% (the share of readmissions found among the 10% of patients with the highest predicted risk) are also reported.

## Results Summary

The best model by PR-AUC is **Gradient Boosting**. A model with no skill would score 0.090 (the share of positives).

| Model | PR-AUC | ROC-AUC | Recall@top 10% |
|---|---|---|---|
| Gradient Boosting | **0.180** | 0.653 | 23.9% |
| Random Forest | 0.171 | 0.657 | 22.9% |
| Logistic Regression | 0.163 | 0.646 | 22.0% |
| SVM (linear) | 0.162 | 0.646 | 22.4% |
| Decision Tree | 0.154 | 0.624 | 23.3% |

Performance is modest: the best model has about twice the no-skill PR-AUC, and flagging the 10% highest-risk patients captures about 24% of readmissions (2.4 times better than random). The differences between models are small and come from a single train/test split. In the inferential tests, prior inpatient visits had the largest effect (Cohen's d = 0.27), and all effects were small.

## Pipeline Enhancements

1. **User upload with fallback logic:** upload, then local file, then UCI download, with file checks.
2. **Automated profiling and validation:** early, clear errors if the data is not what the pipeline expects.
3. **Data-driven cleaning in one function:** missing-value threshold instead of hard-coded columns, with logging of rows removed.
4. **New feature engineering:** diagnosis groups, age midpoints, prior-visit total, and medication counts.
5. **Adaptive conditional logic:** class weights only when the positive class is rare; linear SVM for large data.
6. **Model comparison summary:** sorted comparison table, ROC and precision-recall curves, confusion matrix, feature importance, and an auto-generated results summary.

## Limitations and Next Steps

Models use default hyperparameters and a single train/test split, so the next steps are cross-validation with confidence intervals, hyperparameter tuning, threshold selection based on follow-up capacity, and calibration checks. Keeping one encounter per patient ensures independent rows but weakens the prior-visit features. The data cover 1999–2008, so practice patterns and coding have changed, and performance should be audited by race, gender, and age before any real use.

## References

- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International*, 2014, 781670.
- UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>
