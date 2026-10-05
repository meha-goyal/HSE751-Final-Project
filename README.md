# HSE751-Final-Project

Predicting 30-Day Hospital Readmission in Diabetic Patients
Data Selection

For my project, I'm proposing to use the Diabetes 130-US Hospitals for Years 1999–2008 dataset from the UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

The dataset comes from the Health Facts database (Cerner), which includes ten years of inpatient diabetic encounters at 130 US hospitals and integrated delivery networks (Strack et al., 2014). The target variable is readmitted, which has the values <30, >30, NO, and the binary target will specifically be 1 is readmitted is <30 and 0 otherwise.

I chose this dataset because my capstone project will be on heart failure readmission prediction, so exploring a different chronic condition and its readmission factors may be helpful for my overall goal. This dataset has many clinical features across demographics and lab results, and the data seems to be large and overall well documented.

Specifically in terms of predictor variables, there's data on demographics (race, gender, age weight), Admission and discharge (admisson_type_id, admission_source_id, etc), the hospital stay length(time_in_hosiptal), diagnoses, lab results, and treatment summary, all of which provide a comprehensive view of the patient's inpatient care and can provide some predictive support for potential readmission risk.

Data Quality Assessment

Several variables like weight, payer_code, medical_speciality, and race have missing values. Since weight is almost entirely missing, i'll probably drop the weight column. The other ones have a missing shares, but I'll create an Unknown category rather than dropping rows because they aren't as depleted.

Patients who passed away or went to hopsice will also be excluded because they can't be readmitted.

Several patients are repeated, so I'll keep one encounter per patient_nbr.

For integer-coded categories (admission_type_id, discharge_disposition_id, admission_source_id), I plan to convert them to categorical, and merge rare codes and codes that mean the same thing. I am also planning on mapping high-cardinality ICD-9 codes to broad diagnostic groups (circulatory, respiratory, diabetes, digestive, injury, musculoskeletal, genitourinary, neoplasms, other).

To handle the class imbalance, I will use class weights and stratified splits when selecting parts of the data for training and validation.

Some challenges with this dataset include the class imbalance, which reduces the meaning of a simple accuracy value since you would score about 89% by predicting not radmitted for everyone, and the large number of missing values that may be informative based on the feature. Also, only readmission to the same system within 30 days is observed, so readmissions elsewhere are not available.

Project Planning

My proposed project problem is, at the time of discharge, predict whether a hospitalized diabetic patient will be readmitted to the hospital within 30 days, using demographics, admission and discharge information, prior-year utilization, laboratory indicators, diagnoses, and medication management.

The target variable is readmitted, suitable for binary in that if readmitted is < 30 then the value is 1, otherwise it is 0.

I am planning to use PR-AUC (Average Precision) as my evaluation metrics, because it is threshold-free and focused on the rare positive class, so it is more informative than ROC-AUC under imbalance (since the positive class is only about 11% of encounters).

UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
