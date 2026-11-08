# Salifort Motors: Employee Attrition Analysis

A capstone project that analyzes HR survey data from Salifort Motors, finds what drives employees to leave, and builds models that predict attrition.

## Business question

What is likely to make an employee leave the company, and can we predict who will?

## Dataset

`HR_capstone_dataset.csv`: 15,000 rows and 10 columns, including satisfaction level, last evaluation score, number of projects, average monthly hours, tenure, work accident, promotion in the last 5 years, department, and salary. The target is `left`. Source: [Kaggle HR Analytics](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction).

## Approach

1. **Cleaning:** renamed columns to snake_case, dropped 3,008 duplicate rows (leaving 11,991), checked missing values and tenure outliers.
2. **EDA:** plots of hours, projects, satisfaction, tenure, salary, promotion and department against attrition (about 17% of employees left).
3. **Encoding:** `salary` as an ordinal variable, `department` one-hot encoded.
4. **Modeling:** decision tree and random forest tuned with 4-fold `GridSearchCV`, refit on AUC, evaluated on a stratified 25% test set.
5. **Round 2:** dropped `satisfaction_level` (possible data leakage) and replaced monthly hours with an `overworked` flag (more than 175 hours per month).

## Results (test set)

| Round | Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| 1 (all features) | Decision tree | 0.936 | 0.914 | 0.925 | 0.978 |
| 1 (all features) | Random forest | 0.964 | 0.918 | 0.940 | 0.983 |
| 2 (no satisfaction) | Decision tree | 0.784 | 0.918 | 0.846 | 0.952 |
| 2 (no satisfaction) | Random forest | 0.870 | 0.904 | 0.887 | 0.968 |

## Key findings

- Employees with 2 projects (54% left) or 6 to 7 projects (45% for 6, and all with 7) leave most often.
- Many leavers worked very long hours with low satisfaction, and a second group worked fewer hours than average and was also dissatisfied.
- Attrition peaks at 5 years of tenure (45%), and nobody with 7 or more years left.
- Lower salary and no recent promotion go with higher attrition. Department makes little difference.

## Recommendations

- Cap the number of projects per employee.
- Investigate why employees at 4 to 5 years of tenure leave, and offer clearer promotion paths.
- Be clear about overtime expectations, and recognize employees who perform well without extreme hours.
- Use predictions to improve working conditions company-wide, not to penalize individuals.

## Run it

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook Activity__Course_7_Salifort_Motors_project_lab.ipynb
```

`HR_capstone_dataset.csv` 

## Tools

Python, pandas, NumPy, scikit-learn, seaborn, Matplotlib, Jupyter.
