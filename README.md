# Hotel Booking Demand – Machine Learning Analysis

End-to-end machine learning project on the Hotel Booking Demand dataset (119,390 bookings), completed for a second-year AI module. The goal was to help a hotel reduce cancellations, forecast revenue and understand customer segments.

## Notebooks
| Notebook | Task | Models |
|---|---|---|
| `1_Preprocessing.ipynb` | Missing values, duplicates, outliers, feature engineering (total guests, total nights, revenue, season, weekend ratio, lead-time category), encoding, scaling, correlation heatmap | – |
| `2_Classification.ipynb` | Predict booking cancellations | Logistic Regression, Decision Tree, Random Forest (tuned with GridSearchCV) |
| `3_Regression.ipynb` | Predict booking revenue | Linear Regression, Random Forest, Gradient Boosting |
| `4_Clustering.ipynb` | Customer / booking segmentation | K-Means (elbow method, k = 3), Agglomerative, DBSCAN |

## Key results
| Task | Best model | Result |
|---|---|---|
| Cancellation prediction | Tuned Random Forest | Accuracy 89.8%, precision 89.8%, recall 82.2%, F1 85.8%, ROC-AUC 0.961 |
| Revenue prediction | Random Forest Regressor | R² 0.93, MAE 0.098, RMSE 0.256 (on scaled revenue) |
| Customer segmentation | K-Means (3 clusters) | Silhouette 0.26, Davies-Bouldin 1.51 |

ADR was removed from the regression inputs because `total_revenue` is calculated from it (to avoid data leakage). Agglomerative Clustering and DBSCAN were run on a 5,000-row sample because they need too much memory for the full dataset.

## How to run
1. Download `hotel_bookings.csv` from the Hotel Booking Demand dataset on Kaggle.
2. Open the notebooks in Google Colab (or Jupyter) and upload the CSV.
3. Run the notebooks in order. `1_Preprocessing.ipynb` creates `hotel_bookings_cleaned.csv`, which the other three use.

## Tech stack
Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn, Google Colab

## What I learned / next steps
- The full ML workflow: preprocessing, feature engineering, training, evaluation and business interpretation
- Choosing suitable metrics for each task (precision/recall/F1, MAE/RMSE/R², Silhouette/Davies-Bouldin)
- Next: fit the scaler on the training set only (after the split), try one-hot encoding instead of label encoding, and compare all clustering models on the same sample
