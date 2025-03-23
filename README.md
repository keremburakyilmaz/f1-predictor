# Gentlemen! Start! Your! Engines!

This project downloads F1 data (race, qualifying, and weather) from previous years and trains a model with a random forest based on this data in order to predict the next race's positions:
- **Top 3**
- **Midfield (Positions 4–10)**
- **Backmarker (Positions 11 and below)**

---

## Classification Report

| Class        | Precision | Recall |
|--------------|-----------|--------|
| Backmarker   | 0.88      | 0.79   |
| Midfield     | 0.86      | 0.71   |
| Top 3        | 0.47      | 1.00   |

**Metric Definitions:**
- **Precision**: Correct positive predictions / All predicted positives
- **Recall**: Correct positive predictions / All actual positives



## Confusion Matrix

|                      | Predicted: Backmarker | Predicted: Midfield | Predicted: Top 3 |
|----------------------|-----------------------|----------------------|------------------|
| **Actual: Backmarker** | 15                    | 4                    | 0                |
| **Actual: Midfield**   | 2                     | 24                   | 8                |
| **Actual: Top 3**      | 0                     | 0                    | 7     



## Observations
- The classifier performs **strongly on Backmarker and Midfield** classes.
- **Top 3 drivers are correctly recalled 100% of the time**, but some Midfield drivers are wrongly predicted as Top 3, leading to lower precision.
- Most misclassifications occur between **Midfield and Top 3**, indicating a potential need for more discriminative features.
- **Overall accuracy: 77%** demonstrates solid baseline performance.

---

## Methods:
- FastF1 API for the data
- ThreadPoolExecutor for multithreading to speed up the data downloading process
- Random Forest (both decision trees and random forest both implemented by me)

## Roadmap
- [x] Train the model with at least 50% accuracy (baby steps!).
- [ ] Implement a cross validation and resampling to increase accuracy.
- [ ] Change Random Forest to Gradient Boosting for improved classification.
- [ ] Increase data amount for more accurate predictions.
- [ ] Change the model into predicting the finish positions for each racer rather than putting them into 3 classes.
