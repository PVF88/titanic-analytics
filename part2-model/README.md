# Part 2 — Predictive Model (Titanic survival)

## Target
Binary classification:
- `survived` (0/1)

## Models Used
- Logistic Regression
- RandomForestClassifier

## Evaluation Metrics
- Accuracy
- F1-score
- ROC-AUC (if probability available)

## How to Run
```bash
cd part2-model
pip install -r requirements.txt
python train.py
```

Outputs:
- `model/model.pkl`
- Printed metrics in console

## Notes
- Uses `train_test_split(stratify=y)` for balanced evaluation.
- Uses sklearn preprocessing where needed (though Part 1 already one-hot encoded key categoricals).
