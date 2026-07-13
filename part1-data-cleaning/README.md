# Part 1 — Data Cleaning (Titanic)

## Business Question
**Predict survivability ("Survived")** and produce a clean dataset by fixing:
- missing values
- wrong formats
- outliers
- duplicates

## Dataset
Titanic passenger dataset fetched from:
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv

## Cleaning Steps Performed
1. **Load data**
2. **Remove duplicates** (if any)
3. **Fix data types / formats**
   - Ensure `age` and `fare` are numeric
4. **Handle missing values**
   - `age`: fill with **median age** grouped by `sex`
   - `embarked`: fill with **mode embarked**
5. **Outlier handling**
   - `fare`: cap at the **99th percentile** to reduce extreme outliers impact
6. **Feature engineering**
   - Create `cabin_known` (1 if cabin present else 0)
   - Drop very sparse raw `deck`/`cabin` if present in a messy way
7. **Encode categorical variables**
   - One-hot encode `sex`, `embarked`, `class`
8. **Save cleaned dataset**
   - `data_clean/titanic_cleaned.csv`

## How to Run
```bash
cd part1-data-cleaning
pip install -r requirements.txt
python main.py
```

Cleaned output:
- `data_clean/titanic_cleaned.csv`
