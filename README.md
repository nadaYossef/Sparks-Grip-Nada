# Task 7: Stock Market Prediction Analysis (Numerical & Textual Data)

This project explores the intersection of financial time-series data and news headlines to predict stock market movements. The approach combines **numerical analysis** with **textual sentiment analysis**, utilizing **deep learning models** (LSTM and CNN) and ultimately creating a **hybrid model**.

---

## Steps Followed

### 1. Importing Libraries

Standard data science and deep learning libraries were used:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sklearn
```

---

### 2. Data Wrangling & Loading

- **Textual Data**: Indian news headlines from a dataset with over 1.5 million rows.
  - Dataset: `india-news-headlines.csv`
  - Columns: `publish_date`, `headline_category`, `headline_text`

- **Numerical Data**: NASDAQ Composite Index historical data.
  - Dataset: `^IXIC.csv` from Yahoo Finance
  - Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`

---

### 3. EDA (Exploratory Data Analysis)

- Verified **null values** and **data types** (none found).
- Checked for **duplicates** (none found).
- Applied **boxplots** and **density plots**:
  - Most numerical columns contained **significant outliers**.
  - Distributions were **heavily skewed**, justifying **outlier removal** using Z-score thresholding.

---

### 4. Feature Engineering

- **Converted dates** to datetime objects for merging and time-based filtering.
- **Cleaned headlines**: Removed nulls, special characters, and irrelevant rows.
- **Merged datasets** based on dates to align market performance with corresponding news data.
- **Outlier removal** was applied to ensure model stability and accuracy.
- Textual data was preprocessed for NLP using:
  - Tokenization
  - Stopword removal
  - Lemmatization
  - TF-IDF / word embeddings

---

### 5. Model 1: LSTM for Numerical Data

- Trained an **LSTM model** on stock prices to predict the closing price.
- Features used: `Open`, `High`, `Low`, `Volume`, `Adj Close`
- Results:
  - **Root Mean Squared Error (RMSE)**: Achieved low error rates.
  - Model captured the **temporal trends** well.
  - Visualization showed **reasonable prediction alignment** with real prices.

---

### 6. Model 2: CNN for Textual News Data

- Converted cleaned headlines into word embeddings.
- Used a **1D CNN model** to classify whether a news headline had a **positive or negative impact** on the market.
- Results:
  - Achieved **good classification accuracy**.
  - Captured **semantic features** from headlines that influenced the stock.

---

### 7. Hybrid Deep Learning Model

- Combined the **LSTM (numerical)** and **CNN (textual)** models into a **hybrid model**.
- The final model used both price movements and news sentiment to predict future closing prices.

---

## Key Insights

- Stock prices are **highly sensitive to recent news**, especially related to politics, economy, and major global events.
- Outliers and skewed distributions can **heavily distort prediction models** — careful preprocessing is essential.
- **LSTM excels** in capturing patterns in sequential data like stock prices.
- **CNN performs well** in detecting sentiment and context from textual data.
- A **hybrid model combining both** types of data yielded the **most robust performance**, showing potential for real-world application in algo-trading systems.

---

## Final Results

| Model Type | Input | Output | Performance |
|------------|--------|--------|-------------|
| LSTM       | Numerical | Future Close Price | RMSE: Low |
| CNN        | Textual | Sentiment Label | High Accuracy |
| Hybrid     | Numerical + Textual | Enhanced Prediction | Most Stable & Accurate |

---

## Datasets Used

- The csv file i uploaded here
- NASDAQ Historical Data from [Yahoo Finance](https://finance.yahoo.com/)

---

## Conclusion

This project demonstrates how combining **numerical time series** with **news sentiment analysis** can greatly improve the prediction of stock market movements. With proper preprocessing, deep learning models can be highly effective in capturing and leveraging real-world patterns from complex data sources.
