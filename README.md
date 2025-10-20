# 🧠 Kaiburr Task 5  
### Text Classification of Consumer Complaints using Machine Learning  

---

## 📋 Project Overview  

This project is developed as part of the **Kaiburr Assessment - Task 5**.  
The goal is to build a **text classification model** that automatically categorizes consumer complaints into relevant financial product categories using machine learning techniques.

---

## ⚙️ Technologies & Libraries Used  

- Python  
- Pandas, NumPy  
- scikit-learn  
- NLTK (for text preprocessing)  
- Joblib (for model saving/loading)  

---

## 🧾 Dataset  

The dataset used is the **Consumer Complaints Database** provided by the **U.S. Consumer Financial Protection Bureau (CFPB)**.  

📂 **Dataset Source:**  
[Download complaints.csv.zip](https://files.consumerfinance.gov/ccdb/complaints.csv.zip)

⚠️ **Note:**  
The full dataset is too large to upload to GitHub (>100 MB).  
A smaller version named **`complaints_sample.csv`** (first 1000 rows) is included in this repository for demonstration and testing purposes.

---


## 🧹 Data Preprocessing  

- Removed non-alphabetic characters and converted all text to lowercase.  
- Lemmatized words using `WordNetLemmatizer`.  
- Combined similar categories (e.g., “Credit card or prepaid card” → “Credit card”).  
- Removed rare product classes (with fewer than 2 samples).  
- Split the dataset into **train (80%)** and **test (20%)** sets.  

---

## 🧠 Model Architecture  

- **Vectorization:** TF-IDF Vectorizer (`max_features=7000`, `ngram_range=(1,2)`)  
- **Model:** Logistic Regression (`solver='liblinear'`, `class_weight='balanced'`)  
- **Training Epochs:** Adjustable via command line argument  

---

## High-level Directory Overview

```bash
Kaiburr-Task-5
├── main.py
├── complaints_sample.csv
├── requirements.txt
├── README.md
├── outputs
│ ├── model.joblib
│ └── vectorizer.joblib
└── screenshots
├── model_evaluation.png
├── directories.png
└── prediction.png
```


**Note:** Task building, and screenshots are documented inside their respective folders.

## 🚀 How to Run the Project  

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Kaiburr-Task-5.git
cd Kaiburr-Task-5
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

3️⃣ Run the Script
```bash
python main.py --file_path path-to-complaints.csv --out_dir outputs --epochs 3 --max_rows 1000
```

📊 Output

After successful execution, the following are displayed in the terminal:

* Accuracy score

* Classification report (Precision, Recall, F1-score)

* 5 sample predictions with actual vs predicted categories

💾 Files saved to outputs/ folder:

model.joblib
vectorizer.joblib
sample_predictions.txt


🖼️ Screenshots

📸 Output Screenshot:
* Model evaluation & classification report
  <img width="1013" height="679" alt="Model Evaluation" src="https://github.com/user-attachments/assets/391c985d-c181-4c63-8709-fca4d35d4422" />
  
* Directories present
  <img width="963" height="520" alt="directories" src="https://github.com/user-attachments/assets/548287cd-abb2-4450-8342-fa5a6de3eb1e" />

* Predictions
  <img width="1018" height="545" alt="Predictions" src="https://github.com/user-attachments/assets/49a7607b-abf5-4787-b131-407939413cce" />

---

## 🧾 License
This project is licensed under the **Unlicense** — free and open for anyone to use.

## 💖 Support
If you like this project, please ⭐ **star the repo** and connect with me on [LinkedIn](https://www.linkedin.com/in/anush-erappareddy-95a8352a2/) 😊

---










