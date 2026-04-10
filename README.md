# 📰 Fake News Detection using Machine Learning

## 📌 Overview

This project is a **Fake News Detection System** built using **Machine Learning and Natural Language Processing (NLP)**.
It classifies news articles as **REAL or FAKE** based on textual patterns.

The system uses **TF-IDF vectorization with n-grams** and a **SGDClassifier** to achieve high accuracy.

---

## 🚀 Features

* 🧠 NLP-based text preprocessing (cleaning, stopwords removal, lemmatization)
* 📊 TF-IDF feature extraction with bi-grams
* 🤖 Machine Learning model (SGDClassifier)
* 📈 High accuracy (~95–99% on test data)
* 🌐 Streamlit web app for real-time prediction
* 💾 Model saving & loading using Pickle

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* NLTK
* Streamlit

---

## 📊 Dataset

This project uses the **Fake and Real News Dataset** from Kaggle.

🔗 Download here:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

After downloading, place the files in the `data/` folder:

* `Fake.csv`
* `True.csv`

---

## ⚙️ How to Run the Project

### 1. Clone Repository

```bash
git clone https://github.com/ArchitaJain7/fake-news-detection.git
cd fake-news-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model (Optional)

Run the Jupyter Notebook to train the model:

```bash
Notebook/Fake_News_Detection.ipynb
```

---

### 4. Run Streamlit App

```bash
streamlit run app.py
```

👉 Open in browser:

```
http://localhost:8501
```

---

## 🧪 Example

**Input:**

```
Government announces new tax reform
```

**Output:**

```
REAL NEWS 🟢
```

---

## 📈 Model Performance

* Logistic Regression: ~98%
* Naive Bayes: ~95%
* SGDClassifier: ~96–99% (selected model)

---

## ⚠️ Limitations

* The model predicts based on **text patterns**, not factual verification
* May misclassify news from **different domains (domain shift)**
* Performance depends on training dataset distribution

---

## 🔮 Future Improvements

* Use Deep Learning models (LSTM, BERT)
* Add real-time news API integration
* Improve generalization with diverse datasets
* Deploy app online

---

## 👩‍💻 Author

**Archita Jain**

---

## ⭐ Acknowledgment

Dataset sourced from Kaggle Fake News Dataset
