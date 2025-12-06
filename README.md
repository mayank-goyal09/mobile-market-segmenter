# 📱 Mobile Market Segmenter 🎯  
### *Classify Customers Like a Pro with K-Nearest Neighbors!*

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</div>

---

## 🚀 **What's This About?**

Ever wondered how companies like **Amazon, Netflix, or Spotify** know exactly what you want? 🤔  
They use **Machine Learning** to segment customers into groups based on behavior!

**Mobile Market Segmenter** is a **K-Nearest Neighbors (KNN) classification** project that:
- 📊 Analyzes mobile usage patterns (data usage, call duration, texts, etc.)
- 👥 Segments customers into **market groups** (Budget Users, Power Users, Premium Subscribers, etc.)
- 🎯 Helps businesses **target the right audience** with personalized campaigns

---

## 💡 **Why KNN?**

**K-Nearest Neighbors** is like asking your 5 closest friends for advice! 🙋‍♂️  
- Simple, interpretable, and powerful for classification tasks
- Works great when similar customers have similar behaviors
- No assumptions about data distribution (non-parametric)

---

## 🛠️ **Tech Stack**

| Tool | Purpose |
|------|---------||
| **Python 🐍** | Core programming language |
| **Pandas 📊** | Data manipulation & analysis |
| **NumPy 🔢** | Numerical computations |
| **Scikit-learn 🤖** | KNN implementation & evaluation |
| **Streamlit 🎨** | Interactive web app |
| **Matplotlib/Seaborn 📈** | Data visualization |

---

## 📂 **Project Structure**

```
mobile-market-segmenter/
│
├── app.py                    # 🎨 Streamlit app
├── model.py                  # 🤖 KNN model training
├── requirements.txt          # 📦 Dependencies
├── README.md                 # 📖 You're here!
├── data/
│   └── mobile_users.csv      # 📊 Dataset
├── models/
│   └── knn_model.pkl         # 💾 Trained model
└── utils/
    └── preprocessing.py      # 🧹 Data preprocessing
```

---

## 🎯 **Features**

✅ **Data Preprocessing** (handling missing values, feature scaling)  
✅ **KNN Classification** (optimal K selection using elbow method)  
✅ **Model Evaluation** (accuracy, precision, recall, F1-score)  
✅ **Confusion Matrix Visualization**  
✅ **Interactive Streamlit App** (predict customer segments in real-time)  
✅ **Feature Importance Analysis**

---

## 🏃‍♂️ **Quick Start**

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/mayank-goyal09/mobile-market-segmenter.git
cd mobile-market-segmenter
```

### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3️⃣ **Train the Model**
```bash
python model.py
```

### 4️⃣ **Run the Streamlit App** 🚀
```bash
streamlit run app.py
```

### 5️⃣ **Open in Browser**
Visit: `http://localhost:8501`

---

## 📊 **How It Works**

### **Step 1: Data Collection**
- Customer demographics (age, income, location)
- Mobile usage behavior (data usage, call minutes, SMS count)

### **Step 2: Data Preprocessing**
- Handle missing values
- Feature scaling (StandardScaler)
- Train-test split (80-20)

### **Step 3: KNN Training**
- Find optimal K value (using cross-validation)
- Train KNN classifier
- Save the model

### **Step 4: Evaluation**
- Confusion matrix
- Accuracy, Precision, Recall, F1-Score
- Visualize results

### **Step 5: Deployment**
- Streamlit app for real-time predictions
- Input customer data → Get segment prediction instantly!

---

## 🎨 **Streamlit App Preview**

The app allows you to:
- 📥 Upload customer data (CSV)
- 🔍 Predict market segments for new customers
- 📊 Visualize segment distribution
- 🎯 Download predictions as CSV

---

## 📈 **Model Performance**

| Metric | Score |
|--------|-------|
| **Accuracy** | 92% |
| **Precision** | 90% |
| **Recall** | 89% |
| **F1-Score** | 89.5% |

---

## 🧠 **What I Learned**

✅ **KNN Algorithm** (distance metrics, K selection)  
✅ **Feature Scaling** (why it's crucial for distance-based algorithms)  
✅ **Model Evaluation** (confusion matrix, classification report)  
✅ **Streamlit Deployment** (building interactive ML apps)  
✅ **Business Applications** (customer segmentation strategies)

---

## 🔮 **Future Enhancements**

- 🌟 Add **ensemble methods** (Random Forest, XGBoost) for comparison
- 📊 Include **clustering algorithms** (K-Means) for unsupervised segmentation
- 🎯 Implement **A/B testing** framework
- 🚀 Deploy on **AWS/Azure** for scalability
- 📱 Create a **mobile app** version

---

## 🤝 **Contributing**

Found a bug? Have a cool feature idea? 🚀  
Feel free to **fork**, **star** ⭐, and submit a **pull request**!

---

## 📝 **License**

This project is **MIT Licensed**. Feel free to use it for learning!

---

## 👨‍💻 **About Me**

**Mayank Goyal** | Data Analyst Intern @ SpacECE Foundation  
🎓 11th Class Student | ML Enthusiast | Python Developer  

📦 **Connect with me:**  
- [GitHub](https://github.com/mayank-goyal09)  
- [LinkedIn](https://linkedin.com/in/mayank-goyal09)  

---

<div align="center">
  <h3>⭐ If you found this project helpful, give it a star! ⭐</h3>
  <p><i>Made with ❤️ by Mayank Goyal</i></p>
</div>