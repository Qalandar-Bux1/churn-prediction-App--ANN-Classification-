# 🏦 Bank Customer Churn Prediction (ANN + Streamlit)

This project is a **Bank Customer Churn Prediction App** built using an **Artificial Neural Network (ANN)** in TensorFlow/Keras.  
The model predicts whether a customer will **exit (churn)** or **stay** based on features such as credit score, geography, gender, age, balance, etc.  

The app is deployed with **Streamlit** for an interactive user interface. 🚀

---

## ✨ Features
- Preprocessed data with **Label Encoding** and **OneHot Encoding**
- **StandardScaler** for feature scaling
- ANN model trained in TensorFlow/Keras
- Interactive **Streamlit UI** to input customer details
- Real-time prediction with churn probability

---

## 🧑‍💻 Tech Stack
- **Python 3.12+**
- **TensorFlow / Keras**
- **scikit-learn**
- **pandas, numpy**
- **Streamlit**

---

## 📂 Project Structure
📦 churn-prediction-app
┣ 📜 app.py # Streamlit app
┣ 📜 model.keras # Trained ANN model
┣ 📜 scaler.pkl # Scaler for features
┣ 📜 label_encoder_gender # Gender label encoder
┣ 📜 onehot_encoder_geo # Geography one-hot encoder
┣ 📜 requirements.txt # Dependencies
┗ 📜 README.md # Project documentation



---

## ⚙️ Installation & Usage

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app
2️⃣ Install Dependencies



📊 Example Input
Feature	Value
Credit Score	600
Geography	France
Gender	Male
Age	40
Tenure	3
Balance	60,000
NumOfProducts	2
HasCrCard	1
IsActiveMember	1
EstimatedSalary	50,000

✅ Prediction Output
🚨 Customer Will Exit (Churn)

✅ Customer Will Stay

With probability score
