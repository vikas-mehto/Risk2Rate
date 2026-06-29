# 🩺 Risk2Rate – AI-Powered Health Insurance Premium Predictor




---

## 📌 Overview

**Risk2Rate** is a Machine Learning-powered web application that predicts an individual's health insurance premium using demographic, medical, lifestyle, and policy-related information.

The application leverages a **Random Forest Regression** model trained on historical insurance data to estimate annual and monthly premiums. It also provides BMI analysis, health insights, and downloadable prediction reports through an intuitive **Streamlit** interface.

---

## 🎥 Project Demo

📺 **Watch the complete project demonstration**

**https://youtu.be/knC5EmH2ahY**

---

## 🚀 Live Demo

🌐 **Streamlit App**

**🌐 Live Demo:** [Risk2Rate Web App](https://risk2rate-kgfqpof8xxftirsedjqelh.streamlit.app/)

---

## ✨ Features

* 🤖 AI-powered health insurance premium prediction
* 💰 Annual and monthly premium estimation
* 📊 BMI calculation and health category analysis
* 💡 Personalized premium insights and recommendations
* 📄 Downloadable PDF prediction report
* ⚡ Fast and responsive Streamlit dashboard
* ☁️ Deployed on Streamlit Community Cloud

---

## 🛠 Tech Stack

| Category                 | Technologies                           |
| ------------------------ | -------------------------------------- |
| **Programming Language** | Python                                 |
| **Frontend**             | Streamlit                              |
| **Machine Learning**     | Scikit-learn (Random Forest Regressor) |
| **Data Processing**      | Pandas, NumPy                          |
| **Explainable AI**       | SHAP                                   |
| **Visualization**        | Matplotlib                             |
| **Report Generation**    | ReportLab                              |

---

## 🏗️ System Architecture

```text
                    User
                      │
                      ▼
             Streamlit Web Interface
                      │
                      ▼
       Input Validation & Preprocessing
                      │
                      ▼
      Random Forest Regression Model
                      │
          ┌───────────┼────────────┐
          ▼           ▼            ▼
 Premium Prediction  BMI Analysis  Health Insights
                      │
                      ▼
           PDF Report Generation
```

---

## 📂 Project Structure

```text
Risk2Rate/
│
├── app/
│   └── app.py
│
├── model/
│   ├── insurance_model.pkl
│   └── shap_explainer.pkl
│
├── assets/
│   └── demo.gif
│
├── insurance.csv
├── requirements.txt
├── runtime.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/vikas-mehto/Risk2Rate.git
cd Risk2Rate
```

### Create a Virtual Environment

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app/app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📋 Input Parameters

The model predicts premiums using the following inputs:

* Age
* Gender
* Height & Weight (BMI)
* Smoking Status
* City
* Policy Type
* Sum Insured
* Insurance Company
* Network Hospitals
* Pre-existing Disease
* Waiting Period
* Co-payment Percentage
* Claim Settlement Ratio

---

## 📈 Output

The application provides:

* 💰 Annual Premium Prediction
* 📅 Monthly Premium Prediction
* 📊 BMI Score & Category
* 💡 Personalized Health Insights
* 📄 Downloadable PDF Report

---

## 🧠 Machine Learning Workflow

1. User enters health and policy information.
2. Input data is validated and preprocessed.
3. Features are passed to the trained Random Forest Regression model.
4. Premium is predicted.
5. BMI analysis and health insights are generated.
6. Results are displayed with an option to download a PDF report.

---

## 💡 Challenges Solved

* Built an end-to-end machine learning prediction pipeline.
* Integrated a trained Random Forest model into a Streamlit application.
* Managed model serialization and deployment compatibility.
* Implemented efficient model loading using Streamlit caching.
* Successfully deployed the application on Streamlit Community Cloud.

---

## 🔮 Future Enhancements

* Insurance plan recommendation system
* Multiple ML model comparison
* Premium trend visualization
* User authentication
* Medical report upload and analysis
* Multi-language support

---

## 👨‍💻 Author

**Vikas Mehto**

B.Tech Information Technology
Delhi Technological University (DTU)

* **GitHub:** https://github.com/vikas-mehto
* **LinkedIn:** https://www.linkedin.com/in/vikas-613781256

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

It helps improve the project and motivates future development.

---

<p align="center">
Made with ❤️ using <strong>Python</strong>, <strong>Streamlit</strong>, and <strong>Machine Learning</strong>.
</p>
