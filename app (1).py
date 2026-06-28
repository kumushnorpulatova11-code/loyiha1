
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
st.set_page_config(
    page_title="Mijozlar Chiqib Ketish Paneli",
    page_icon="📊",
    layout="wide"
)
 
df = pd.read_csv("./Churn_Modelling.csv")
 
def risk_level(row):
    score = 0
    if row["Age"] > 45:
        score += 1
    if row["CreditScore"] < 500:
        score += 1
    if row["IsActiveMember"] == 0:
        score += 1
    if row["Balance"] > 100000:
        score += 1
    if score >= 3:
        return "Yuqori Xavf"
    elif score >= 2:
        return "O'rta Xavf"
    else:
        return "Past Xavf"
 
df["RiskLevel"] = df.apply(risk_level, axis=1)
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[18, 30, 45, 60, 100],
    labels=["Yosh (18-30)", "Katta (30-45)", "O'rta Yosh (45-60)", "Keksa (60+)"]
)
 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
 
X = df[["CreditScore", "Age", "Balance", "NumOfProducts", "EstimatedSalary"]]
y = df["Exited"]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
 
# O'zbekcha ustun nomlari lugati
USTUN_NOMLARI = {
    'RowNumber':       'Qator Raqami',
    'CustomerId':      'Mijoz ID',
    'Surname':         'Familiya',
    'CreditScore':     'Kredit Balli',
    'Geography':       'Mamlakat',
    'Gender':          'Jinsi',
    'Age':             'Yosh',
    'Tenure':          'Xizmat Yili',
    'Balance':         'Balans',
    'NumOfProducts':   'Mahsulotlar Soni',
    'HasCrCard':       'Kredit Karta',
    'IsActiveMember':  'Faol Mijoz',
    'EstimatedSalary': 'Taxminiy Maosh',
    'Exited':          'Chiqib Ketgan',
    'RiskLevel':       'Xavf Darajasi',
    'AgeGroup':        'Yosh Guruhi'
}
 
def uzbekcha_jadval(data):
    """DataFrame ni ko'rsatish uchun o'zbekchaga o'tkazadi"""
    d = data.copy().rename(columns=USTUN_NOMLARI)
    if 'Jinsi' in d.columns:
        d['Jinsi'] = d['Jinsi'].map({'Female': 'Ayol', 'Male': 'Erkak'})
    if 'Mamlakat' in d.columns:
        d['Mamlakat'] = d['Mamlakat'].map({'France': 'Fransiya', 'Spain': 'Ispaniya', 'Germany': 'Germaniya'})
    if 'Chiqib Ketgan' in d.columns:
        d['Chiqib Ketgan'] = d['Chiqib Ketgan'].map({0: "Qolgan", 1: "Ketgan"})
    return d
 
# Grafiklar uchun alohida o'zbekcha dataframe
df_uz = df.copy()
df_uz["Jinsi"]        = df["Gender"].map({'Female': 'Ayol', 'Male': 'Erkak'})
df_uz["Mamlakat"]     = df["Geography"].map({'France': 'Fransiya', 'Spain': 'Ispaniya', 'Germany': 'Germaniya'})
df_uz["Holati"]       = df["Exited"].map({0: "Qolgan", 1: "Ketgan"})
df_uz["Xavf Darajasi"]= df["RiskLevel"].astype(str)
df_uz["Yosh Guruhi"]  = df["AgeGroup"].astype(str)
 
# Eski inglizcha ustunlarni o'chiramiz (grafiklarda chiqib qolmasligi uchun)
df_uz = df_uz.drop(columns=["Gender", "Geography", "Exited", "RiskLevel", "AgeGroup"])
 
st.sidebar.title("Navigatsiya")
menu = st.sidebar.radio(
    "Sahifani tanlang",
    ["Bosh sahifa", "Ma'lumotlar To'plami", "Chiqish Tahlili", "Xavf Tahlili", "Bashorat"]
)
 
if menu == "Bosh sahifa":
    st.title("📊 Mijozlar Chiqib Ketish Tahlili Paneli")
    st.markdown("""
    ### Loyiha Haqida
    Bu loyiha mijozlarning chiqib ketish xatti-harakatini tahlil qiladi va bankni tark etish xavfi ostidagi mijozlarni aniqlaydi.
 
    ### Maqsadlar
    - Mijozlar chiqib ketishini tahlil qilish
    - Yuqori xavfli mijozlarni aniqlash
    - Bashorat modeli yaratish
    - Interaktiv panel yaratish
    """)
    st.success("Mashinaviy O'rganish Modeli: Random Forest Klassifikatori")
    st.success("Model Aniqligi: 79.85%")
 
    total_customers = len(df)
    churn_customers = df["Exited"].sum()
    churn_rate = round(df["Exited"].mean() * 100, 2)
 
    col1, col2, col3 = st.columns(3)
    col1.metric("Jami Mijozlar", total_customers)
    col2.metric("Chiqib Ketgan Mijozlar", churn_customers)
    col3.metric("Chiqib Ketish Darajasi (%)", churn_rate)
 
    st.subheader("📊 Mijozlar Statistikasi")
    avg_age = round(df["Age"].mean(), 1)
    avg_balance = round(df["Balance"].mean(), 2)
    avg_credit = round(df["CreditScore"].mean(), 1)
 
    col1, col2, col3 = st.columns(3)
    col1.metric("O'rtacha Yosh", avg_age)
    col2.metric("O'rtacha Balans", f"${avg_balance:,.0f}")
    col3.metric("O'rtacha Kredit Balli", avg_credit)
 
    st.subheader("📈 Chiqib Ketish Darajasi Ko'rsatkichi")
    st.progress(churn_rate / 100)
    st.write(f"Joriy Mijozlar Chiqib Ketish Darajasi: {churn_rate}%")
 
elif menu == "Ma'lumotlar To'plami":
    st.title("📋 Ma'lumotlar To'plami")
 
    st.subheader("Ma'lumotlar To'plami O'lchami")
    st.write(df.shape)
 
    st.subheader("Ma'lumotlar To'plamiga Ko'rinish")
    st.dataframe(uzbekcha_jadval(df.head()))
 
    st.subheader("Ustunlar")
    uzbekcha_ustunlar = [USTUN_NOMLARI.get(col, col) for col in df.columns.tolist()]
    st.write(uzbekcha_ustunlar)
 
    st.subheader("Yo'qolgan Qiymatlar")
    yoqolgan = df.isnull().sum().reset_index()
    yoqolgan.columns = ['Ustun', "Yo'qolgan Qiymatlar"]
    yoqolgan['Ustun'] = yoqolgan['Ustun'].map(lambda x: USTUN_NOMLARI.get(x, x))
    st.dataframe(yoqolgan)
 
elif menu == "Chiqish Tahlili":
    st.title("📈 Chiqish Tahlili")
 
    # 1. Chiqib ketish taqsimoti
    st.subheader("Mijozlar Chiqib Ketish Taqsimoti")
    fig, ax = plt.subplots()
    holat_son = df_uz["Holati"].value_counts()
    holat_son.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    plt.title("Mijozlar Chiqib Ketish Taqsimoti")
    plt.xlabel("Holati")
    plt.ylabel("Mijozlar Soni")
    plt.xticks(rotation=0)
    st.pyplot(fig)
 
    # 2. Jins bo'yicha chiqib ketish
    st.subheader("Jins Bo'yicha Chiqib Ketish")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Jinsi", hue="Holati", data=df_uz, ax=ax,
                  palette={"Qolgan": "#2ecc71", "Ketgan": "#e74c3c"})
    plt.title("Jins Bo'yicha Chiqib Ketish")
    plt.xlabel("Jinsi")
    plt.ylabel("Mijozlar Soni")
    ax.legend(title="Holati")
    st.pyplot(fig)
 
    # 3. Mamlakat bo'yicha chiqib ketish
    st.subheader("Mamlakat Bo'yicha Chiqib Ketish")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Mamlakat", hue="Holati", data=df_uz, ax=ax,
                  palette={"Qolgan": "#2ecc71", "Ketgan": "#e74c3c"})
    plt.title("Mamlakat Bo'yicha Chiqib Ketish")
    plt.xlabel("Mamlakat")
    plt.ylabel("Mijozlar Soni")
    ax.legend(title="Holati")
    st.pyplot(fig)
 
    # 4. Yosh va Balans tarqalma grafik
    st.subheader("Yosh va Balans (Tarqalma Grafik)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_uz, x="Age", y="Balance", hue="Holati", ax=ax,
                    palette={"Qolgan": "#2ecc71", "Ketgan": "#e74c3c"})
    plt.title("Chiqib Ketish Bo'yicha Yosh va Balans")
    plt.xlabel("Yosh")
    plt.ylabel("Balans")
    ax.legend(title="Holati")
    st.pyplot(fig)
 
    # 5. Yosh guruhi bo'yicha segmentatsiya
    st.subheader("👥 Yosh Guruhi Bo'yicha Mijozlar Segmentatsiyasi")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Yosh Guruhi", hue="Holati", data=df_uz, ax=ax,
                  palette={"Qolgan": "#2ecc71", "Ketgan": "#e74c3c"})
    plt.title("Yosh Guruhi Bo'yicha Mijozlar Segmentatsiyasi")
    plt.xlabel("Yosh Guruhi")
    plt.ylabel("Mijozlar Soni")
    ax.legend(title="Holati")
    st.pyplot(fig)
 
elif menu == "Xavf Tahlili":
    st.title("⚠️ Xavf Tahlili")
 
    st.subheader("Xavf Darajasi Taqsimoti")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Xavf Darajasi", data=df_uz, ax=ax,
                  palette={"Past Xavf": "#2ecc71", "O'rta Xavf": "#f39c12", "Yuqori Xavf": "#e74c3c"})
    plt.title("Mijozlar Xavf Darajalari")
    plt.xlabel("Xavf Darajasi")
    plt.ylabel("Mijozlar Soni")
    st.pyplot(fig)
 
    st.subheader("Xavf Xulosasi")
    xavf_xulosasi = df["RiskLevel"].value_counts().reset_index()
    xavf_xulosasi.columns = ['Xavf Darajasi', 'Mijozlar Soni']
    st.dataframe(xavf_xulosasi)
 
    st.subheader("Eng Yuqori Xavfli Mijozlar")
    high_risk = df[df["RiskLevel"] == "Yuqori Xavf"]
    st.dataframe(
        uzbekcha_jadval(
            high_risk[["CustomerId", "Age", "CreditScore", "Balance", "RiskLevel"]].head(10)
        )
    )
 
    st.subheader("📥 Yuqori Xavfli Mijozlarni Yuklab Olish")
    csv = high_risk.to_csv(index=False)
    st.download_button(
        label="Yuqori Xavfli Mijozlar CSV ni Yuklab Olish",
        data=csv,
        file_name="yuqori_xavfli_mijozlar.csv",
        mime="text/csv"
    )
 
    st.subheader("📊 Xususiyat Ahamiyati")
    importance = pd.DataFrame({
        "Xususiyat": X.columns,
        "Ahamiyat": model.feature_importances_
    })
    importance["Xususiyat"] = importance["Xususiyat"].map(lambda x: USTUN_NOMLARI.get(x, x))
    importance = importance.sort_values(by="Ahamiyat", ascending=False)
    st.dataframe(importance)
 
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Ahamiyat", y="Xususiyat", data=importance, ax=ax)
    plt.title("Xususiyat Ahamiyati")
    plt.xlabel("Ahamiyat Darajasi")
    plt.ylabel("Xususiyat")
    st.pyplot(fig)
 
elif menu == "Bashorat":
    st.title("🤖 Chiqib Ketish Bashorati")
    st.write("Chiqib ketish xavfini bashorat qilish uchun mijoz ma'lumotlarini kiriting.")
 
    credit_score = st.number_input("Kredit Balli", min_value=300, max_value=900, value=600)
    age = st.number_input("Yosh", min_value=18, max_value=100, value=40)
    balance = st.number_input("Balans", min_value=0.0, value=50000.0)
    products = st.number_input("Mahsulotlar Soni", min_value=1, max_value=4, value=1)
    salary = st.number_input("Taxminiy Maosh", min_value=0.0, value=50000.0)
 
    if st.button("Chiqib Ketishni Bashorat Qilish"):
        prediction = model.predict([[credit_score, age, balance, products, salary]])
        probability = model.predict_proba([[credit_score, age, balance, products, salary]])
        churn_probability = round(probability[0][1] * 100, 2)
 
        st.metric("Chiqib Ketish Ehtimoli", f"{churn_probability}%")
 
        if churn_probability > 70:
            st.error("🔴 Yuqori Xavfli Mijoz")
            st.info("""
Tavsiya:
 
- Sodiqlik bonusi taklif qilish
- Mijoz bilan bevosita bog'lanish
- Maxsus chegirma paketi taqdim etish
""")
        elif churn_probability > 40:
            st.warning("🟡 O'rta Xavfli Mijoz")
            st.info("""
Tavsiya:
 
- Reklama takliflari yuborish
- Shaxsiy marketing kampaniyasi
""")
        else:
            st.success("🟢 Past Xavfli Mijoz")
            st.info("""
Tavsiya:
 
- Mijoz barqaror
- Muntazam xizmatlarni davom ettirish
""")
 
        if prediction[0] == 1:
            st.error("⚠️ Mijoz Chiqib Ketishi Mumkin")
        else:
            st.success("✅ Mijoz Qolishi Mumkin")