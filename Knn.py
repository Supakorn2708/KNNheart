from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การจำแนกข้อมูลด้วยเทคนิค Machine Learning')
st.image("./img/Men.jpg")
col1, col2 = st.columns(2)

with col1:
   st.header("หัวใจแข็งแรง")
   st.image("./img/heart1.jpg")

with col2:
   st.header("เป็นโรคหัวใจ")
   st.image("./img/heart2.jpg")

html_7 = """
<div style="background-color:#c5f18a;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>การทำนายโรคหัวใจ ❤️‍🩹</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
st.markdown("")

st.subheader("ข้อมูลส่วนแรก 10 แถว")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

# สถิติพื้นฐาน
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# การเลือกแสดงกราฟตามฟีเจอร์
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

# วาดกราฟ boxplot
st.write(f"### ❤️‍🔥 Boxplot: {feature} แยกข้อมูลโรคหัวใจ")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='variety', y=feature, ax=ax)
st.pyplot(fig)

# วาด pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🐇 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='variety')
    st.pyplot(fig2)

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

Age = st.number_input("กรุณาเลือกข้อมูล Age")
Sex = st.number_input("กรุณาเลือกข้อมูล Sex")
ChestPainType = st.number_input("กรุณาเลือกข้อมูล ChestPainType")
RestingBP = st.number_input("กรุณาเลือกข้อมูล RestingBP")
Cholesterol = st.number_input("กรุณาเลือกข้อมูล Cholesterol")
FastingBS = st.number_input("กรุณาเลือกข้อมูล FastingBS")
RestingECG = st.number_input("กรุณาเลือกข้อมูล RestingECG")
MaxHR = st.number_input("กรุณาเลือกข้อมูล MaxHR")
ExerciseAngina = st.number_input("กรุณาเลือกข้อมูล ExerciseAngina")
Oldpeak = st.number_input("กรุณาเลือกข้อมูล Oldpeak")
ST_Slope = st.number_input("กรุณาเลือกข้อมูล ST_Slope")

if st.button("ทำนายผล"):
    #st.write("ทำนาย")
   dt = pd.read_csv("./data/iris-3.csv") 
   X = dt.drop('variety', axis=1)
   y = dt.variety   

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  
    
   x_input = np.array([[Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == 'Setosa':
    st.image("./img/iris1.jpg")
   elif out[0] == 'Versicolor':       
    st.image("./img/iris2.jpg")
   else:
    st.image("./img/iris3.jpg")
else:
    st.write("ไม่ทำนาย")