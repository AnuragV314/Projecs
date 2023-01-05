import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import webbrowser


st.set_page_config(
    page_title="Disease P",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.text("hii! I'm Anurag Verma")


# loading the savd model
diabetes_modle = pickle.load(open('saved models/diabetes_model.sav', 'rb'))
heart_disease_modle = pickle.load(open('saved models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('saved models/parkinsons_model.sav', 'rb'))

#sidebar
with st.sidebar:
    selected = option_menu("Multiple Disease Predictions",
                            ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson Disease Prediction"], 
                            icons=['shield-plus', 'heart', 'person'], menu_icon="justify", default_index=0)


# ---------------------------------------------------------


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
import pandas as pd
# all data 
Diabetes_dataset = pd.read_csv('dataset/diabetes.csv')
heart_dataset = pd.read_csv('dataset/heart.csv')
parkinsons_dataset = pd.read_csv('dataset/parkinsons.csv')


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Diabetes Prediction Page

if selected=="Diabetes Prediction":
    # page title
    st.title('Diabetes Prediction using Machine Learning')
    st.image('img/diabetess.jpg')
    st.header('DataSet')
    st.dataframe(Diabetes_dataset)
    st.write("##")

    # input feature
    col1, col2, col3 = st.columns(3)

    # 1st row
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies')       
    with col2:
        Glucose = st.number_input('Glucose Level')  
    with col3:
        BloodPressure = st.number_input('Blood Pressure value')  

    # 2nd row  
    with col1:
        SkinThickness = st.number_input('Skin Thickness value')   
    with col2:
        Insulin = st.number_input('Insulin Level')    
    with col3:
        BMI = st.number_input('BMI value') 

    # 3rd row       
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.number_input('Age of the Person')

    st.write("##")    

    # code for prediction   
    diab_diagnosis = ''

    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        diab_pred = diabetes_modle.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

        if diab_pred[0]==1:
            diab_diagnosis = 'The persion is Diabetic.'
        else:
            diab_diagnosis = 'The persion is Not Diabetic.'

    st.success(diab_diagnosis)        




#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# Heart Disease Prediction Page

if selected=="Heart Disease Prediction":
    # page title
    st.title('Heart Disease Prediction using Machine Learning')
    st.image('img/heart.jpg')
    st.header('DataSet')
    st.dataframe(heart_dataset)
    st.write("##")

    # input feature
    col1, col2, col3 = st.columns(3)

    # 1st row
    with col1:
        age = st.number_input('Age')   
    with col2:
        sex = st.number_input('Sex')   
    with col3:
        cp = st.number_input('Chest Pain types')

    # 2nd row
    with col1:
        trestbps = st.number_input('Resting Blood Pressure')    
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')  
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
      
    # 3rd row
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results') 
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')       
    with col3:
        exang = st.number_input('Exercise Induced Angina')
     
    # 4th row
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')       
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')       
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy')
      
    # 5th row
    with col1:
        thal = st.number_input('thal:  0 = normal; 1 = fixed defect; 2 = reversable defect') 


    st.write("##")    

    # code for prediction   
    heart_diagnosis = ''

    # creating a button for prediction
    if st.button('Heart Disease Test Result'):
        data = np.array([age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal], dtype=float)
        heart_prediction = heart_disease_modle.predict([data])
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having Heart Disease'
        else:
          heart_diagnosis = 'The person does not have any Heart Disease'
        
    st.success(heart_diagnosis)   
      

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# Parkinson's Disease Detection

if selected=="Parkinson Disease Prediction":
    # page title
    st.title('Parkinson Disease Prediction using Machine Learning')
    st.image('img/parkinson.jpg')
    st.header('DataSet')
    st.dataframe(parkinsons_dataset)
    st.write("##")

    # input feature
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.number_input('MDVP:RAP')
        
    with col2:
        PPQ = st.number_input('MDVP:PPQ')
        
    with col3:
        DDP = st.number_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.number_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.number_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.number_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.number_input('MDVP:APQ')
        
    with col4:
        DDA = st.number_input('Shimmer:DDA')
        
    with col5:
        NHR = st.number_input('NHR')
        
    with col1:
        HNR = st.number_input('HNR')
        
    with col2:
        RPDE = st.number_input('RPDE')
        
    with col3:
        DFA = st.number_input('DFA')
        
    with col4:
        spread1 = st.number_input('spread1')
        
    with col5:
        spread2 = st.number_input('spread2')
        
    with col1:
        D2 = st.number_input('D2')
        
    with col2:
        PPE = st.number_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):                        
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
        if parkinsons_prediction[0] == 0:
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)


# --------------------------------------------------
# --------------------------------------------------
# social media link

# with st.sidebar:
#     link = option_menu("Contact Me",
#                             ["linkedin", "github", "twitter", "Nothing"], 
#                             icons=['linkedin', 'github', 'twitter', 'card-text'], menu_icon="justify", default_index=3)

# if link=="linkedin":
#     webbrowser.open_new_tab('https://www.linkedin.com/in/anurag-verma-865218183/')
# if link=="github":
#     webbrowser.open_new_tab('https://github.com/AnuragV314')    
# if link=="twitter":
#     webbrowser.open_new_tab('https://twitter.com/Anurag_S314') 


