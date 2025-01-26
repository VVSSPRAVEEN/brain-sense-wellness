
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('models/random_forest_model.joblib')
scaler = joblib.load('models/scaler.joblib')

def predict_mental_health(input_data):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0]

# Create the Streamlit app
st.title('Brain Sense Wellness')
st.write('Mental Health Prediction System')

# Create input form
st.sidebar.header('User Input Parameters')

def user_input_features():
    emotions = st.sidebar.slider('Self Reported Emotions (1-10)', 1, 10, 5)
    sleep = st.sidebar.slider('Sleep Hours', 0, 12, 7)
    lifestyle = st.sidebar.slider('Lifestyle Activities (1-10)', 1, 10, 5)
    eating = st.sidebar.slider('Eating Habits (1-5)', 1, 5, 3)
    sentiment = st.sidebar.slider('Sentiment Scores (0-1)', 0.0, 1.0, 0.5)
    linguistic = st.sidebar.slider('Linguistic Features (0-1)', 0.0, 1.0, 0.5)
    engagement = st.sidebar.slider('Social Media Engagement (hours)', 0, 12, 2)
    age = st.sidebar.slider('Age', 18, 80, 30)
    occupation = st.sidebar.selectbox('Occupation', ['Student (0)', 'Employed (1)', 'Unemployed (2)'])
    social = st.sidebar.selectbox('Social Status', ['Single (0)', 'Married (1)', 'Divorced (2)'])
    stress = st.sidebar.slider('Work Stress (1-10)', 1, 10, 5)
    life_events = st.sidebar.selectbox('Major Life Events Recently?', ['No (0)', 'Yes (1)'])
    financial = st.sidebar.slider('Financial Situation (1-5)', 1, 5, 3)
    
    # Extract numeric values from selection
    occupation = int(occupation.split('(')[1].split(')')[0])
    social = int(social.split('(')[1].split(')')[0])
    life_events = int(life_events.split('(')[1].split(')')[0])
    
    data = {
        'self_reported_emotions': emotions,
        'sleep_habit': sleep,
        'lifestyle_activities': lifestyle,
        'eating_habits': eating,
        'sentiment_scores': sentiment,
        'linguistic_features': linguistic,
        'engagement_metrics': engagement,
        'age': age,
        'occupation': occupation,
        'social_status': social,
        'work_stress': stress,
        'major_life_events': life_events,
        'financial_situation': financial
    }
    return data

# Get user input
user_data = user_input_features()

# Show user input
st.subheader('User Input:')
st.write(pd.DataFrame([user_data]))

# Make prediction
if st.button('Predict Mental Health Status'):
    prediction = predict_mental_health(user_data)
    st.subheader('Prediction:')
    
    status_map = {
        0: 'Healthy',
        1: 'Mild Concern',
        2: 'Moderate Concern',
        3: 'Severe Concern'
    }
    
    status = status_map[prediction]
    if prediction == 0:
        st.success(f'Mental Health Status: {status}')
    elif prediction == 1:
        st.info(f'Mental Health Status: {status}')
    elif prediction == 2:
        st.warning(f'Mental Health Status: {status}')
    else:
        st.error(f'Mental Health Status: {status}')
    
    st.subheader('Recommendations:')
    if prediction == 0:
        st.write("- Maintain your current healthy lifestyle")
        st.write("- Continue regular exercise and good sleep habits")
        st.write("- Practice preventive mental wellness activities")
    elif prediction == 1:
        st.write("- Consider talking to a friend or family member about your feelings")
        st.write("- Improve sleep schedule and physical activity")
        st.write("- Try stress-reduction techniques like meditation")
    elif prediction == 2:
        st.write("- Consider consulting a mental health professional")
        st.write("- Focus on self-care and stress management")
        st.write("- Maintain regular sleep schedule and healthy diet")
    else:
        st.write("- Strongly recommend seeking professional help")
        st.write("- Contact mental health support services")
        st.write("- Reach out to trusted friends or family members")
