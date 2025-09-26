import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

# Load model safely
pipe = None
try:
    pipe = pickle.load(open('pipe.pkl','rb'))
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.title('IPL Win Predictor')

# use updated API: st.columns instead of st.beta_columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=1, value=160, step=1)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, value=0, step=1)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, value=0.0, step=0.1)
with col5:
    input_wickets = st.number_input('Wickets out', min_value=0, max_value=10, value=0, step=1)

if st.button('Predict Probability'):
    # basic validation
    if batting_team == bowling_team:
        st.error("Batting and bowling team cannot be the same.")
    elif score > target:
        st.error("Score cannot be greater than target.")
    else:
        runs_left = target - score
        balls_left = max(0, int(120 - (overs * 6)))
        wickets_remaining = max(0, 10 - int(input_wickets))

        # protect against division by zero
        crr = (score / overs) if overs > 0 else 0.0
        if balls_left <= 0:
            st.error("No balls left — match is finished or overs value invalid.")
        else:
            rrr = (runs_left * 6) / balls_left

            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [selected_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets': [wickets_remaining],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            try:
                result = pipe.predict_proba(input_df)
                loss = result[0][0]
                win = result[0][1]
                st.header(f"{batting_team} - {round(win*100,2)}%")
                st.header(f"{bowling_team} - {round(loss*100,2)}%")
            except Exception as e:
                st.error(f"Prediction failed: {e}")