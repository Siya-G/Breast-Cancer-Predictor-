import streamlit as st
import pickle
import numpy as np

# load the model
with open('cancer_model.pkl', 'rb') as f:
    model = pickle.load(f)

#add a title and description
st.title('Cancer Predictor App')
st.subheader('Warning: This app is for educational purposes only. It is not a substitute for professional medical advice.')

st.write('Enter the following details to predict your breastcancer chances')

worst_perimeter = st.number_input('Worst Perimeter (must be a number between 0 and 250)', min_value=0, max_value=250, value=0)
worst_concave_points = st.number_input('Worst Concave Points (must be a number between 0 and 10)', min_value=0, max_value=10, value=0)
mean_texture = st.number_input('Mean Texture (must be a number between 0 and 40)', min_value=0, max_value=40, value=0)
worst_texture = st.number_input('Worst Texture (must be a number between 0 and 40)', min_value=0, max_value=40, value=0)
mean_concave_points = st.number_input('Mean Concave Points (must be a number between 0 and 10)', min_value=0, max_value=10, value=0)
concave_points_error = st.number_input('Concave Points Error (must be a number between 0 and 0.1)', min_value=0.0, max_value=0.1, value=0.0)
mean_perimeter = st.number_input('Mean Perimeter (must be a number between 0 and 250)', min_value=0, max_value=250, value=0)
area_error = st.number_input('Area Error (must be a number between 0 and 1000)', min_value=0, max_value=1000, value=0)
mean_smoothness = st.number_input('Mean Smoothness (must be a number between 0 and 1)', min_value=0.0, max_value=1.0, value=0.0)
worst_area = st.number_input('Worst Area (must be a number between 0 and 1000)', min_value=0, max_value=1000, value=0)
# create a button to predict the cancer chances

if st.button('Predict Cancer Chances'):

    input_data = np.array([[worst_perimeter, worst_concave_points, mean_texture, worst_texture, mean_concave_points, concave_points_error, mean_perimeter, area_error, mean_smoothness, worst_area]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.write('You have a high chance of having breast cancer according to the data. Please consult a doctor for further diagnosis.')
    else:
        st.write('You have a low chance of having breast cancer according to the data. Please consult a doctor for more accurate diagnosis.')