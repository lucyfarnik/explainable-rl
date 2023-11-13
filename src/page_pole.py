import streamlit as st
from src.component_property import properties, progress


with st.container():
    st.write("This is a placeholder for the pole animation")

mass_train_mean, mass_train_std_dev, mass_test_mean, mass_test_std_dev = properties("mass")
angle_train_mean, angle_train_std_dev, angle_test_mean, angle_test_std_dev = properties("angle")
progress()
# with st.expander("Select initial angle and mass of pole"):
#     angle = st.slider("Initial angle in degrees", 0, 180, 1)
#     mass = st.slider("Pole mass in kilograms", 0, 100, 1)

# import time
#
# latest_iteration = st.empty()
# bar = st.progress(0)
# from property_component import progress
# progress(latest_iteration, bar)

