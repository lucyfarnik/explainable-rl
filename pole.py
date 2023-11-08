import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

with st.container():
    st.write("This is a placeholder for the pole animation")

with st.expander("Select initial angle of pole"):
    angle = st.slider("Initial angle in degrees", 0, 180, 1)
with st.expander("Select the mass of the pole"):
    mass = st.slider("Pole mass in kilograms", 0, 100, 1)

col1, col2 = st.columns([2, 2])

with st.expander("Select distributions"):
    with col1:
        st.subheader("A gaussian distribution")
        arr_1 = [random.gauss(3, 1) for _ in range(400)]
        arr_2 = [random.gauss(4, 2) for _ in range(400)]

        bins = np.linspace(-10,10,100)

        fig_1, (ax_1, ax_2) = plt.subplots(1, 2)
        fig_1.set_figwidth(4)
        fig_1.set_figheight(2)
        ax_1.hist(arr_1, bins, alpha=0.5)
        ax_2.hist(arr_2, bins, alpha=0.5)
    col1.pyplot(fig_1)

    with col2:
        st.subheader("A uniform distribution")
        arr_2 = np.random.uniform(angle, mass, size=20)
        fig_2, ax_2 = plt.subplots()
        fig_2.set_figwidth(2)
        fig_2.set_figheight(1)
        ax_2.hist(arr_2, bins=20)
    col2.pyplot(fig_2)


import time

'Starting a computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(10):
    # Update the progress bar with each iteration.
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i + 1)
    time.sleep(0.1)

'...and now we\'re done!'
