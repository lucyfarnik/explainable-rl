import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sb
import time


def properties(name: str, min: float, max: float):
    with st.expander(f"Select distributions for {name}"):
        col1, col2 = st.columns([2, 2])
        with col2:
            train_mean = st.slider(
                label=f'Select a mean {name} for training set',
                min_value= min, value=0.00, max_value= max, key=f'{name}_train_mean', step=0.01)

            train_std_dev = st.slider(
                label='Select a std dev for training set',
                min_value=0.01, value=1.00, max_value= max, key=f'{name}_train_std_dev', step=0.01)

            test_mean = st.slider(
                label=f'Select a mean {name} for test set',
                min_value= min, value = 0.00, max_value=max, key=f'{name}_test_mean')

            test_std_dev = st.slider(
                label='Select a std dev for test set',
                min_value=0.01, value=1.00, max_value=max, key=f'{name}_test_std_dev', step=0.01)

        with col1:
            st.subheader(name.capitalize())
            x = np.linspace(-10, 10, 100)
            train_distribution = sp.stats.norm.pdf(x, loc=train_mean, scale=train_std_dev)
            test_distribution = sp.stats.norm.pdf(x, loc=test_mean, scale=test_std_dev)

            fig1 = sb.lineplot(x=x, y=train_distribution, label='Train')
            sb.lineplot(x=x, y=test_distribution, label='Test')
            st.pyplot(plt)
    return train_mean, train_std_dev, test_mean, test_std_dev


def agent_position_select():
    # with st.container:
    col_a, col_b = st.columns([2, 2])
    with col_a:
        with st.expander("Select initial position of the agent"):
            x_vals = st.slider(label='This is the initial x-position',
                               min_value=1.00, value=1.00, max_value=10.00)
            y_vals = st.slider(label='This is the initial y-position',
                                   min_value=1.00, value=1.00, max_value=10.00)
    return x_vals, y_vals


def IC_pole(colC, colD):
    colC, colD = st.columns([2, 2])
    with colC:
        with st.expander("Select initial angle and mass of pole"):
            angle = st.slider("Initial angle in degrees", 0, 180, 1)
            mass = st.slider("Pole mass in kilograms", 0, 100, 1)


# def IC_gridworld(x_vals,y_vals):
#     x_vals = st.slider(label='This is the initial x-position',
#               min_value=1.00, value=1.00, max_value=10.00)
#     y_vals = st.slider(label='This is the initial y-position',
#                   min_value=1.00, value=1.00, max_value=10.00)
#
# def IC_pole(angle, mass):
#     angle = st.slider(label='This is the initial angle of the pole in degress',
#                       min_value=0.01, value=1.00, max_value=90.00)
#     mass = st.slider(label='This is the pole mass in kilograms',
#                      min_value=1.00, value=1.00, max_value=100.00)


def progress():
    "Starting a computation..."
    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(20):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i + 1}')
        bar.progress(i + 1)
        time.sleep(0.1)
    '...and now we\'re done!'
