import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sb


def Properties():
    col1, col2 = st.columns([2, 2])
    with st.expander("Select distributions"):
        with col2:
            train_mean = st.slider(
                label='Select a mean for training set',
                min_value=-10.00,value=0.00, max_value=10.00, key='train_mean', step=0.01)

            train_std_dev = st.slider(
                label='Select a std dev for training set',
                min_value=0.01,value=1.00,max_value= 10.00, key='train_std_dev', step=0.01)

            test_mean = st.slider(
                label='Select a mean for test set',
                min_value=-10.00, max_value=10.00, key='y slider')

            test_std_dev = st.slider(
                label='Select a std dev for test set',
                min_value=0.01,value=1.00, max_value=10.00, key='test_std_dev',step=0.01)

        with col1:
            st.subheader("A gaussian distribution")
            x = np.linspace(-10, 10, 100)
            train_distribution = sp.stats.norm.pdf(x, loc=train_mean, scale=train_std_dev)
            test_distribution = sp.stats.norm.pdf(x, loc=test_mean, scale=test_std_dev)

            fig1 = sb.lineplot(x=x, y=train_distribution, label='Train')
            sb.lineplot(x=x, y=test_distribution, label='Test')
            st.pyplot(plt)
