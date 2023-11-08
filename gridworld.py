import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy as sp


with st.container():
    st.write("This is a placeholder for the gridworld animation")


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

    #
    #     bins = np.linspace(-10,10,100)
    #
    #     fig_1, (ax_1, ax_2) = plt.subplots(1, 2)
    #     fig_1.set_figwidth(4)
    #     fig_1.set_figheight(2)
    #     ax_1.hist(arr_1, bins, alpha=0.5)
    #     ax_2.hist(arr_2, bins, alpha=0.5)
    # col1.pyplot(fig_1)




    #     st.subheader("A uniform distribution")
    #     arr_2 = np.random.uniform(5, 15, size=20)
    #     fig_2, ax_2 = plt.subplots()
    #     fig_2.set_figwidth(2)
    #     fig_2.set_figheight(1)
    #     ax_2.hist(arr_2, bins=20)
    # col2.pyplot(fig_2)


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

col3, col4 = st.columns([2,2])

with st.container():
    st.button('Up')
with st.container():
    with col3:
        st.button('Left')
    with col4:
        st.button('Right')
with st.container():
    st.button('Down')



