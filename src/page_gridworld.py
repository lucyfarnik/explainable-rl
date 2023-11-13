import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sb
# import scipy as sp

from src.component_property import agent_position_select, properties, progress


with st.container():
    st.write("This is a placeholder for the gridworld animation")

agent_x_pos, agent_y_pos = agent_position_select()
# with st.expander("Select initial position of the agent"):
#     x_vals = st.slider(label='This is the initial x-position',
#                        min_value=1.00, value=1.00, max_value=10.00)
#     y_vals = st.slider(label='This is the initial y-position',
#                        min_value=1.00, value=1.00, max_value=10.00)

# with st.expander("Select distributions"):
#
#     with col2:
#         train_mean = st.slider(
#             label='Select a mean for training set',
#             min_value=-10.00,value=0.00, max_value=10.00, key='train_mean', step=0.01)
#
#         train_std_dev = st.slider(
#             label='Select a std dev for training set',
#             min_value=0.01,value=1.00,max_value= 10.00, key='train_std_dev', step=0.01)
#
#         test_mean = st.slider(
#             label='Select a mean for test set',
#             min_value=-10.00, max_value=10.00, key='y slider')
#
#         test_std_dev = st.slider(
#             label='Select a std dev for test set',
#             min_value=0.01,value=1.00, max_value=10.00, key='test_std_dev',step=0.01)
#
#     with col1:
#         st.subheader("A gaussian distribution")
#         x = np.linspace(-10, 10, 100)
#         train_distribution = sp.stats.norm.pdf(x, loc=train_mean, scale=train_std_dev)
#         test_distribution = sp.stats.norm.pdf(x, loc=test_mean, scale=test_std_dev)
#
#         fig1 = sb.lineplot(x=x, y=train_distribution, label='Train')
#         sb.lineplot(x=x, y=test_distribution, label='Test')
#         st.pyplot(plt)

position_train_mean, position_train_std_dev, position_test_mean, position_test_std_dev = properties("position", 0.01, 10.00)

latest_iteration = st.empty()
bar = st.progress(0)
"Start a computation..."
progress()
"...and we\'re done"

# import time
#
# 'Starting a computation...'
# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)
#
# for i in range(10):
#     # Update the progress bar with each iteration.
#     latest_iteration.text(f'Iteration {i+1}')
#     bar.progress(i + 1)
#     time.sleep(0.1)
# '...and now we\'re done!'


with st.container():
    st.button('Up')
    st.button('Right')
    st.button('Left')
    st.button('Down')
