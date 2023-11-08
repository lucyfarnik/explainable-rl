import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random


class Properties:
    col1, col2 = st.columns([2, 2])

    with st.expander("Select distributions"):
        with col1:
            st.subheader("A gaussian distribution")
            arr_1 = [random.gauss(3, 1) for _ in range(400)]
            arr_2 = [random.gauss(4, 2) for _ in range(400)]

            bins = np.linspace(-10, 10, 100)

            fig_1, (ax_1, ax_2) = plt.subplots(1, 2)
            fig_1.set_figwidth(4)
            fig_1.set_figheight(2)
            ax_1.hist(arr_1, bins, alpha=0.5)
            ax_2.hist(arr_2, bins, alpha=0.5)
        col1.pyplot(fig_1)

        with col2:
            st.write("Test")
        #     st.subheader("A uniform distribution")
        #     arr_2 = np.random.uniform(5, 15, size=20)
        #     fig_2, ax_2 = plt.subplots()
        #     fig_2.set_figwidth(2)
        #     fig_2.set_figheight(1)
        #     ax_2.hist(arr_2, bins=20)
        # col2.pyplot(fig_2)
