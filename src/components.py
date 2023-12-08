"""Higher level components for the streamlit app."""
import numpy as np
import pandas as pd
import streamlit as st
import scipy

from src.parameter import Parameter


def input_section(parameters: list[Parameter]) -> None:
    """The input section of the page.
    Renders a streamlit expander with a tab for each parameter.

    Args:
        parameters (list[Parameter]): The parameters of the environment that the user can adjust.
    """
    st.header("Inputs")
    # with st.expander("Inputs", expanded=True):
    tabs = st.tabs([param.name for param in parameters])
    for i, param in enumerate(parameters):
        with tabs[i]:
            properties(param)


def properties(parameter: Parameter) -> None:
    """A tab in the input section for a single parameter.
        Allows the user to adjust the mean and standard deviation of the parameter for the training and test sets.
        Displays the distribution of the parameter on a graph.
        The values of the sliders are automatically stored in the session state by streamlit.


    Args:
        parameter (Parameter): The parameter to be rendered.
    """
    # Set up 2 equal width columns
    col1, col2 = st.columns([1, 1])

    # Render sliders
    distributions = {}
    with col2:
        for mode in ["train", "test"]:
            # Slider for mean
            mean = st.slider(
                label=f"Select a mean {parameter.name.lower()} for {mode}ing set",
                min_value=parameter.min,
                value=parameter.default,
                max_value=parameter.max,
                key=f"{parameter.key}_{mode}_mean",
                step=0.1,
            )

            # Slider for std dev
            std_dev = st.slider(
                label=f"Select a std dev for {mode}ing set",
                min_value=0.01,
                value=parameter.default * 0.2
                or 0.1,  # approx. "Or 0.1" in case default is 0.
                max_value=parameter.max,
                key=f"{parameter.key}_{mode}_std_dev",
                step=0.01,
            )
            distributions[mode] = scipy.stats.norm(mean, std_dev)

    # Render graph
    with col1:
        # Header
        st.subheader(parameter.name_with_unit)

        # Generate x-axis of graph
        x = np.linspace(parameter.min, parameter.max, 100)

        x_label = parameter.name_with_unit

        # Generate data for graph
        data = pd.DataFrame(
            {
                x_label: x,
                "Train": distributions["train"].pdf(x),
                "Test": distributions["test"].pdf(x),
            }
        )

        # Plot graph
        st.line_chart(data, x=x_label, y=["Train", "Test"])
