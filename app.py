import streamlit as st
from st_pages import Page, show_pages


def app():
    """The entry point of the streamlit app."""

    # Read the readme file and render it as markdown as a landing page.
    with open("README.md", "r") as f:
        readme = f.read()

    st.markdown(readme)

    # Now to specify which pages should be in the sidebar and what their titles/icons are
    show_pages(
        [
            Page("src/page_pole.py", "Pole"),
            Page("src/page_grid_world.py", "Grid World"),
        ]
    )


if __name__ == "__main__":
    app()
