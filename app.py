from st_pages import Page, show_pages, add_page_title

add_page_title()  # By default this adds indentation

# Now to specify which pages should be in the sidebar and what their titles/icons are

show_pages(
    [
        Page("src/page_pole.py", "Pole"),
        Page("src/page_gridworld.py", "Gridworld"),
    ]
)
