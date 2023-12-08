# WORK IN PROGRESS — Explainable RL

The concept of misgeneralization in artificial intelligence refers to the tendency of AI systems to apply learned patterns or behaviors in contexts where they are not applicable or intended.
In reinforcement learning specifically, this issue is amplified due to the dynamic nature of learning from environmental feedback.
Mismatches between training environments and real-world applications often lead to suboptimal or even potentially harmful decisions.

Our interactive dashboard allows users to visually explore this concept by enabling the user to specify the training distribution with respect to certain parameters of an RL environment, and then train an agent on this distribution and observe its behavior off-distribution.

See the docs on [GitHub Pages](https://iaitp.github.io/2023-The-Paper-Clippers/)

## Installation
1. Clone the repo:`git clone https://github.com/iaitp/2023-The-Paper-Clippers.git`
2. Install the dependencies: `pip install -r requirements.txt`
3. Run the Streamlit App: `streamlit run app.py`
4. Open your web browser and navigate to [http://localhost.8501]