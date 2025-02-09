# 🏀 NBA Analysis and Visualization

This repository focuses on analyzing NBA player performance data across multiple dimensions such as age, position, and era. It aggregates season‐long statistics to provide insights into how players’ per–game production evolves over their careers and how league trends have shifted over time. The repository includes interactive visualizations that compare the performance curves of superstar players with league averages, highlighting key performance milestones and trends over different decades.

This repository also explores predictive modeling by developing regression models that forecast turnovers based on assists and personal fouls based on defensive statistics like steals and blocks. In addition, it features a classification model that predicts a player’s position from their performance metrics. These models offer insights into the relationships between various in–game statistics and serve as tools to quantify player contributions in different aspects of the game.

This repository further delves into player similarity analysis by using unsupervised learning techniques. An autoencoder is employed to compress the multidimensional statistical profiles into latent representations, and nearest neighbor algorithms identify players with similar performance profiles. Dimensionality reduction methods are then used to visualize these relationships, providing a nuanced view of player comparisons across seasons.

For more information about the training process, please check the `nba-vis.ipynb` file in the `training` folder.

[Check here to see my other ML projects and tasks](https://github.com/verneylmavt/ml-model).

## 🎈 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://verneylogyt-nba-vis.streamlit.app/)

![Demo GIF](https://github.com/verneylmavt/st-nba-vis/blob/main/assets/demo.gif)

If you encounter message `This app has gone to sleep due to inactivity`, click `Yes, get this app back up!` button to wake the app back up.

## ⚙️ Running Locally

If the demo page is not working, you can fork or clone this repository and run the application locally by following these steps:

<!-- ### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- pip (Python Package Installer)

### Installation Steps -->

1. Clone the repository:

   ```bash
   git clone https://github.com/verneylmavt/st-nba-vis.git
   cd st-nba-vis
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## ⚖️ Acknowledgement

I acknowledge the use of the **NBA Stats (1947-present)** dataset provided by **Sumitro Datta** on Kaggle. This dataset has been instrumental in conducting the research and developing this project.

- **Dataset Name**: NBA Stats (1947-present)
- **Source**: [https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats)
- **License**: Creative Commons 1.0
- **Description**: This dataset contains player statistics from the NBA, ABA, and BAA leagues from 1947 to the present. It includes individual player totals such as points, assists, rebounds, and other relevant performance metrics.

I deeply appreciate the efforts of Sumitro Datta in making this dataset available.
