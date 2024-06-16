# Dicoding Analisis Data dengan Python
[Air Quality Data Dashboard Streamlit App](https://rizkysrdicoding.streamlit.app/)

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)

## Overview
This project is a data analysis and visualization project focused on air quality data. It includes code for data wrangling, exploratory data analysis (EDA), and a Streamlit dashboard for interactive data exploration. This project aims to analyze data on the air quality Dataset.

## Project Structure
- `dashboard/`: This directory contains dashboard.py which is used to create dashboards of data analysis results.
- `data/`: Directory containing the raw CSV data files.
- `Analisis_Data.ipynb/`: This file is used to perform data analysis.
- `requirements.txt/`: This file lists all the modules needed.
- `url.txt/`: This file contain the streamlit dashboard url.
- `README.md`: This documentation file.

## Installation
1. Clone this repository to your local machine:
```
git clone https://github.com/Rizky-SR/dicoding-analisis-data-dengan-python.git
```
2. Go to the project directory
```
cd dicoding-analisis-data-dengan-python
```
3. Install the required Python packages by running:
```
pip install -r requirements.txt
```

## Usage
1. **Data Wrangling**: Data wrangling scripts are available in the `Analisis_Data.ipynb` file to prepare and clean the data.

2. **Exploratory Data Analysis (EDA)**: Explore and analyze the data using the provided Python scripts. EDA insights can guide your understanding of air quality public data patterns.

3. **Visualization**: Run the Streamlit dashboard for interactive data exploration:

```
cd Analisis_Data.ipynb/dashboard
streamlit run dashboard.py
```
Access the dashboard in your web browser at `https://rizkysrdicoding.streamlit.app/`.

## Data Sources
The project uses Air Quality Dataset from [Belajar Analisis Data dengan Python's Final Project](https://github.com/marceloreis/HTI/tree/master/PRSA_Data_20130301-20170228) offered by [Dicoding](https://www.dicoding.com/).
