# Snowflake Table Comparison
[![Open with Streamlit](https://img.shields.io/badge/-Open%20with%20Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://snow-table-comparison.streamlit.app/)

[![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Snowflake](https://img.shields.io/badge/-Snowflake-29B5E8?style=for-the-badge&logo=snowflake&logoColor=white)](https://snowflake.com/)
[![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

This project contains a Streamlit app for comparing two tables in Snowflake to identify differences in rows and columns. It uses Snowflake's external browser-based authentication to securely access the data.

## Roadmap
- Bug | External browser authentication doesn't work with the [streamlit hosted web version](https://snow-table-comparison.streamlit.app/), only locally 
- Feature | Better error handling and form validation for input values
- Feature | Add filters to value level analysis
- Bug | Value Level Analysis - The chart counts the dummy rows
- Documentation | Refresh the documentation and readme

## Setup Instructions

### Creating a Virtual Environment

```bash
python3 -m venv venv && source venv/bin/activate && pip3 install --upgrade pip && pip3 install -r requirements.txt 
```

### Running the Streamlit App

```bash
streamlit run app.py
```

## Examples (Snowflake Sample Data DB)

### Value Level Analysis
![Value Level Analysis 1](row_level_analysis.png)
### Column and Aggregate Analysis
![Column and Aggregate Analysis](column_and_aggregate_analysis.gif)