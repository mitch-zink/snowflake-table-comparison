# Snowflake Table Comparison

[![Open with Streamlit](https://img.shields.io/badge/-Open%20with%20Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://snow-table-comparison.streamlit.app/)

[![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Snowflake](https://img.shields.io/badge/-Snowflake-29B5E8?style=for-the-badge&logo=snowflake&logoColor=white)](https://snowflake.com/)
[![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

## Overview

This tool analyzes differences between two Snowflake tables across four main sections:

### 1. Row Level Analysis

Compares data row-by-row to highlight discrepancies and matches between tables, ensuring granular alignment. Provides details on mismatched rows, matched rows with differing columns, and summary counts

### 2. Column Analysis

Examines columns to identify differences in column presence and data types across the two tables. This includes highlighting columns present in one table but not the other and mismatches in data types for shared columns

### 3. Aggregate Analysis

Performs aggregate checks such as sum, count, and distinct value comparisons for matching columns. Results are displayed with flags for matches or mismatches to ensure consistency in aggregate data

### 4. Schema Analysis

Compares schemas by analyzing row counts for all tables within each schema. Flags discrepancies and displays a summary of matched and mismatched tables

### 5. Date Column Analysis

Compares unique key counts grouped by month-year between the two tables, ensuring temporal consistency. Handles missing months dynamically by generating a complete date range and filling gaps with zero counts for clear visualization

## Roadmap

- Bug | External browser authentication doesn't work with the [streamlit hosted web version](https://snow-table-comparison.streamlit.app/), only locally

## Setup Instructions

### For Mac/Linux

1. **Creating a Virtual Environment and Installing Dependencies, and run the app**

```bash
python3 -m venv venv && source venv/bin/activate && pip3 install --upgrade pip && pip3 install -r requirements.txt && streamlit run app.py
```

### For Windows

1. **Allow Script Execution (if necessary)**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

2. **Creating a Virtual Environment and Installing Dependencies**

```powershell
py -m venv venv; .\venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install -r requirements.txt
```

3. **Running the Streamlit App**

```powershell
streamlit run app.py
```

## Example Run

![Sample Run](test_run.gif)
