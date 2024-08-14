"""
Description: This application compares two tables in Snowflake
Author: Mitch Zink
Last Updated: 4/6/2024
"""

import pandas as pd  # Data manipulation
import snowflake.connector
import streamlit as st  # UI
import plotly.express as px  # Interactive visualizations
import concurrent.futures  # Parallel execution of queries
import sqlparse  # Formatting SQL queries
import time


# Global variables to store generated queries
generated_aggregate_queries = []
generated_column_queries = []
generated_row_queries = []
generated_schema_queries = []


# Function to display generated queries for a specific analysis section
def display_generated_queries_for_section(queries, section_name):
    """
    Function to display generated queries for a specific analysis section.
    """
    if queries:  # Check if there are any queries to display
        with st.expander(f"{section_name} Queries 👨‍💻"):
            for query in queries:
                st.code(query, language="sql")


# Function to query data from Snowflake
def fetch_data(ctx, query):
    return pd.read_sql(query, ctx)


# AGGREGATE ANALYSIS FUNCTIONS - START
# Function to fetch schema details from Snowflake
def agg_analysis_fetch_schema(ctx, catalog, schema, table_name, filter_conditions=""):
    # First attempt with normal filter
    where_clause = f" AND {filter_conditions}" if filter_conditions else ""
    schema_query = f"""
    SELECT COLUMN_NAME, DATA_TYPE 
    FROM {catalog}.information_schema.columns 
    WHERE TABLE_CATALOG = '{catalog}' AND TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table_name}' {where_clause};
    """
    df = pd.read_sql(schema_query, ctx)
    # If no rows returned, attempt to fetch with max DELETED date using QUALIFY
    if df.empty:
        schema_query_with_deleted = f"""
        SELECT COLUMN_NAME, DATA_TYPE 
        FROM snowflake.account_usage.columns 
        WHERE TABLE_CATALOG = '{catalog}' AND TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table_name}'
        {where_clause}
        QUALIFY ROW_NUMBER() OVER(PARTITION BY COLUMN_NAME ORDER BY DELETED DESC) = 1;
        """
        df = pd.read_sql(schema_query_with_deleted, ctx)
    formatted_schema_query = sqlparse.format(
        schema_query, reindent=True, keyword_case="lower"
    )
    # Check if the formatted schema query already exists in generated_column_queries to avoid duplicates
    if formatted_schema_query not in generated_column_queries:
        generated_column_queries.append(
            formatted_schema_query
        )  # Add the formatted schema query if it's unique
    return df


# Function to perform aggregate analysis between two tables in Snowflake
def perform_aggregate_analysis(
    ctx, full_table_name1, full_table_name2, filter_conditions=""
):
    catalog1, schema1, table1 = full_table_name1.split(".")
    catalog2, schema2, table2 = full_table_name2.split(".")

    df_schema1 = agg_analysis_fetch_schema(ctx, catalog1, schema1, table1)
    df_schema2 = agg_analysis_fetch_schema(ctx, catalog2, schema2, table2)

    # Adjust the schema data frames to treat all timestamp types as equivalent
    df_schema1["DATA_TYPE"] = df_schema1["DATA_TYPE"].replace(
        ["TIMESTAMP_LTZ", "TIMESTAMP_NTZ", "TIMESTAMP_TZ"], "TIMESTAMP"
    )
    df_schema2["DATA_TYPE"] = df_schema2["DATA_TYPE"].replace(
        ["TIMESTAMP_LTZ", "TIMESTAMP_NTZ", "TIMESTAMP_TZ"], "TIMESTAMP"
    )

    matching_columns = pd.merge(
        df_schema1, df_schema2, on=["COLUMN_NAME", "DATA_TYPE"], how="inner"
    )

    # If there are no matching columns, return a message indicating the same
    if matching_columns.empty:
        return pd.DataFrame(
            {
                "Column Name": ["No matching columns for aggregation"],
                "Result": ["N/A"],
                "Table 1 Value": ["N/A"],
                "Table 2 Value": ["N/A"],
                "Description": ["N/A"],
            }
        )

    # Generate aggregate expressions for each column based on its data type
    aggregates = [
        agg_analysis_aggregate_expression(row["COLUMN_NAME"], row["DATA_TYPE"])
        for _, row in matching_columns.iterrows()
        if agg_analysis_aggregate_expression(row["COLUMN_NAME"], row["DATA_TYPE"])
    ]

    results1, results2 = None, None

    # Execute aggregate queries in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                agg_analysis_execute_aggregate_query,
                ctx,
                catalog1,
                schema1,
                table1,
                aggregates,
                filter_conditions,
            ),
            executor.submit(
                agg_analysis_execute_aggregate_query,
                ctx,
                catalog2,
                schema2,
                table2,
                aggregates,
                filter_conditions,
            ),
        ]
        for future in concurrent.futures.as_completed(futures):
            if future.result() is not None:
                if results1 is None:
                    results1 = future.result()
                else:
                    results2 = future.result()

    # If one or both tables failed to fetch results, return an error message
    if results1 is None or results2 is None:
        st.error("Failed to fetch results from one or both tables.")
        return pd.DataFrame()

    comparison_results_list = []
    for column in results1.columns:
        val1, val2 = results1[column].iloc[0], results2[column].iloc[0]
        str_val1, str_val2 = str(val1), str(val2)
        result = "Match" if str_val1 == str_val2 else "Mismatch"
        comparison_results_list.append(
            {
                "Column Name": column,
                "Result": result,
                "Table 1 Value": str_val1,
                "Table 2 Value": str_val2,
                "Description": agg_analysis_get_check_description(column),
            }
        )

    return pd.DataFrame(
        comparison_results_list,
        columns=[
            "Column Name",
            "Result",
            "Table 1 Value",
            "Table 2 Value",
            "Description",
        ],
    )

    # Compare the results of aggregate analysis between both tables
    comparison_results_list = []
    for column in results1.columns:
        val1, val2 = results1[column].iloc[0], results2[column].iloc[0]
        str_val1, str_val2 = str(val1), str(val2)
        result = "Match" if str_val1 == str_val2 else "Mismatch"

        # Append the comparison results to a list
        comparison_results_list.append(
            {
                "Column Name": column,
                "Result": result,
                "Table 1 Value": str_val1,
                "Table 2 Value": str_val2,
                "Description": agg_analysis_get_check_description(column),
            }
        )

    # Convert the list of comparison results to a DataFrame
    comparison_results = pd.DataFrame(comparison_results_list)
    return comparison_results[
        ["Column Name", "Result", "Table 1 Value", "Table 2 Value", "Description"]
    ]


# Function to generate descriptions based on aggregate column names (case-insensitive)
def agg_analysis_get_check_description(column):
    column = (
        column.lower()
    )  # Convert column name to lower case for case-insensitive comparison
    if "total_" in column:
        return "Compares the sum of numeric values in this column between both tables."
    elif "count_" in column:
        return (
            "Compares the count of non-null values in this column between both tables."
        )
    elif "approx_distinct_" in column:
        return "Compares the approximate count of distinct values in this column between both tables."
    elif "unique_" in column:
        return "Compares the count of unique text values in this column between both tables."
    elif "min_" in column:
        return "Compares the minimum value in this date or timestamp column between both tables."
    elif "max_" in column:
        return "Compares the maximum value in this date or timestamp column between both tables."
    elif "row_count" in column:
        return "Compares the total row count between both tables."
    else:
        return "Specific description not available for this check."


# Function to define aggregate expressions based on the data type of a column.
def agg_analysis_aggregate_expression(column_name, data_type):
    numeric_types = [
        "NUMBER",
        "FLOAT",
        "DECIMAL",
        "INT",
        "INTEGER",
        "BIGINT",
        "SMALLINT",
        "TINYINT",
        "DOUBLE",
        "REAL",
    ]
    text_types = ["VARCHAR", "TEXT", "STRING", "CHAR", "CHARACTER", "NCHAR", "NVARCHAR"]
    timestamp_types = [
        "DATE",
        "DATETIME",
        "TIMESTAMP",
        "TIMESTAMP_LTZ",
        "TIMESTAMP_NTZ",
        "TIMESTAMP_TZ",
    ]

    # Handling numeric types with SUM, COUNT, and APPROX_COUNT_DISTINCT
    if data_type.upper() in numeric_types:
        return f"SUM({column_name}::NUMBER) AS total_{column_name.lower()}, COUNT({column_name}) AS count_{column_name.lower()}, APPROX_COUNT_DISTINCT({column_name}) AS approx_distinct_{column_name.lower()}"
    # Handling string types with COUNT and APPROX_COUNT_DISTINCT for unique values
    elif data_type.upper() in text_types:
        return f"COUNT({column_name}) AS count_{column_name.lower()}, APPROX_COUNT_DISTINCT({column_name}) AS unique_{column_name.lower()}"
    # Handling date and timestamp types with MIN, MAX, and COUNT
    elif data_type.upper() in timestamp_types:
        # Convert timestamps to a consistent string format for proper comparison
        # This will ignore timezone information by formatting to a string without timezone.
        return (
            f"TO_CHAR(MIN({column_name}), 'YYYY-MM-DD HH24:MI:SS') AS min_{column_name.lower()}, "
            f"TO_CHAR(MAX({column_name}), 'YYYY-MM-DD HH24:MI:SS') AS max_{column_name.lower()}, "
            f"COUNT({column_name}) AS count_{column_name.lower()}"
        )
    else:
        # For data types that do not match any of the above categories, return None
        # This will exclude them from aggregate queries
        return None


# Function to execute an aggregate query on a table in Snowflake
def agg_analysis_execute_aggregate_query(
    ctx, catalog, schema, table, aggregates, filter_conditions=""
):
    where_clause = f"WHERE {filter_conditions}" if filter_conditions else ""
    aggregates_sql = ", ".join(aggregates)
    query = f"""WITH table_agg AS (
                  SELECT COUNT(*) AS row_count, {aggregates_sql}
                  FROM "{catalog}"."{schema}"."{table}"
                  {where_clause}
               )
               SELECT * FROM table_agg;"""
    formatted_query = sqlparse.format(query, reindent=True, keyword_case="lower")
    if formatted_query not in generated_aggregate_queries:  # Ensure uniqueness
        generated_aggregate_queries.append(formatted_query)
    return pd.read_sql(query, ctx)


# Function to plot the results of aggregate analysis, indicating pass/fail status for each check
def plot_aggregate_analysis_summary(aggregate_results):
    if aggregate_results.empty:
        st.error("No results to display.")
        return

    # Count occurrences of "Match" and "Mismatch" in the "Result" column
    results_count = aggregate_results["Result"].value_counts().reset_index()
    results_count.columns = ["Result", "Count"]

    # Check if results_count dataframe has both "Match" and "Mismatch" rows
    expected_results = ["Match", "Mismatch"]
    for result in expected_results:
        if result not in results_count["Result"].values:
            # Use pd.concat instead of append
            results_count = pd.concat(
                [results_count, pd.DataFrame({"Result": [result], "Count": [0]})],
                ignore_index=True,
            )

    # Ensure the order of results is consistent for the pie chart
    results_count["Result"] = pd.Categorical(
        results_count["Result"], categories=expected_results
    )
    results_count.sort_values("Result", inplace=True)

    # Create the pie chart
    fig = px.pie(
        results_count,
        names="Result",
        values="Count",
        hole=0.4,
        color_discrete_sequence=["#2980b9"],
    )

    fig.update_traces(textinfo="percent+label", textposition="inside")
    fig.update_layout(showlegend=True, title_text="")

    st.plotly_chart(fig, use_container_width=True)


# AGGREGATE ANALYSIS FUNCTIONS - END


# SCHEMA ANALYSIS FUNCTIONS - START


# Function to compare schemas of two tables in Snowflake and fetch schema details
def schema_analysis(ctx, full_table_name_1, full_table_name_2, st):
    def fetch_schema_info(ctx, database, schema):
        query = f"""
        SELECT TABLE_SCHEMA, TABLE_NAME, ROW_COUNT
        FROM "{database}".INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = '{schema}'
        AND TABLE_TYPE = 'BASE TABLE';
        """
        df = pd.read_sql(query, ctx)
        return df[["TABLE_NAME", "ROW_COUNT"]], query

    db_name_1, schema_name_1, _ = full_table_name_1.split(".")
    db_name_2, schema_name_2, _ = full_table_name_2.split(".")

    # Adjusted to ensure unique comparison across both database and schema
    if db_name_1 == db_name_2 and schema_name_1 == schema_name_2:
        st.warning(
            "Skipping schema analysis because the database and schema are the same."
        )
        return None, []

    df_schema1, query1 = fetch_schema_info(ctx, db_name_1, schema_name_1)
    df_schema2, query2 = fetch_schema_info(ctx, db_name_2, schema_name_2)

    df_schema1.rename(
        columns={"ROW_COUNT": f"{db_name_1}.{schema_name_1} Row Count"}, inplace=True
    )
    df_schema2.rename(
        columns={"ROW_COUNT": f"{db_name_2}.{schema_name_2} Row Count"}, inplace=True
    )

    # Ensure that the merge and comparison logic accounts for both database and schema names
    df_merged = pd.merge(df_schema1, df_schema2, on="TABLE_NAME", how="outer")
    df_merged["Test"] = df_merged.apply(
        lambda row: (
            "Match"
            if row[f"{db_name_1}.{schema_name_1} Row Count"]
            == row[f"{db_name_2}.{schema_name_2} Row Count"]
            else "Mismatch"
        ),
        axis=1,
    )

    test_counts = df_merged["Test"].value_counts().reset_index()
    test_counts.columns = ["Test Result", "Count"]

    # Donut Chart for Test Results
    fig = px.pie(
        test_counts,
        names="Test Result",
        values="Count",
        hole=0.4,  # Creates a donut chart
        color_discrete_sequence=["#2980b9"],
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

    formatted_queries = [
        sqlparse.format(query, reindent=True, keyword_case="lower")
        for query in [query1, query2]
    ]
    return df_merged, formatted_queries


# SCHEMA ANALYSIS FUNCTIONS - END


# ROW LEVEL ANALYSIS FUNCTIONS - START
# Function to plot the comparison results using a bar chart
def row_level_analysis_plot_comparison_results(
    differences,
    matched_but_different,
    rows_fetched_from_first,
    rows_fetched_from_second,
):
    # Check for dummy rows (indicating a complete match) and adjust counts
    if not differences.empty and differences.iloc[0, 0] == "All rows match":
        differences_count = 0
    else:
        differences_count = len(differences)

    if (
        not matched_but_different.empty
        and matched_but_different.iloc[0, 0] == "All rows match"
    ):
        matched_but_different_count = 0
    else:
        matched_but_different_count = len(matched_but_different)

    # Prepare data for the plot with adjusted counts
    counts = {
        "Row Discrepancies": differences_count,  # Renamed from "Differences Between Tables"
        "Column Discrepancies": matched_but_different_count,  # Renamed from "Detailed Differences for Matched Rows"
        "Rows Fetched from Table 1": rows_fetched_from_first,
        "Rows Fetched from Table 2": rows_fetched_from_second,
    }
    data = pd.DataFrame(list(counts.items()), columns=["Type", "Count"])

    # Plotting adjustments to prevent clipping
    fig = px.bar(
        data,
        x="Type",
        y="Count",
        text="Count",
        color="Type",
        color_discrete_sequence=["#2980b9"],
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Count",
        margin=dict(t=60),
    )  # Increase top margin to prevent clipping
    st.plotly_chart(
        fig, use_container_width=True
    )  # Ensure the chart uses the container width to fit properly


# Function to compare two dataframes by a specific key column
def row_level_analysis_compare_dataframes_by_key(df1, df2, key_column):
    # Handling the scenario where both dataframes are identical
    if df1.equals(df2):
        # Creating a dummy row to indicate that the dataframes match completely
        dummy_row = pd.DataFrame(
            [
                {
                    key_column: "All rows match",
                    "Column": "N/A",
                    "Table 1": "N/A",
                    "Table 2": "N/A",
                }
            ]
        )
        return (
            dummy_row,
            dummy_row,
        )  # Returning dummy rows for both differences and matched_but_different

    # Ensure column names in both dataframes are lowercase for consistency
    df1.columns = df1.columns.str.lower()
    df2.columns = df2.columns.str.lower()
    key_column_lower = key_column.lower()

    # Verify that the key column exists in both dataframes
    if key_column_lower not in df1.columns or key_column_lower not in df2.columns:
        st.error(f"Key column '{key_column}' not found in one or both tables.")
        return pd.DataFrame(), pd.DataFrame()

    # Merge dataframes to find differences
    merged_df = df1.merge(
        df2,
        on=key_column_lower,
        how="outer",
        indicator=True,
        suffixes=("_table_1", "_table_2"),
    )
    differences = merged_df[merged_df["_merge"] != "both"]

    compare_cols = [
        col for col in df1.columns if col != key_column_lower and col in df2.columns
    ]
    diff_data = {"key": [], "column": [], "Table 1": [], "Table 2": []}
    for _, row in merged_df.iterrows():
        for col in compare_cols:
            table_1_val, table_2_val = row[f"{col}_table_1"], row[f"{col}_table_2"]
            if pd.isna(table_1_val) and pd.isna(table_2_val):
                continue  # Skip comparison if both values are NaN
            if table_1_val != table_2_val:
                diff_data["key"].append(row[key_column_lower])
                diff_data["column"].append(col)
                diff_data["Table 1"].append(table_1_val)
                diff_data["Table 2"].append(table_2_val)

    matched_but_different = pd.DataFrame(diff_data)
    return differences, matched_but_different


def row_level_analysis_fetch_data(
    ctx, full_table_name, key_column, filter_conditions="", row_count=50
):
    """
    Fetches data from Snowflake for row level analysis, capturing and storing the executed SQL queries.
    """
    base_filter = f"WHERE {filter_conditions}" if filter_conditions else ""
    # Define queries for fetching data from the top and bottom of the table based on the key column
    queries = [
        f"SELECT * FROM {full_table_name} {base_filter} ORDER BY {key_column} ASC LIMIT {row_count}",
        f"SELECT * FROM {full_table_name} {base_filter} ORDER BY {key_column} DESC LIMIT {row_count}",
    ]

    dfs = []
    for query in queries:
        formatted_query = sqlparse.format(query, reindent=True, keyword_case="lower")
        # Store the formatted queries for display or further use
        generated_row_queries.append(formatted_query)
        try:
            df = pd.read_sql(query, ctx)
            dfs.append(df)
        except Exception as exc:
            st.error(f"Query execution failed: {exc}")
            return pd.DataFrame()

    # Concatenate dataframes to combine the top and bottom fetched rows
    df_combined = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
    return df_combined


def row_level_analysis(
    ctx,
    full_table_name1,
    full_table_name2,
    key_column,
    filter_conditions="",
    row_count=50,
):
    """
    Performs row level analysis between two tables, including fetching data, comparing dataframes,
    and plotting comparison results.
    """
    df1 = row_level_analysis_fetch_data(
        ctx, full_table_name1, key_column, filter_conditions, row_count
    )
    df2 = row_level_analysis_fetch_data(
        ctx, full_table_name2, key_column, filter_conditions, row_count
    )

    differences, matched_but_different = row_level_analysis_compare_dataframes_by_key(
        df1, df2, key_column
    )

    row_level_analysis_plot_comparison_results(
        differences, matched_but_different, len(df1), len(df2)
    )

    return differences, matched_but_different


# ROW LEVEL ANALYSIS FUNCTIONS - END


# COLUMN ANALYSIS FUNCTIONS - START
# Function to compare schemas of two tables in Snowflake and fetch schema details
def column_analysis(ctx, full_table_name1, full_table_name2):
    catalog1, schema1, table1 = full_table_name1.split(".")
    catalog2, schema2, table2 = full_table_name2.split(".")

    df_schema1 = agg_analysis_fetch_schema(ctx, catalog1, schema1, table1)
    df_schema2 = agg_analysis_fetch_schema(ctx, catalog2, schema2, table2)

    comparison_results = pd.merge(
        df_schema1,
        df_schema2,
        on="COLUMN_NAME",
        how="outer",
        indicator="Column Presence",
    )
    comparison_results["Column Presence"] = comparison_results["Column Presence"].map(
        {
            "left_only": "Only in Table 1",
            "right_only": "Only in Table 2",
            "both": "In Both Tables",
        }
    )

    comparison_results = comparison_results.rename(
        columns={
            "DATA_TYPE_x": "Data Type Table 1",
            "DATA_TYPE_y": "Data Type Table 2",
        }
    )

    comparison_results["Data Type Match"] = comparison_results.apply(
        lambda row: (
            "Match"
            if row["Data Type Table 1"] == row["Data Type Table 2"]
            else "Mismatch"
        ),
        axis=1,
    )

    # Set "Data Type Match" to "N/A" for columns not present in both tables
    comparison_results.loc[
        comparison_results["Column Presence"] != "In Both Tables", "Data Type Match"
    ] = "N/A"

    return comparison_results


# Function to plot schema comparison results using a bar chart
def column_analysis_comparison_results(column_comparison_results):
    counts = column_comparison_results["Column Presence"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        text="Count",
        color_discrete_sequence=["#2980b9"],
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(
        xaxis_title="Column Presence",
        yaxis_title="Count",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        showlegend=False,
    )
    return fig  # Return the figure instead of directly displaying it


# Function to plot a summary of the schema comparison focusing on data type matches/mismatches
def plot_column_comparison_summary(column_comparison_results):
    data_type_counts = (
        column_comparison_results["Data Type Match"]
        .value_counts()
        .reindex(["Match", "Mismatch"], fill_value=0)
        .reset_index()
    )
    data_type_counts.columns = ["Match Status", "Count"]
    fig_data_types = px.bar(
        data_type_counts,
        x="Match Status",
        y="Count",
        text="Count",
        color_discrete_sequence=["#2980b9"],
    )
    fig_data_types.update_traces(texttemplate="%{text}", textposition="outside")
    fig_data_types.update_layout(
        xaxis_title="Data Type Match",
        yaxis_title="Count",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        legend_title="Match Status",
    )
    return fig_data_types  # Return the figure instead of directly displaying it


# Function to display column analysis charts side by side
def display_column_analysis_charts(column_comparison_results):
    # Generate both figures for the column analysis
    fig1 = column_analysis_comparison_results(column_comparison_results)
    fig2 = plot_column_comparison_summary(column_comparison_results)

    # Use Streamlit's columns feature to display figures side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)


# COLUMN ANALYSIS FUNCTIONS - END


# # Main function to run the Snowflake Table Comparison Tool
def main():
    st.set_page_config(
        page_title="Snowflake Table Comparison Tool",
        page_icon="❄️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("❄️ Snowflake Table Comparison Tool")
    # Initialize flags for aggregate, row level, column, and schema analysis
    agg_analysis_flag = "❌"
    row_level_analysis_flag = "❌"
    column_analysis_flag = "⏳"  # Placeholder for column analysis
    schema_analysis_flag = "⏳"  # Placeholder for schema analysis

    # Configuration sidebar setup within a form
    with st.sidebar.form(key="config_form"):
        st.sidebar.header("Configuration ⚙️")
        user = st.sidebar.text_input("Username 🧑‍💼").upper()
        use_external_browser_auth = st.sidebar.checkbox(
            "Use External Browser Authentication"
        )
        password = ""
        authenticator = "externalbrowser" if use_external_browser_auth else "snowflake"
        if not use_external_browser_auth:
            password = st.sidebar.text_input("Password 🔒", type="password")
        account = st.sidebar.text_input("Account 🏦").upper()
        warehouse = st.sidebar.text_input("Warehouse 🏭").upper()

        row_count = st.sidebar.slider(
            "Number of Rows from Top/Bottom",
            min_value=10,
            max_value=1000,
            value=50,
            step=10,
        )
        key_column = st.sidebar.text_input(
            "Unique Key Column 🗝️", placeholder="UNIQUE_KEY"
        ).upper()
        full_table_name1 = st.sidebar.text_input(
            "Table 1 ❄️", placeholder="DATABASE.SCHEMA.TABLE"
        ).upper()
        full_table_name2 = st.sidebar.text_input(
            "Table 2 ❄️", placeholder="DATABASE.SCHEMA.TABLE"
        ).upper()
        filter_conditions = st.sidebar.text_area(
            "Filter conditions (optional) ✨",
            placeholder="EMAIL = 'mitch@example.com' AND DATE::DATE >= '2024-01-01'::DATE",
        )

    if "progress" not in st.session_state:
        st.session_state.progress = 0
    if "status_message" not in st.session_state:
        st.session_state.status_message = "Ready"

    def update_progress(new_progress, new_message):
        if new_progress is not None:
            st.session_state.progress = new_progress
            progress_bar.progress(st.session_state.progress)
        if new_message:
            st.session_state.status_message = new_message
            status_message.text(st.session_state.status_message)

    # Placeholders for progress bar and status message
    progress_bar = st.progress(st.session_state.progress)
    status_message = st.empty()
    status_message.text(st.session_state.status_message)

    if st.sidebar.button("Run Comparison 🚀"):
        if (
            not user
            or not account
            or not warehouse
            or not full_table_name1
            or not full_table_name2
        ):
            st.error("Please fill in all required fields.")
            return

        if not full_table_name1.count(".") == 2 or not full_table_name2.count(".") == 2:
            st.error(
                "Please ensure both tables are in the format: DATABASE.SCHEMA.TABLE"
            )
            return

        try:
            # st.snow()
            update_progress(5, "Connecting to Snowflake 🏂")
            ctx = snowflake.connector.connect(
                user=user,
                account=account,
                password=password,
                authenticator=authenticator,
                warehouse=warehouse,
            )

            update_progress(15, "Working on Row Level Analysis 🏂")

            st.header("Row Level Analysis 🔎")
            with st.spinner("🏂"):
                differences, matched_but_different = row_level_analysis(
                    ctx,
                    full_table_name1,
                    full_table_name2,
                    key_column,
                    filter_conditions,
                    row_count,
                )

                if (
                    not differences.empty
                    and not differences.iloc[0, 0] == "All rows match"
                ):
                    with st.expander("Row Discrepancies ⚠️"):
                        st.dataframe(differences)

                if (
                    not matched_but_different.empty
                    and not matched_but_different.iloc[0, 0] == "All rows match"
                ):
                    with st.expander("Column Discrepancies ⚠️"):
                        st.dataframe(matched_but_different)

                display_generated_queries_for_section(generated_row_queries, "")
                # Calculate the counts for differences and matched_but_different
                differences_count = (
                    0
                    if not differences.empty
                    and differences.iloc[0, 0] == "All rows match"
                    else len(differences)
                )
                matched_but_different_count = (
                    0
                    if not matched_but_different.empty
                    and matched_but_different.iloc[0, 0] == "All rows match"
                    else len(matched_but_different)
                )

                # Set the flag for row level analysis
                if differences_count == 0 and matched_but_different_count == 0:
                    row_level_analysis_flag = "✅"

                # Update progress with the flags and line breaks
                progress_message = f"Aggregate Analysis: {agg_analysis_flag}\nRow Level Analysis: {row_level_analysis_flag}"

            update_progress(40, "Working on Column Analysis 🏂")
            st.header("Column Analysis 🔎")
            with st.spinner(" 🏂"):
                column_comparison_results = column_analysis(
                    ctx, full_table_name1, full_table_name2
                )
                column_comparison_results = column_analysis(
                    ctx, full_table_name1, full_table_name2
                )
                display_column_analysis_charts(column_comparison_results)

            with st.expander("Results 📊"):
                st.dataframe(column_comparison_results)

            display_generated_queries_for_section(generated_column_queries, "")

            update_progress(60, "Working on Schema Analysis 🏂")

            st.header("Schema Analysis 🔎")

            with st.spinner(" 🏂"):
                df_merged, formatted_queries = schema_analysis(
                    ctx, full_table_name1, full_table_name2, st
                )
                with st.expander("Results 📊"):
                    if df_merged is not None:
                        st.write(df_merged)
                display_generated_queries_for_section(formatted_queries, "")

            update_progress(80, "Working on Aggregate Analysis 🏂")
            st.header("Aggregate Analysis 🔎")

            with st.spinner(" 🏂"):
                aggregate_results = perform_aggregate_analysis(
                    ctx, full_table_name1, full_table_name2, filter_conditions
                )
                plot_aggregate_analysis_summary(aggregate_results)
                with st.expander("Results 📊"):
                    st.dataframe(aggregate_results)
                display_generated_queries_for_section(generated_aggregate_queries, "")

                # Set the flag for aggregate analysis
                if all(aggregate_results["Result"] == "Match"):
                    agg_analysis_flag = "✅"

                # Update progress with the flags and line breaks
                progress_message = f"Aggregate Analysis: {agg_analysis_flag}\nRow Level Analysis: {row_level_analysis_flag}"

            update_progress(100, "Analysis completed ✅")

            update_progress(100, progress_message)

        except Exception as e:
            update_progress(0, f"Failed to run analysis: {e}")


if __name__ == "__main__":
    main()
