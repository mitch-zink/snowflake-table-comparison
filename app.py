"""
Description: This application compares two tables in Snowflake
Author: Mitch Zink
Last Updated: 4/3/2024
"""
import pandas as pd  # Data manipulation
import snowflake.connector  
import streamlit as st  # UI
import plotly.express as px  # Interactive visualizations
import concurrent.futures  # Parallel execution of aggregate queries
import sqlparse  # Formatting SQL queries
import time

# Global variables to store generated queries
generated_aggregate_queries = []
generated_column_queries = []  
generated_schema_queries = []  

# Function to query data from Snowflake
def fetch_data(ctx, query):
    return pd.read_sql(query, ctx)

# Function to plot the comparison results using a bar chart
def plot_comparison_results(
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

    if not matched_but_different.empty and matched_but_different.iloc[0, 0] == "All rows match":
        matched_but_different_count = 0
    else:
        matched_but_different_count = len(matched_but_different)

    # Prepare data for the plot with adjusted counts
    counts = {
        "Differences Between Tables": differences_count,
        "Detailed Differences for Matched Rows": matched_but_different_count,
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
def compare_dataframes_by_key(df1, df2, key_column):
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


# Function to compare schemas of two tables in Snowflake and fetch schema details
def compare_schemas(ctx, full_table_name1, full_table_name2):
    catalog1, schema1, table1 = full_table_name1.split(".")
    catalog2, schema2, table2 = full_table_name2.split(".")

    df_schema1 = fetch_schema(ctx, catalog1, schema1, table1)
    df_schema2 = fetch_schema(ctx, catalog2, schema2, table2)

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
        lambda row: "Match" if row["Data Type Table 1"] == row["Data Type Table 2"] else "Mismatch",
        axis=1,
    )

    # Set "Data Type Match" to "N/A" for columns not present in both tables
    comparison_results.loc[
        comparison_results["Column Presence"] != "In Both Tables", "Data Type Match"
    ] = "N/A"
    
    return comparison_results



# Function to fetch schema details from Snowflake
def fetch_schema(ctx, catalog, schema, table_name, filter_conditions=""):
    where_clause = f" AND {filter_conditions}" if filter_conditions else ""
    schema_query = f"""
    SELECT COLUMN_NAME, DATA_TYPE 
    FROM snowflake.account_usage.columns 
    WHERE TABLE_CATALOG = '{catalog}' AND TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table_name}' AND DELETED IS NULL{where_clause};
    """
    # Formatting schema fetch query for readability
    formatted_schema_query = sqlparse.format(schema_query, reindent=True, keyword_case="lower")

    # Check if the formatted schema query already exists in generated_schema_queries to avoid duplicates
    if formatted_schema_query not in generated_schema_queries:
        generated_schema_queries.append(formatted_schema_query)  # Add the formatted schema query if it's unique

    # Returning the dataframe result of the schema query
    return pd.read_sql(schema_query, ctx)

# Function to perform aggregate analysis between two tables in Snowflake
def perform_aggregate_analysis(
    ctx, full_table_name1, full_table_name2, filter_conditions=""
):
    catalog1, schema1, table1 = full_table_name1.split(".")
    catalog2, schema2, table2 = full_table_name2.split(".")

    df_schema1 = fetch_schema(ctx, catalog1, schema1, table1)
    df_schema2 = fetch_schema(ctx, catalog2, schema2, table2)
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
        aggregate_expression(row["COLUMN_NAME"], row["DATA_TYPE"])
        for _, row in matching_columns.iterrows()
        if aggregate_expression(row["COLUMN_NAME"], row["DATA_TYPE"])
    ]

    results1, results2 = None, None
    
    # Execute aggregate queries in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                execute_aggregate_query,
                ctx,
                catalog1,
                schema1,
                table1,
                aggregates,
                filter_conditions,
            ),
            executor.submit(
                execute_aggregate_query,
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

    # Compare the results of aggregate analysis between both tables
    comparison_results_list = []
    for column in results1.columns:
        val1, val2 = results1[column].iloc[0], results2[column].iloc[0]
        str_val1, str_val2 = str(val1), str(val2)
        result = "Pass" if str_val1 == str_val2 else "Fail"

        # Append the comparison results to a list
        comparison_results_list.append(
            {
                "Column Name": column,
                "Result": result,
                "Table 1 Value": str_val1,
                "Table 2 Value": str_val2,
                "Description": get_check_description(column),
            }
        )

    # Convert the list of comparison results to a DataFrame
    comparison_results = pd.DataFrame(comparison_results_list)
    return comparison_results[
        ["Column Name", "Result", "Table 1 Value", "Table 2 Value", "Description"]
    ]


# Function to generate descriptions based on aggregate column names (case-insensitive)
def get_check_description(column):
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
def aggregate_expression(column_name, data_type):
    numeric_types = ["NUMBER", "FLOAT", "DECIMAL", "INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "DOUBLE", "REAL"]
    text_types = ["VARCHAR", "TEXT", "STRING", "CHAR", "CHARACTER", "NCHAR", "NVARCHAR"]
    timestamp_types = ["DATE", "DATETIME", "TIMESTAMP", "TIMESTAMP_LTZ", "TIMESTAMP_NTZ", "TIMESTAMP_TZ"]

    # Handling numeric types with SUM, COUNT, and APPROX_COUNT_DISTINCT
    if data_type.upper() in numeric_types:
        return f"SUM({column_name}::NUMBER) AS total_{column_name.lower()}, COUNT({column_name}) AS count_{column_name.lower()}, APPROX_COUNT_DISTINCT({column_name}) AS approx_distinct_{column_name.lower()}"
    # Handling string types with COUNT and APPROX_COUNT_DISTINCT for unique values
    elif data_type.upper() in text_types:
        return f"COUNT({column_name}) AS count_{column_name.lower()}, APPROX_COUNT_DISTINCT({column_name}) AS unique_{column_name.lower()}"
    # Handling date and timestamp types with MIN, MAX, and COUNT
    elif data_type.upper() in timestamp_types:
        # Ensure timestamps are converted to a standard format for proper comparison
        return f"TO_CHAR(MIN({column_name}), 'YYYY-MM-DD HH24:MI:SS') AS min_{column_name.lower()}, TO_CHAR(MAX({column_name}), 'YYYY-MM-DD HH24:MI:SS') AS max_{column_name.lower()}, COUNT({column_name}) AS count_{column_name.lower()}"
    else:
        # For data types that do not match any of the above categories, return None
        # This will exclude them from aggregate queries
        return None


# Function to execute an aggregate query on a table in Snowflake
def execute_aggregate_query(ctx, catalog, schema, table, aggregates, filter_conditions=""):
    where_clause = f"WHERE {filter_conditions}" if filter_conditions else ""
    aggregates_sql = ", ".join(aggregates)
    query = f"""WITH table_agg AS (
                  SELECT COUNT(*) AS row_count, {aggregates_sql}
                  FROM "{catalog}"."{schema}"."{table}"
                  {where_clause} 
               )
               SELECT * FROM table_agg;"""
    # Use sqlparse to format the query
    formatted_query = sqlparse.format(query, reindent=True, keyword_case="lower")
    generated_aggregate_queries.append(formatted_query)  # Store the formatted query
    return pd.read_sql(query, ctx)

# Function to display generated aggregate queries in a collapsible code block
def display_all_generated_queries():
    with st.expander("Generated Queries 👨‍💻"):
        # Schema Queries Section
        if generated_schema_queries:
            st.markdown("### Schema Queries")
            for query in generated_schema_queries:
                st.code(query, language="sql")
        # Column Queries Section
        if generated_column_queries:
            st.markdown("### Column Queries")
            for query in generated_column_queries:
                st.code(query, language="sql")  
        # Aggregate Queries Section
        if generated_aggregate_queries:
            st.markdown("### Aggregate Queries")
            for query in generated_aggregate_queries:
                st.code(query, language="sql")


# Function to plot schema comparison results using a bar chart
def plot_schema_comparison_results(schema_comparison_results):
    counts = schema_comparison_results["Column Presence"].value_counts().reset_index()
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
    )  # Hide the legend since all bars are the same color
    st.plotly_chart(fig, use_container_width=True)


# Function to plot a summary of the schema comparison focusing on data type matches/mismatches
def plot_schema_comparison_summary(schema_comparison_results):
    data_type_counts = (
        schema_comparison_results["Data Type Match"]
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
    st.plotly_chart(fig_data_types, use_container_width=True)


# Function to plot the results of aggregate analysis, indicating pass/fail status for each check
def plot_aggregate_analysis_summary(aggregate_results):
    results_count = (
        aggregate_results["Result"]
        .value_counts()
        .reindex(["Pass", "Fail"], fill_value=0)
        .reset_index()
    )
    results_count.columns = ["Result", "Count"]
    fig = px.bar(
        results_count,
        x="Result",
        y="Count",
        labels={"Result": "Test Result"},
        color_discrete_sequence=["#2980b9"],
    )  
    st.plotly_chart(fig, use_container_width=True)

# Main function to run the Snowflake Table Comparison Tool
def main():
    st.title("❄️ Snowflake Table Comparison Tool")

    # Configuration sidebar setup
    st.sidebar.header("Configuration ⚙️")
    user = st.sidebar.text_input("User 🧑‍💼").upper()
    account = st.sidebar.text_input("Account 🏦").upper()
    warehouse = st.sidebar.text_input("Warehouse 🏭").upper()
    use_external_browser_auth = st.sidebar.checkbox("Use External Browser Authentication")
    password = ""
    authenticator = "externalbrowser" if use_external_browser_auth else "snowflake"
    if not use_external_browser_auth:
        password = st.sidebar.text_input("Password 🔒", type="password")
    row_count = st.sidebar.slider(
        "Number of Rows from Top/Bottom",
        min_value=10,
        max_value=1000,
        value=50,
        step=10,
    )
    key_column = st.sidebar.text_input("Unique Key Column 🗝️", placeholder="UNIQUE_KEY").upper()
    full_table_name1 = st.sidebar.text_input(
        "Table 1 📝", placeholder="DATABASE.SCHEMA.TABLE"
    ).upper()
    full_table_name2 = st.sidebar.text_input(
        "Table 2 ✏️", placeholder="DATABASE.SCHEMA.TABLE"
    ).upper()
    filter_conditions = st.sidebar.text_area(
        "Filter conditions (optional) ✨",
        placeholder="EMAIL = 'mitch@example.com' AND DATE::DATE >= '2024-01-01'::DATE",
    )

    # Initialize session state for progress and status updates
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    if 'status_message' not in st.session_state:
        st.session_state.status_message = "Ready"

    # Helper function to update progress and status message
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
        if not user or not account or not warehouse or not full_table_name1 or not full_table_name2:
            st.error("Please fill in all required fields.")
            return

        if not full_table_name1.count('.') == 2 or not full_table_name2.count('.') == 2:
            st.error("Please ensure both tables are in the format: DATABASE.SCHEMA.TABLE")
            return

        try:
            st.snow()
            update_progress(10, "Connecting to Snowflake...")
            ctx = snowflake.connector.connect(
                user=user,
                account=account,
                password=password,
                authenticator=authenticator,
                warehouse=warehouse,
            )
            update_progress(20, "Connected to Snowflake ✅")
            time.sleep(3)

            base_filter = f"WHERE {filter_conditions}" if filter_conditions else ""
            queries = [
                f"SELECT * FROM {full_table_name1} {base_filter} ORDER BY {key_column} ASC LIMIT {row_count}",
                f"SELECT * FROM {full_table_name1} {base_filter} ORDER BY {key_column} DESC LIMIT {row_count}",
                f"SELECT * FROM {full_table_name2} {base_filter} ORDER BY {key_column} ASC LIMIT {row_count}",
                f"SELECT * FROM {full_table_name2} {base_filter} ORDER BY {key_column} DESC LIMIT {row_count}"
            ]

            # Format each query using sqlparse and add to the generated_column_queries list
            for query in queries:
                formatted_query = sqlparse.format(query, reindent=True, keyword_case="lower")
                generated_column_queries.append(formatted_query)
            
            # Fetch data from both tables
            update_progress(30, "Running Snowflake Queries 🏃‍♂️💨")
            time.sleep(1)
            dfs = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_query = {executor.submit(fetch_data, ctx, query): query for query in queries}
                for future in concurrent.futures.as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        dfs[query] = future.result()
                    except Exception as exc:
                        st.error(f"Query {query} generated an exception: {exc}")
                        return

            df1 = pd.concat([dfs[queries[0]], dfs[queries[1]]])
            df2 = pd.concat([dfs[queries[2]], dfs[queries[3]]])

            update_progress(40, "Working on Row Level Analysis...")
            differences, matched_but_different = compare_dataframes_by_key(df1, df2, key_column)
            st.header("Row Level Analysis 🔎")
            plot_comparison_results(differences, matched_but_different, len(df1), len(df2))
            if not differences.empty:
                st.dataframe(differences)
            if not matched_but_different.empty:
                st.dataframe(matched_but_different)
                

            with st.expander("Queries 🐸"):
                for query in generated_column_queries:
                    st.code(query, language="sql")


            update_progress(50, "Row Level Analysis completed ✅")
            time.sleep(1)
            
            update_progress(60, "Working on Column Analysis...")
            time.sleep(1)
            schema_comparison_results = compare_schemas(ctx, full_table_name1, full_table_name2)
            st.header("Column Analysis 🔎")
            st.dataframe(schema_comparison_results)
            plot_schema_comparison_results(schema_comparison_results)
            plot_schema_comparison_summary(schema_comparison_results)
            # Right after plotting schema comparison results:
            if generated_schema_queries:
                with st.expander("Queries 🤖"):
                    for query in generated_schema_queries:
                        st.code(query, language="sql")
            update_progress(70, "Column Analysis completed ✅")
            time.sleep(1)

            update_progress(80, "Working on Aggregate Analysis...")
            aggregate_results = perform_aggregate_analysis(ctx, full_table_name1, full_table_name2, filter_conditions)
            st.header("Aggregate Analysis 🔎")
            st.dataframe(aggregate_results)
            plot_aggregate_analysis_summary(aggregate_results)
            
            # After plotting the aggregate analysis summary:
            if generated_aggregate_queries:
                with st.expander("Queries 🥷"):
                    for query in generated_aggregate_queries:
                        st.code(query, language="sql")

            update_progress(99, "Aggregate analysis completed ✅")
            time.sleep(1)

            update_progress(100, "Analysis completed ✅")

        except Exception as e:
            update_progress(0, f"Failed to run analysis: {e}")

if __name__ == "__main__":
    main()
