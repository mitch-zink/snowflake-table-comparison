"""
Description: This application compares two tables in Snowflake
Author: Mitch Zink
Last Updated: 4/3/2024
"""

import pandas as pd  # For data manipulation
import snowflake.connector  # For connecting to Snowflake
import streamlit as st  # UI
import plotly.express as px  # For interactive visualizations
import concurrent.futures  # For parallel execution of aggregate queries
import sqlparse  # For formatting SQL queries

# Global variables to store generated queries
generated_queries = []
generated_schema_queries = []  # For storing schema fetch queries

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
    # Prepare data for the plot with additional row counts
    counts = {
        "Differences Between Tables": len(differences),
        "Detailed Differences for Matched Rows": len(matched_but_different),
        "Rows Fetched from First Table": rows_fetched_from_first,
        "Rows Fetched from Second Table": rows_fetched_from_second,
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
        title_text="Comparison Results",
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
                    "column": "N/A",
                    "original_value": "N/A",
                    "modified_value": "N/A",
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
        suffixes=("_original", "_modified"),
    )
    differences = merged_df[merged_df["_merge"] != "both"]

    compare_cols = [
        col for col in df1.columns if col != key_column_lower and col in df2.columns
    ]
    diff_data = {"key": [], "column": [], "original_value": [], "modified_value": []}
    for _, row in merged_df.iterrows():
        for col in compare_cols:
            original_val, modified_val = row[f"{col}_original"], row[f"{col}_modified"]
            if pd.isna(original_val) and pd.isna(modified_val):
                continue  # Skip comparison if both values are NaN
            if original_val != modified_val:
                diff_data["key"].append(row[key_column_lower])
                diff_data["column"].append(col)
                diff_data["original_value"].append(original_val)
                diff_data["modified_value"].append(modified_val)

    matched_but_different = pd.DataFrame(diff_data)
    return differences, matched_but_different


# Compare schemas of two tables in Snowflake
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
            "left_only": "Only in First Table",
            "right_only": "Only in Second Table",
            "both": "In Both Tables",
        }
    )
    comparison_results["Data Type Match"] = comparison_results.apply(
        lambda row: "Match" if row["DATA_TYPE_x"] == row["DATA_TYPE_y"] else "Mismatch",
        axis=1,
    )
    comparison_results.loc[
        comparison_results["Column Presence"] != "In Both Tables", "Data Type Match"
    ] = "N/A"
    return comparison_results


# Function to fetch schema details from Snowflake
def fetch_schema(ctx, catalog, schema, table_name):
    query = f"""
    SELECT DISTINCT COLUMN_NAME, DATA_TYPE 
    FROM snowflake.account_usage.columns 
    WHERE TABLE_CATALOG = '{catalog}' AND TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table_name}' AND DELETED IS NULL;
    """
    # Store the formatted query for later display
    formatted_query = sqlparse.format(query, reindent=True, keyword_case="upper")
    generated_schema_queries.append(formatted_query)  # Store the schema fetch query
    return pd.read_sql(query, ctx)

# Function to display generated queries in a collapsible code block
def display_generated_schema_queries():
    with st.expander("Column Queries 👨‍💻"):
        for query in generated_schema_queries:
            st.code(query, language="sql")

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
    # Handling numeric types with SUM, COUNT, and APPROX_COUNT_DISTINCT
    if data_type.upper() in ["NUMBER", "FLOAT", "DECIMAL"]:
        return f"SUM({column_name}::NUMBER) AS total_{column_name.lower()}, COUNT({column_name}) AS count_{column_name.lower()}, APPROX_COUNT_DISTINCT({column_name}) AS approx_distinct_{column_name.lower()}"
    # Handling string types with COUNT and APPROX_COUNT_DISTINCT for unique values
    elif data_type.upper() in ["VARCHAR", "TEXT"]:
        return f"COUNT({column_name}) AS count_{column_name.lower()}, APPROX_COUNT_DISTINCT({column_name}) AS unique_{column_name.lower()}"
    # Handling date and timestamp types with MIN, MAX, and COUNT
    elif data_type.upper() in ["DATE", "TIMESTAMP"]:
        return f"MIN({column_name}) AS min_{column_name.lower()}, MAX({column_name}) AS max_{column_name.lower()}, COUNT({column_name}) AS count_{column_name.lower()}"
    return None


# Function to execute an aggregate query on a table in Snowflake
def execute_aggregate_query(
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
    # Use sqlparse to format the query
    formatted_query = sqlparse.format(query, reindent=True, keyword_case="upper")
    generated_queries.append(formatted_query)  # Store the formatted query
    return pd.read_sql(query, ctx)


# Function to display generated aggregate queries in a collapsible code block
def display_generated_queries():
    with st.expander("Aggregate Queries 👨‍💻"):
        for query in generated_queries:
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
    )  # A shade of blue
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
        color_discrete_sequence=["#3498db"],
    )  # Another shade of blue
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
        color_discrete_sequence=["#5DADE2"],
    )  # Lighter shade of blue
    st.plotly_chart(fig, use_container_width=True)

# Main function to run the Snowflake Table Comparison Tool
def main():
    st.title("❄️ Snowflake Table Comparison Tool")
    status_message = st.empty()  # Centralized status message area for all updates
    progress_bar = st.progress(0)

    # Configuration sidebar setup
    st.sidebar.header("Configuration ⚙️")
    user = st.sidebar.text_input("User 🧑‍💼")
    account = st.sidebar.text_input("Account 🏦")
    warehouse = st.sidebar.text_input("Warehouse 🏭")
    authenticator = (
        "externalbrowser"
        if st.sidebar.checkbox("Use External Browser Authentication")
        else "snowflake"
    )
    password = (
        ""
        if authenticator == "externalbrowser"
        else st.sidebar.text_input("Password 🔒", type="password")
    )

    comparison_type = st.sidebar.radio(
        "Choose Comparison Type 🔄",
        ["Value Level Analysis", "Column and Aggregate Analysis"],
    )

    # Value Level Analysis Inputs
    if comparison_type == "Value Level Analysis":
        st.sidebar.subheader("Value Level Analysis Inputs")
        row_count = st.sidebar.slider(
            "Number of Rows from Top/Bottom",
            min_value=10,
            max_value=1000,
            value=50,
            step=10,
        )
        key_column = st.sidebar.text_input("Unique Key Column 🗝️")
        full_table_name1 = st.sidebar.text_input(
            "First Table 📝", "DATABASE.SCHEMA.TABLE"
        )
        full_table_name2 = st.sidebar.text_input(
            "Second Table ✏️", "DATABASE.SCHEMA.TABLE"
        )

    # Column and Aggregate Analysis Inputs
    elif comparison_type == "Column and Aggregate Analysis":
        st.sidebar.subheader("Column and Aggregate Analysis Inputs")
        full_table_name1 = st.sidebar.text_input(
            "First Table 📝", "DATABASE.SCHEMA.TABLE"
        )
        full_table_name2 = st.sidebar.text_input(
            "Second Table ✏️", "DATABASE.SCHEMA.TABLE"
        )
        filter_conditions = st.sidebar.text_area(
            "Filter conditions (optional) ✨",
            placeholder="email = 'mitch@example.com' AND date::date >= '2024-01-01'::date",
        )

    # Run Comparison Button
    if st.sidebar.button("Run Comparison 🚀"):
        st.snow()
        status_message.info("Preparing to connect to Snowflake...")
        progress_bar.progress(10)

        # Connect to Snowflake and run the comparison
        try:
            status_message.text("Connecting to Snowflake...")
            ctx = snowflake.connector.connect(
                user=user,
                account=account,
                password=password,
                authenticator=authenticator,
                warehouse=warehouse,
            )
            status_message.success("Connected to Snowflake ✅")
            progress_bar.progress(20)

            # Run the selected comparison type
            if comparison_type == "Value Level Analysis":
                status_message.text("Fetching data for Value Level Analysis...")
                query_top1 = f"SELECT * FROM {full_table_name1} ORDER BY {key_column} ASC LIMIT {row_count}"
                query_bottom1 = f"SELECT * FROM {full_table_name1} ORDER BY {key_column} DESC LIMIT {row_count}"
                query_top2 = f"SELECT * FROM {full_table_name2} ORDER BY {key_column} ASC LIMIT {row_count}"
                query_bottom2 = f"SELECT * FROM {full_table_name2} ORDER BY {key_column} DESC LIMIT {row_count}"

                # Fetch data from both tables in parallel using ThreadPoolExecutor
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_df = {
                        executor.submit(fetch_data, ctx, query_top1): "top_first",
                        executor.submit(fetch_data, ctx, query_bottom1): "bottom_first",
                        executor.submit(fetch_data, ctx, query_top2): "top_second",
                        executor.submit(
                            fetch_data, ctx, query_bottom2
                        ): "bottom_second",
                    }
                    dfs = {}
                    for future in concurrent.futures.as_completed(future_to_df):
                        part_name = future_to_df[future]
                        try:
                            dfs[part_name] = future.result()
                        except Exception as exc:
                            status_message.error(
                                f"{part_name} table fetch generated an exception: {exc}"
                            )

                df1 = pd.concat([dfs["top_first"], dfs["bottom_first"]])
                df2 = pd.concat([dfs["top_second"], dfs["bottom_second"]])
                progress_bar.progress(40)

                differences, matched_but_different = compare_dataframes_by_key(
                    df1, df2, key_column
                )

                total_rows_fetched_first = len(dfs["top_first"]) + len(
                    dfs["bottom_first"]
                )
                total_rows_fetched_second = len(dfs["top_second"]) + len(
                    dfs["bottom_second"]
                )

                plot_comparison_results(
                    differences,
                    matched_but_different,
                    total_rows_fetched_first,
                    total_rows_fetched_second,
                )

                st.subheader("Differences between Tables 📊")
                st.dataframe(differences)
                st.subheader("Detailed Differences for Matched Rows 📑")
                st.dataframe(matched_but_different)
                progress_bar.progress(80)

            elif comparison_type == "Column and Aggregate Analysis":
                status_message.text("Fetching schema comparison data...")
                schema_comparison_results = compare_schemas(
                    ctx, full_table_name1, full_table_name2
                )
                progress_bar.progress(40)

                st.header("Column Analysis 🔍")
                st.dataframe(
                    schema_comparison_results[
                        [
                            "COLUMN_NAME",
                            "DATA_TYPE_x",
                            "DATA_TYPE_y",
                            "Column Presence",
                            "Data Type Match",
                        ]
                    ].rename(
                        columns={
                            "DATA_TYPE_x": "Data Type (First Table)",
                            "DATA_TYPE_y": "Data Type (Second Table)",
                        }
                    )
                )
                plot_schema_comparison_summary(schema_comparison_results)
                plot_schema_comparison_results(schema_comparison_results)
                display_generated_schema_queries()

                status_message.text("Performing aggregate analysis...")
                aggregate_results = perform_aggregate_analysis(
                    ctx, full_table_name1, full_table_name2, filter_conditions
                )
                progress_bar.progress(60)

                st.header("Aggregate Analysis 🔍")
                st.dataframe(aggregate_results)
                plot_aggregate_analysis_summary(aggregate_results)
                progress_bar.progress(80)
                display_generated_queries()

            ctx.close()
            progress_bar.progress(100)
            status_message.text("Disconnected from Snowflake")

        except Exception as e:
            progress_bar.progress(0)
            status_message.error(f"Failed to run comparison: {e}")


if __name__ == "__main__":
    main()