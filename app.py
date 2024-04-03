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
    WHERE TABLE_CATALOG = '{catalog}' AND TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table_name}';
    """
    return pd.read_sql(query, ctx)


# Perform aggregate analysis between two tables in Snowflake
def perform_aggregate_analysis(ctx, full_table_name1, full_table_name2):
    catalog1, schema1, table1 = full_table_name1.split(".")
    catalog2, schema2, table2 = full_table_name2.split(".")

    df_schema1 = fetch_schema(ctx, catalog1, schema1, table1)
    df_schema2 = fetch_schema(ctx, catalog2, schema2, table2)
    matching_columns = pd.merge(
        df_schema1, df_schema2, on=["COLUMN_NAME", "DATA_TYPE"], how="inner"
    )

    aggregates = [
        aggregate_expression(row["COLUMN_NAME"], row["DATA_TYPE"])
        for _, row in matching_columns.iterrows()
        if aggregate_expression(row["COLUMN_NAME"], row["DATA_TYPE"])
    ]
    if aggregates:
        # Use ThreadPoolExecutor to run aggregate queries in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_table = {
                executor.submit(
                    execute_aggregate_query, ctx, catalog1, schema1, table1, aggregates
                ): "first",
                executor.submit(
                    execute_aggregate_query, ctx, catalog2, schema2, table2, aggregates
                ): "second",
            }
            for future in concurrent.futures.as_completed(future_to_table):
                table_name = future_to_table[future]
                try:
                    results = future.result()  # Get the result of each future
                    if table_name == "first":
                        results1 = results
                    else:
                        results2 = results
                except Exception as exc:
                    print(f"{table_name} generated an exception: {exc}")

        # Compare results and generate descriptions
        checks = []
        for column in results1.columns:
            result = (
                "Pass"
                if results1[column].iloc[0] == results2[column].iloc[0]
                else "Fail"
            )
            description = get_check_description(
                column
            )  # Get the description for each check
            checks.append(
                {"Check": column, "Result": result, "Description": description}
            )
        return pd.DataFrame(checks)
    else:
        return pd.DataFrame(
            {
                "Check": ["No matching columns for aggregation"],
                "Result": ["N/A"],
                "Description": ["N/A"],
            }
        )


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
# This helps in creating SQL query parts for different types of aggregation analysis.
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


# Function to execute an aggregate query in Snowflake and return the results as a DataFrame.
# This function constructs and executes a SQL query with aggregated metrics.
def execute_aggregate_query(ctx, catalog, schema, table, aggregates):
    aggregates_sql = ", ".join(aggregates)
    query = f'WITH table_agg AS (SELECT COUNT(*) AS row_count, {aggregates_sql} FROM "{catalog}"."{schema}"."{table}") SELECT * FROM table_agg;'
    return pd.read_sql(query, ctx)


# Function to plot schema comparison results using a bar chart.
# This visualization helps in understanding the presence of columns in compared tables.
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


# Function to plot a summary of the schema comparison focusing on data type matches/mismatches.
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


# Function to plot the results of aggregate analysis, indicating pass/fail status for each check.
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
        ["Row Level Analysis", "Column and Aggregate Analysis"],
    )
    if comparison_type == "Row Level Analysis":
        st.sidebar.subheader("Row Level Analysis Inputs")
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
    elif comparison_type == "Column and Aggregate Analysis":
        st.sidebar.subheader("Schema and Aggregate Analysis Inputs")
        full_table_name1 = st.sidebar.text_input(
            "First Table 📝", "DATABASE.SCHEMA.TABLE"
        )
        full_table_name2 = st.sidebar.text_input(
            "Second Table ✏️", "DATABASE.SCHEMA.TABLE"
        )

    if st.sidebar.button("Run Comparison 🚀"):
        st.snow()
        status_message.info("Preparing to connect to Snowflake...")
        progress_bar.progress(10)

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

            if comparison_type == "Row Level Analysis":
                status_message.text("Fetching data for row level analysis...")
                query_top1 = f"SELECT * FROM {full_table_name1} ORDER BY {key_column} ASC LIMIT {row_count}"
                query_bottom1 = f"SELECT * FROM {full_table_name1} ORDER BY {key_column} DESC LIMIT {row_count}"
                query_top2 = f"SELECT * FROM {full_table_name2} ORDER BY {key_column} ASC LIMIT {row_count}"
                query_bottom2 = f"SELECT * FROM {full_table_name2} ORDER BY {key_column} DESC LIMIT {row_count}"

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

                st.subheader("Schema Analysis 🔍")
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

                status_message.text("Performing aggregate analysis...")
                aggregate_results = perform_aggregate_analysis(
                    ctx, full_table_name1, full_table_name2
                )
                progress_bar.progress(60)

                st.header("Aggregate Analysis 🔍")
                st.dataframe(aggregate_results)
                plot_aggregate_analysis_summary(aggregate_results)
                progress_bar.progress(80)

            ctx.close()
            progress_bar.progress(100)
            status_message.text("Disconnected from Snowflake")

        except Exception as e:
            progress_bar.progress(0)
            status_message.error(f"Failed to run comparison: {e}")


if __name__ == "__main__":
    main()
