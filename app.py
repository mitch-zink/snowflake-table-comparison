"""
Description: This application compares two tables in Snowflake
Author: Mitch Zink
Last Updated: 4/3/2024
"""

# Import necessary libraries
import pandas as pd
import snowflake.connector
import streamlit as st
import plotly.express as px
import concurrent.futures

# Function to compare two dataframes by a specific key column
def compare_dataframes_by_key(df1, df2, key_column):
    # Ensure column names in both dataframes are lowercase for consistency
    df1.columns = df1.columns.str.lower()
    df2.columns = df2.columns.str.lower()
    key_column_lower = key_column.lower()
    
    # Merge dataframes to find differences
    merged_df = df1.merge(df2, on=key_column_lower, how="outer", indicator=True, suffixes=("_original", "_modified"))
    differences = merged_df[merged_df["_merge"] != "both"]
    
    # Identify specific differences between matched rows
    matched_but_different = pd.DataFrame()
    if not differences.empty:
        compare_cols = [col for col in df1.columns if col != key_column_lower and "_merge" not in col]
        diff_data = {"key": [], "column": [], "original_value": [], "modified_value": []}
        for _, row in differences.iterrows():
            for col in compare_cols:
                original_val, modified_val = row[f"{col}_original"], row[f"{col}_modified"]
                if original_val != modified_val:
                    diff_data["key"].append(row[key_column_lower])
                    diff_data["column"].append(col)
                    diff_data["original_value"].append(original_val)
                    diff_data["modified_value"].append(modified_val)
        matched_but_different = pd.DataFrame(diff_data)
    return differences, matched_but_different

# Compare schemas of two tables in Snowflake
def compare_schemas(ctx, full_table_name1, full_table_name2):
    # Extract catalog, schema, and table names
    catalog1, schema1, table1 = full_table_name1.split('.')
    catalog2, schema2, table2 = full_table_name2.split('.')
    
    # Fetch schemas from Snowflake
    df_schema1 = fetch_schema(ctx, catalog1, schema1, table1)
    df_schema2 = fetch_schema(ctx, catalog2, schema2, table2)
    
    # Compare column presence and data type match between the two schemas
    comparison_results = pd.merge(df_schema1, df_schema2, on="COLUMN_NAME", how="outer", indicator="Column Presence")
    comparison_results["Column Presence"] = comparison_results["Column Presence"].map({"left_only": "Only in First Table", "right_only": "Only in Second Table", "both": "In Both Tables"})
    comparison_results["Data Type Match"] = comparison_results.apply(lambda row: "Match" if row["DATA_TYPE_x"] == row["DATA_TYPE_y"] else "Mismatch", axis=1)
    comparison_results.loc[comparison_results["Column Presence"] != "In Both Tables", "Data Type Match"] = "N/A"
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
    catalog1, schema1, table1 = full_table_name1.split('.')
    catalog2, schema2, table2 = full_table_name2.split('.')
    
    df_schema1 = fetch_schema(ctx, catalog1, schema1, table1)
    df_schema2 = fetch_schema(ctx, catalog2, schema2, table2)
    matching_columns = pd.merge(df_schema1, df_schema2, on=["COLUMN_NAME", "DATA_TYPE"], how="inner")
    
    aggregates = [aggregate_expression(row["COLUMN_NAME"], row["DATA_TYPE"]) for _, row in matching_columns.iterrows() if aggregate_expression(row["COLUMN_NAME"], row["DATA_TYPE"])]
    if aggregates:
        # Use ThreadPoolExecutor to run aggregate queries in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_table = {
                executor.submit(execute_aggregate_query, ctx, catalog1, schema1, table1, aggregates): 'first',
                executor.submit(execute_aggregate_query, ctx, catalog2, schema2, table2, aggregates): 'second'
            }
            for future in concurrent.futures.as_completed(future_to_table):
                table_name = future_to_table[future]
                try:
                    results = future.result()  # Get the result of each future
                    if table_name == 'first':
                        results1 = results
                    else:
                        results2 = results
                except Exception as exc:
                    print(f'{table_name} generated an exception: {exc}')
        
        # Compare results and generate descriptions
        checks = []
        for column in results1.columns:
            result = "Pass" if results1[column].iloc[0] == results2[column].iloc[0] else "Fail"
            description = get_check_description(column)  # Get the description for each check
            checks.append({"Check": column, "Result": result, "Description": description})
        return pd.DataFrame(checks)
    else:
        return pd.DataFrame({"Check": ["No matching columns for aggregation"], "Result": ["N/A"], "Description": ["N/A"]})


# Function to generate descriptions based on aggregate column names (case-insensitive)
def get_check_description(column):
    column = column.lower()  # Convert column name to lower case for case-insensitive comparison
    if "total_" in column:
        return "Compares the sum of numeric values in this column between both tables."
    elif "count_" in column:
        return "Compares the count of non-null values in this column between both tables."
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
    query = f"WITH table_agg AS (SELECT COUNT(*) AS row_count, {aggregates_sql} FROM \"{catalog}\".\"{schema}\".\"{table}\") SELECT * FROM table_agg;"
    return pd.read_sql(query, ctx)

# Function to plot schema comparison results using a bar chart.
# This visualization helps in understanding the presence of columns in compared tables.
def plot_schema_comparison_results(schema_comparison_results):
    counts = schema_comparison_results['Column Presence'].value_counts().reset_index()
    counts.columns = ['Category', 'Count']
    fig = px.bar(counts, x='Category', y='Count',
                 text='Count', color_discrete_sequence=['#2980b9'])  # A shade of blue
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(xaxis_title="Column Presence",
                      yaxis_title="Count",
                      uniformtext_minsize=8,
                      uniformtext_mode='hide',
                      showlegend=False)  # Hide the legend since all bars are the same color
    st.plotly_chart(fig, use_container_width=True)

# Function to plot a summary of the schema comparison focusing on data type matches/mismatches.
def plot_schema_comparison_summary(schema_comparison_results):
    data_type_counts = schema_comparison_results["Data Type Match"].value_counts().reindex(["Match", "Mismatch"], fill_value=0).reset_index()
    data_type_counts.columns = ['Match Status', 'Count']
    fig_data_types = px.bar(data_type_counts, x='Match Status', y='Count', text='Count', color_discrete_sequence=['#3498db'])  # Another shade of blue
    fig_data_types.update_traces(texttemplate='%{text}', textposition='outside')
    fig_data_types.update_layout(xaxis_title="Data Type Match", yaxis_title="Count", uniformtext_minsize=8, uniformtext_mode='hide', legend_title="Match Status")
    st.plotly_chart(fig_data_types, use_container_width=True)

# Function to plot the results of aggregate analysis, indicating pass/fail status for each check.
def plot_aggregate_analysis_summary(aggregate_results):
    results_count = aggregate_results['Result'].value_counts().reindex(["Pass", "Fail"], fill_value=0).reset_index()
    results_count.columns = ['Result', 'Count']
    fig = px.bar(results_count, x='Result', y='Count', labels={'Result': 'Verification Result'}, color_discrete_sequence=['#5DADE2'])  # Lighter shade of blue
    st.plotly_chart(fig, use_container_width=True)
def main():
    st.title("❄️ Snowflake Table Comparison Tool")
    status_message = st.empty()
    progress_bar = st.progress(0)

    # Configuration sidebar
    st.sidebar.header("Configuration ⚙️")
    user = st.sidebar.text_input("User 🧑‍💼")
    account = st.sidebar.text_input("Account 🏦")
    warehouse = st.sidebar.text_input("Warehouse 🏭")
    authenticator = "externalbrowser" if st.sidebar.checkbox("Use External Browser Authentication") else "snowflake"
    password = "" if authenticator == "externalbrowser" else st.sidebar.text_input("Password 🔒", type="password")
    
    comparison_type = st.sidebar.radio("Choose Comparison Type 🔄", ["Row-wise Comparison", "Schema and Table Comparison"])
    full_table_name1, full_table_name2 = "", ""
    if comparison_type == "Schema and Table Comparison":
        full_table_name1 = st.sidebar.text_input("First Table (DATABASE.SCHEMA.TABLE) 📝", "DATABASE.SCHEMA.TABLE")
        full_table_name2 = st.sidebar.text_input("Second Table (DATABASE.SCHEMA.TABLE) ✏️", "DATABASE.SCHEMA.TABLE")

    if st.sidebar.button("Run Comparison 🚀"):
        st.snow()
        status_message.info('Preparing to connect to Snowflake...')
        progress_bar.progress(10)
        
        try:
            # Connecting to Snowflake
            status_message.info('Connecting to Snowflake...')
            ctx = snowflake.connector.connect(user=user, account=account, password=password, authenticator=authenticator, warehouse=warehouse)
            status_message.success("Connected to Snowflake ✅")
            progress_bar.progress(20)

            if comparison_type == "Schema and Table Comparison":
                # Schema Comparison
                status_message.info("Fetching schema comparison data...")
                schema_comparison_results = compare_schemas(ctx, full_table_name1, full_table_name2)
                progress_bar.progress(40)
                
                # Displaying original schema comparison DataFrame
                st.subheader("Schema Analysis 🔍")
                st.dataframe(schema_comparison_results[['COLUMN_NAME', 'DATA_TYPE_x', 'DATA_TYPE_y', 'Column Presence', 'Data Type Match']].rename(columns={'DATA_TYPE_x': 'Data Type (First Table)', 'DATA_TYPE_y': 'Data Type (Second Table)'}))
                plot_schema_comparison_summary(schema_comparison_results)
                
                # New call to plot the column presence with a bar chart
                plot_schema_comparison_results(schema_comparison_results)
                
                # Aggregate Analysis
                status_message.info("Performing aggregate analysis...")
                aggregate_results = perform_aggregate_analysis(ctx, full_table_name1, full_table_name2)
                progress_bar.progress(60)
                
                # Displaying original aggregate analysis DataFrame
                st.header("Aggregate Analysis 🔍")
                st.dataframe(aggregate_results)
                plot_aggregate_analysis_summary(aggregate_results)
                progress_bar.progress(80)
                
            # Completion status
            ctx.close()
            progress_bar.progress(100)
            status_message.success("Disconnected from Snowflake 📴")
        except Exception as e:
            progress_bar.progress(0)
            status_message.error(f"Failed to run comparison: {e}")

if __name__ == "__main__":
    main()