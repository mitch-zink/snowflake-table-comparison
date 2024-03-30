"""
Description: This application compares two tables in Snowflake based on a primary key column, then displays the differences
Author: Mitch Zink
Last Updated: 3/30/2024
"""

import pandas as pd
import snowflake.connector
import streamlit as st

def compare_dataframes(df1, df2, key_column):
    """
    Compares two DataFrames based on a key column to find rows that are different or missing.
    Merges the dataframes to identify common and unique rows.
    """
    merged_df = df1.merge(df2, on=key_column, how='outer', indicator=True, suffixes=('_original', '_modified'))
    differences = merged_df[merged_df['_merge'] != 'both']
    
    # Detailed comparison for rows present in both dataframes
    if not merged_df[merged_df['_merge'] == 'both'].empty:
        matched_df = merged_df[merged_df['_merge'] == 'both']
        compare_cols = [col for col in df1.columns if col != key_column and col not in ['_merge']]
        diff_data = {'key': [], 'column': [], 'original_value': [], 'modified_value': []}
        
        # Iterate over each row and column to find differences
        for _, row in matched_df.iterrows():
            for col in compare_cols:
                original_val = row[f'{col}_original']
                modified_val = row[f'{col}_modified']
                if pd.notnull(original_val) != pd.notnull(modified_val) or (pd.notnull(original_val) and original_val != modified_val):
                    diff_data['key'].append(row[key_column])
                    diff_data['column'].append(col)
                    diff_data['original_value'].append(original_val)
                    diff_data['modified_value'].append(modified_val)
        
        # Convert the collected differences into a DataFrame
        matched_but_different = pd.DataFrame(diff_data) if diff_data['key'] else pd.DataFrame(columns=['key', 'column', 'original_value', 'modified_value'])
    else:
        matched_but_different = pd.DataFrame(columns=['key', 'column', 'original_value', 'modified_value'])
    
    # Include relevant columns for the differences DataFrame
    if not differences.empty:
        differences = differences[[key_column] + [col for col in differences.columns if '_original' in col or '_modified' in col or col == '_merge']]
    
    return differences, matched_but_different

def main():
    """
    Main function for the Streamlit app, setting up the layout, inputs, and executing the comparison.
    """
    st.title('Snowflake Table Comparison Tool')
    st.sidebar.header('Configuration')
    # Inputs for Snowflake connection details
    user = st.sidebar.text_input("User")
    account = st.sidebar.text_input("Account")
    warehouse = st.sidebar.text_input("Warehouse")
    database = st.sidebar.text_input("Database")
    schema = st.sidebar.text_input("Schema")
    primary_key_column = st.sidebar.text_input("Primary Key Column for Comparison")
    
    # Option to select authentication method
    use_external_auth = st.sidebar.checkbox("Use External Browser Authentication")
    
    password = ""
    if not use_external_auth:
        password = st.sidebar.text_input("Password", type="password")
        authenticator = 'snowflake'  # Default Snowflake authentication
    else:
        authenticator = 'externalbrowser'  # External browser-based SSO
    
    # SQL query inputs for original and modified datasets
    query_original = st.sidebar.text_area("Query for original table", height=150)
    query_modified = st.sidebar.text_area("Query for modified table", height=150)
    
    # Button to trigger the comparison process
    if st.sidebar.button("Connect and Compare Tables"):
        if user and account and (use_external_auth or password) and primary_key_column:
            with st.spinner("Connecting to Snowflake..."):
                # Establish connection based on the selected authentication method
                try:
                    ctx = snowflake.connector.connect(
                        user=user,
                        account=account,
                        password=password if not use_external_auth else None,
                        authenticator=authenticator,
                        warehouse=warehouse,
                        database=database,
                        schema=schema
                    )
                    st.success("Connected to Snowflake.")

                    # Progress bar for visual feedback
                    progress_bar = st.progress(0)

                    # Fetching and displaying the original dataset
                    st.header("Original Dataset")
                    df_original = pd.read_sql(query_original, ctx)
                    st.dataframe(df_original)
                    progress_bar.progress(33)

                    # Fetching and displaying the modified dataset
                    st.header("Modified Dataset")
                    df_modified = pd.read_sql(query_modified, ctx)
                    st.dataframe(df_modified)
                    progress_bar.progress(67)

                    # Performing and displaying the comparison results
                    st.header("Comparison Results")
                    differences, matched_but_different = compare_dataframes(df_original, df_modified, primary_key_column)
                    progress_bar.progress(100)

                    if not differences.empty:
                        st.error("Rows with differences or missing in one of the tables:")
                        st.dataframe(differences)
                    else:
                        st.success("No rows are missing or exclusively present in one of the tables.")

                    if not matched_but_different.empty:
                        st.error("Rows with the same ID but different values")
                        st.dataframe(matched_but_different)
                    else:
                        st.success("No rows have the same ID but different values.")
                    
                    ctx.close()  # Close the Snowflake connection
                except Exception as e:
                    st.error(f"Failed to connect or compare: {e}")
        else:
            st.error("Please fill in all required fields.")

if __name__ == "__main__":
    main()
