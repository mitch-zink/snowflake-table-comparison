import pandas as pd
import snowflake.connector
import streamlit as st

def compare_dataframes(df1, df2, key_column):
    # Merging DataFrames on the key column
    merged_df = df1.merge(df2, on=key_column, how='outer', indicator=True, suffixes=('_original', '_modified'))
    # Finding rows that are either different or missing in one of the DataFrames
    differences = merged_df[merged_df['_merge'] != 'both']
    # Finding rows with matching keys but different values
    matched_but_different = merged_df[(merged_df['_merge'] == 'both') & 
                                      (merged_df.filter(like='_original').values != merged_df.filter(like='_modified').values).any(axis=1)]
    return differences, matched_but_different

def main():
    st.title("Snowflake Table Comparison Tool")

    # User inputs for Snowflake connection and primary key column
    user = st.text_input("User")
    account = st.text_input("Account")
    authenticator = 'externalbrowser'  # Assuming 'externalbrowser' for simplicity
    warehouse = st.text_input("Warehouse")
    database = st.text_input("Database")
    schema = st.text_input("Schema")
    primary_key_column = st.text_input("Primary Key Column for Comparison")

    # Initialize session state for queries if not already done
    if 'query_original' not in st.session_state:
        st.session_state['query_original'] = ""
    if 'query_modified' not in st.session_state:
        st.session_state['query_modified'] = ""

    # Text areas for SQL queries, bound to session state variables
    st.session_state['query_original'] = st.text_area("Query for original table", value=st.session_state['query_original'], height=150)
    st.session_state['query_modified'] = st.text_area("Query for modified table", value=st.session_state['query_modified'], height=150)

    if st.button("Connect and Compare Tables"):
        with st.spinner("Connecting to Snowflake..."):
            try:
                ctx = snowflake.connector.connect(
                    user=user,
                    account=account,
                    authenticator=authenticator,
                    warehouse=warehouse,
                    database=database,
                    schema=schema
                )
                st.success("Connected to Snowflake.")

                # Loading datasets
                df_original = pd.read_sql(st.session_state['query_original'], ctx)
                df_modified = pd.read_sql(st.session_state['query_modified'], ctx)

                # Comparing DataFrames
                if primary_key_column:  # Proceed if a primary key column has been specified
                    differences, matched_but_different = compare_dataframes(df_original, df_modified, primary_key_column)
                    
                    # Displaying comparison results
                    if not differences.empty:
                        st.write("Rows with differences or missing in one of the tables:")
                        st.dataframe(differences)
                    else:
                        st.success("No rows are missing or exclusively present in one of the tables.")

                    if not matched_but_different.empty:
                        st.write("Rows with the same ID but different values:")
                        st.dataframe(matched_but_different)
                    else:
                        st.success("No rows have the same ID but different values.")
                else:
                    st.error("Please specify a Primary Key Column for Comparison.")

                ctx.close()
            except Exception as e:
                st.error(f"Failed to connect or compare: {e}")

if __name__ == "__main__":
    main()
