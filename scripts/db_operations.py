#week_02_challenge/scripts/db_operations.py

import oracledb
import configparser
import os
import pandas as pd

def get_db_config():
    """Reads database configuration from db_config.ini."""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'db_config.ini')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}. Please create config/db_config.ini.")
    
    config.read(config_path)
    
    if 'oracle_db' not in config:
        raise ValueError("Section 'oracle_db' not found in db_config.ini.")
        
    return config['oracle_db']

def connect_to_oracle():
    """Establishes and returns a connection to the Oracle database."""
    db_config = get_db_config()
    try:
        connection = oracledb.connect(
            user=db_config['user'],
            password=db_config['password'],
            dsn=db_config['dsn']
        )
        print("Successfully connected to Oracle Database.")
        return connection
    except oracledb.Error as e:
        error_obj, = e.args
        print(f"Error connecting to Oracle Database: {error_obj.message} (Code: {error_obj.code})")
        print("Please check your db_config.ini and Oracle XE/Instant Client setup.")
        return None

def create_tables(connection):
    """Creates BANKS and APP_REVIEWS tables if they don't exist."""
    cursor = connection.cursor()
    try:
        # Create BANKS table
        cursor.execute("""
            CREATE TABLE BANKS (
                BANK_ID NUMBER GENERATED BY DEFAULT ON NULL AS IDENTITY PRIMARY KEY,
                BANK_NAME VARCHAR2(255) UNIQUE NOT NULL
            )
        """)
        print("Table BANKS created successfully.")
    except oracledb.Error as e:
        error_obj, = e.args
        if error_obj.code == 942: # ORA-00942: table or view does not exist (already exists)
            print("Table BANKS already exists.")
        else:
            print(f"Error creating BANKS table: {error_obj.message} (Code: {error_obj.code})")
            return False

    try:
        # Create APP_REVIEWS table
        cursor.execute("""
            CREATE TABLE APP_REVIEWS (
                REVIEW_ID VARCHAR2(255) PRIMARY KEY,
                USER_NAME VARCHAR2(255),
                RATING NUMBER(1),
                REVIEW_DATE DATE,
                REVIEW_TEXT CLOB,
                BANK_ID NUMBER NOT NULL, -- Foreign Key to BANKS table
                SOURCE VARCHAR2(50),
                SENTIMENT VARCHAR2(20),
                SENTIMENT_SCORE NUMBER(5, 2),
                PROCESSED_REVIEW_TOKENS CLOB, -- Store as JSON string or CLOB
                EXTRACTED_KEYWORDS CLOB,     -- Store as JSON string or CLOB
                IDENTIFIED_THEME VARCHAR2(255),
                CONSTRAINT FK_BANK
                    FOREIGN KEY (BANK_ID)
                    REFERENCES BANKS(BANK_ID)
            )
        """)
        print("Table APP_REVIEWS created successfully.")
    except oracledb.Error as e:
        error_obj, = e.args
        if error_obj.code == 942: # ORA-00942: table or view does not exist (already exists)
            print("Table APP_REVIEWS already exists.")
        else:
            print(f"Error creating APP_REVIEWS table: {error_obj.message} (Code: {error_obj.code})")
            return False
    
    connection.commit()
    return True

def insert_review_data(connection, df):
    """Inserts processed review data into the Oracle database."""
    cursor = connection.cursor()

    # Get distinct bank names and insert into BANKS table first
    bank_names = df['Bank/App Name'].unique()
    bank_id_map = {}
    
    for bank_name in bank_names:
        try:
            cursor.execute("INSERT INTO BANKS (BANK_NAME) VALUES (:1) RETURNING BANK_ID INTO :2", [bank_name, cursor.var(oracledb.NUMBER)])
            bank_id = cursor.getvalue(0)
            bank_id_map[bank_name] = bank_id
            print(f"Inserted bank '{bank_name}' with ID {bank_id}.")
        except oracledb.IntegrityError as e:
            # If bank already exists, fetch its ID
            if e.args[0].code == 1: # ORA-00001: unique constraint violated
                cursor.execute("SELECT BANK_ID FROM BANKS WHERE BANK_NAME = :1", [bank_name])
                bank_id = cursor.fetchone()[0]
                bank_id_map[bank_name] = bank_id
                print(f"Bank '{bank_name}' already exists with ID {bank_id}.")
            else:
                raise # Re-raise other integrity errors

    connection.commit() # Commit bank inserts

    # Prepare data for APP_REVIEWS insertion
    insert_sql = """
    INSERT INTO APP_REVIEWS (
        REVIEW_ID, USER_NAME, RATING, REVIEW_DATE, REVIEW_TEXT, 
        BANK_ID, SOURCE, SENTIMENT, SENTIMENT_SCORE, 
        PROCESSED_REVIEW_TOKENS, EXTRACTED_KEYWORDS, IDENTIFIED_THEME
    )
    VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12)
    """

    data_to_insert = []
    for index, row in df.iterrows():
        bank_id = bank_id_map.get(row['Bank/App Name'])
        if bank_id is None:
            print(f"Warning: Bank ID not found for '{row['Bank/App Name']}'. Skipping reviewId: {row['reviewId']}")
            continue
            
        data_to_insert.append((
            row['reviewId'],
            row['User Name'],
            row['Rating'],
            row['Date'], # Datetime objects are handled by cx_Oracle
            row['Review Text'],
            bank_id,
            row['Source'],
            row['Sentiment'],
            row['Sentiment_Score'],
            str(row['Processed_Reviews_Tokens']), # Store list as string representation or JSON
            str(row['Extracted_Keywords']),       # Store list as string representation or JSON
            row['Identified_Theme']
        ))

    try:
        cursor.executemany(insert_sql, data_to_insert, batcherrors=True)
        for error in cursor.getbatcherrors():
            print(f"Error inserting row {error.offset}: {error.message}")
        connection.commit()
        print(f"Successfully inserted {len(data_to_insert)} records into APP_REVIEWS.")
    except oracledb.Error as e:
        print(f"Error during bulk insertion: {e}")
        connection.rollback() # Rollback on error
    finally:
        cursor.close()

if __name__ == "__main__":
    processed_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
    input_filepath = os.path.join(processed_data_dir, 'fintech_app_reviews_analyzed.csv')

    if not os.path.exists(input_filepath):
        print(f"Error: Analyzed data file not found at {input_filepath}. Please run analyze_reviews.py first.")
    else:
        df = pd.read_csv(input_filepath)
        
        # Ensure 'Date' column is in datetime format for Oracle insertion
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure 'reviewId' is string
        df['reviewId'] = df['reviewId'].astype(str)

        connection = None
        try:
            connection = connect_to_oracle()
            if connection:
                if create_tables(connection):
                    insert_review_data(connection, df)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            if connection:
                connection.close()
                print("Database connection closed.")