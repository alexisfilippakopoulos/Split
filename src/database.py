import sqlite3
import threading

DB_LOCK = threading.Lock()

class Database():
    def __init__(self, db_path, table_queries) -> None:
        self.db_path = db_path
        self.create_db_schema(table_queries) 
    
    def create_db_schema(self, create_table_queries: dict):
        """
        Creates the database schema.
        """
        for table_name, query in create_table_queries.items():
            self.execute_query(query=query) if not self.check_table_existence(target_table=table_name) else None
        print('[+] Database schema created/loaded successsfully')

    def check_table_existence(self, target_table: str) -> bool:
        """
        Checks if a specific table exists within the database.
        Args:
            target_table: Table to look for.
        Returns:
            True or False depending on existense.
        """
        query = "SELECT name FROM sqlite_master WHERE type ='table'"
        tables = self.execute_query(query=query, values=None, fetch_data_flag=True, fetch_all_flag=True)
        exists = any(table[0] == target_table for table in tables) if tables is not None else False
        return exists
    
    def execute_query(self, query: str, values=None, fetch_data_flag=False, fetch_all_flag=False):
        """
        Executes a given query. Either for retrieval or update purposes.
        Args:
            query: Query to be executed
            values: Query values
            fetch_data_flag: Flag that signals a retrieval query
            fetch_all_flag: Flag that signals retrieval of all table data or just the first row.
        Returns:
            The data fetched for a specified query. If it is not a retrieval query then None is returned. 
        """
        with DB_LOCK:
            try:
                connection = sqlite3.Connection(self.db_path)
                cursor = connection.cursor()
                cursor.execute(query, values) if values is not None else cursor.execute(query)
                fetched_data = (cursor.fetchall() if fetch_all_flag else cursor.fetchone()[0]) if fetch_data_flag else None
                connection.commit()
                connection.close()        
                return fetched_data
            except sqlite3.Error as error:
                print(f'{query} \nFailed with error:\n{error}')