import psycopg
from psycopg import sql
from psycopg import OperationalError, IntegrityError
from logger import logger


class DatabaseConnection:
    """
    A class to manage the connection to a PostgreSQL database.
    This class provides methods to connect to the database, execute queries,
    and insert rows into tables.
    It uses the psycopg library for database interactions.
    Attributes:
        dbname (str): Name of the database.
        user (str): Username for the database connection.
        password (str): Password for the database connection.
        host (str): Host where the database is located.
        port (int): Port number for the database connection.
        connection (psycopg.Connection): Connection object to interact with the database.
    """

    def __init__(self, dbname, user, password, host, port):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.connection = None

    def connect(self):
        """Establish a connection to the PostgreSQL database."""
        try:
            self.connection = psycopg.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            logger.info("Connection to the database established successfully.")
        except OperationalError as e:
            logger.error(f"An error occurred while connecting to the database: {e}")
            raise

    def disconnect(self):
        """Close the connection to the PostgreSQL database."""
        if self.connection:
            self.connection.close()
            logger.info("Connection to the database closed.")

    def execute_query(self, query):
        """
        Execute the specified query and return the result.

        :param query: SQL query to execute.
        :return: Result of the query.
        """
        if not self.connection:
            raise RuntimeError("Database connection is not established.")

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                logger.info(f"Query executed successfully.")
            self.connection.commit()
            return result
        except Exception as e:
            logger.error(f"An error occurred while executing the query: {e}")
            raise

    def insert_rows(self, table, rows, columns, return_columns=['id']):
        """
        Insert multiple rows into the specified table and return the IDs of the inserted rows.

        :param table: Name of the table to insert into.
        :param rows: List of dictionaries, where each dictionary contains column names and values.
        :return: List of IDs of the inserted rows.
        """
        if not self.connection:
            raise RuntimeError("Database connection is not established.")

        if not rows:
            raise ValueError("No rows provided for insertion.")

        # Construct the INSERT query
        query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) RETURNING {}").format(
            sql.Identifier(table),
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join(map(sql.Placeholder, columns)),
            sql.SQL(', ').join(map(sql.Identifier, return_columns)),
        )

        try:
            with self.connection.cursor() as cursor:
                inserted_returns = []
                for row in rows:
                    cursor.execute(query, row)
                    inserted_return = cursor.fetchone()
                    inserted_returns.append(inserted_return)
                self.connection.commit()
                logger.info(f"Inserted {len(rows)} rows successfully.")
                return inserted_returns
        except IntegrityError as e:
            self.connection.rollback()
            logger.error(f"Integrity error occurred: {e}")
            raise
        except Exception as e:
            self.connection.rollback()
            logger.error(f"An error occurred while inserting rows: {e}")
            raise

    def __enter__(self):
        """Context manager entry point."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.disconnect()
