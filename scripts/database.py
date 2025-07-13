"""
database.py
-----------
Handles chat history storage and retrieval using a PostgreSQL database.
- Manages connection, table creation, and message persistence.
- Used for saving and restoring chat sessions.
"""

from dotenv import load_dotenv
import os
import psycopg2
from logger import setup_logger
from langchain_postgres import PostgresChatMessageHistory
import json

class PostgresChatHistory:
    """
    Manages chat history in a PostgreSQL database.
    - Connects to the database using environment variables.
    - Creates the chat history table if needed.
    - Saves chat messages for each session.
    """
    def __init__(self):
        """
        Initialize the database connection and logger.
        """
        load_dotenv()
        self.DB_NAME = os.getenv("DB_NAME")
        self.DB_USER = os.getenv("DB_USER")
        self.DB_PASS = os.getenv("DB_PASS")
        self.DB_HOST = os.getenv("DB_HOST")
        self.logger = setup_logger(__name__, "database.log")
        self.table_name = "chat_history"
        try:
            self.conn = psycopg2.connect(
                dbname=self.DB_NAME,
                user=self.DB_USER,
                password=self.DB_PASS,
                host=self.DB_HOST
            )
            self.logger.info("Connected to the database successfully.")
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise

    def create_chat_table(self):
        """
        Creates the chat history table in the PostgreSQL database.
        """
        try:
            PostgresChatMessageHistory.create_tables(
                self.conn,
                self.table_name
            )
        except Exception as e:
            self.logger.error(f"Failed to create chat table: {e}")
            self.conn.rollback()

    def save_chat_history(self, session_id: str, messages: list = None):
        """
        Saves chat history to the PostgreSQL database.

        Parameters:
            session_id (str): Unique identifier for the chat session.
            messages (list): List of messages to save. If None, retrieves existing messages.
        """
        try:
            if messages is not None:
                with self.conn.cursor() as cursor:
                    query = f"""
                    INSERT INTO {self.table_name} (session_id, message)
                    VALUES (%s, %s)
                    """

                    values = [
                        (session_id, json.dumps({"type": message.type, "data": message.model_dump()}))
                        for message in messages
                    ]

                    cursor.executemany(query, values)

                self.conn.commit()

            self.logger.info(f"Chat history saved for session {session_id}.")
        except Exception as e:
            self.logger.error(f"Failed to save chat history: {e}")

