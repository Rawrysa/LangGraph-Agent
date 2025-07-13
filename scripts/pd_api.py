"""
pd_api.py
---------
Handles authentication and timesheet data retrieval from the PD Project Management API.
- Manages access token retrieval and API requests.
- Provides a method for dynamic timesheet queries with flexible filters.
"""

import requests
from dotenv import load_dotenv
import os
from logger import setup_logger
import json
import urllib.parse
from langgraph.config import get_stream_writer

class PDAPI:
    """
    Client for the PD Project Management API.
    - Handles authentication and access token management.
    - Provides methods for retrieving timesheet records with dynamic filters.
    """
    def __init__(self):
        """
        Initialize the PDAPI client, load environment variables, set up logging, and authenticate to retrieve an access token.
        """
        load_dotenv()
        self.base_url = "https://pd-projectmanagement-api.shesha.dev/api"
        self.logger = setup_logger(__name__, "pd_api.log")
        self.logger.info("Initializing PDAPI client.")
        self.access_token = self.get_token()
        self.logger.info("Access token retrieved successfully.")

    def get_token(self):
        """
        Authenticate with the PD API and retrieve an access token.

        Returns:
            str: The access token if authentication is successful, None otherwise.

        Raises:
            Exception: If authentication fails or the request encounters an error.
        """
        token_url = f"{self.base_url}/TokenAuth/Authenticate"

        try:
            payload = {
                "password": os.getenv("PD_PASSWORD"),
                "userNameOrEmailAddress": os.getenv("PD_USERNAME")
            }
            response = requests.post(token_url, json=payload)
            response.raise_for_status()
            access_token = response.json()['result']['accessToken']
            self.logger.info("Authentication successful. Access token obtained.")
            return access_token
        except requests.exceptions.RequestException as e:
            self.logger.error("Failed to authenticate with PD API: %s", e)
            raise Exception(f"Failed to authenticate with PD API: {e}")

    def get_timesheet(self, filter: str = None):
        """
        Retrieve timesheet records from the PD API using a dynamic filter.

        Parameters:
            filter (str, optional): A JSON string representing the filter to apply to the timesheet query. 
                The filter should be a JSON object (as a string) specifying conditions for the timesheet records to retrieve. 
                Example: '{"and": [{ "==": [ { "var": "weekRange" }, "Feb 03 - Feb 09" ]}, {"==": [ {"var": "status"}, "Submitted"]}]}'
                If the filter is double-encoded (escaped), it will be automatically decoded.

        Returns:
            dict or None: The JSON response from the PD API containing the timesheet records if successful, otherwise None.

        Raises:
            Exception: If there is an error authenticating or retrieving the timesheet data from the PD API.
        """
        self.logger.info("Retrieving timesheet records with filter: %s", filter)
        get_stream_writer()({"get timesheet": f"🫥 Retrieving timesheet records with filter: {filter}"})

        try:
            # Handle double-encoded/escaped filter strings
            if filter and filter.startswith('"') and filter.endswith('"'):

                filter = filter[1:-1]

                unescaped = bytes(filter, "utf-8").decode("unicode_escape")

                filter = json.loads(unescaped)

                filter = urllib.parse.quote(json.dumps(filter))

            timesheet_url = f"{self.base_url}/dynamic/Boxfusion.Projectmanagement/Timesheet/Crud/GetAll"

            if filter:
                timesheet_url += f"?filter={filter}"

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }

            response = requests.get(timesheet_url, headers=headers)
            response.raise_for_status()

            items = response.json()['result']['items']

            self.logger.info("Timesheet data retrieved successfully.")
            return items

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to retrieve timesheet: {e}")
            return None
