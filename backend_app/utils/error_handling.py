"""
Centralized error handling utilities for the Car Troubleshooting Chatbot API.
"""


def handle_error(logger, error_message, exception):
    """Utility function to handle errors consistently."""
    logger.error(f"{error_message}: {exception}")
    raise exception
