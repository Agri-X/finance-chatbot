from datetime import UTC, datetime, timedelta
from typing import Optional
from langchain.tools import tool

import pytz


@tool
def get_current_time(
    format_string: str = "%Y-%m-%d %H:%M:%S %Z%z", timezone: str = "local"
) -> str:
    """
    Retrieves the current date and time, formatted according to specified
    parameters and timezone.

    This function is designed to be comprehensive for AI use, allowing for
    flexible time representation based on various needs (e.g., local time, UTC,
    specific time zones, custom formatting).

    Args:
        format_string (str, optional): A string representing the desired output
            format for the datetime. This uses standard `strftime` directives.
            Common directives include:
            - %Y: Year with century (e.g., 2023)
            - %m: Month as a zero-padded decimal number (e.g., 01, 12)
            - %d: Day of the month as a zero-padded decimal (e.g., 01, 31)
            - %H: Hour (24-hour clock) as a zero-padded decimal (e.g., 00, 23)
            - %M: Minute as a zero-padded decimal (e.g., 00, 59)
            - %S: Second as a zero-padded decimal (e.g., 00, 59)
            - %f: Microsecond as a zero-padded decimal (e.g., 000000, 999999)
            - %j: Day of the year as a zero-padded decimal (e.g., 001, 366)
            - %w: Weekday as a decimal number (0-6, Sunday is 0)
            - %a: Locale's abbreviated weekday name (e.g., Mon, Tue)
            - %A: Locale's full weekday name (e.g., Monday, Tuesday)
            - %b: Locale's abbreviated month name (e.g., Jan, Feb)
            - %B: Locale's full month name (e.g., January, February)
            - %c: Locale's appropriate date and time representation
            - %x: Locale's appropriate date representation
            - %X: Locale's appropriate time representation
            - %Z: Time zone name (e.g., EST, CST)
            - %z: UTC offset in the form +HHMM or -HHMM (e.g., -0500, +0100)
            - %p: Locale's equivalent of either AM or PM
            - %%: A literal '%' character
            Defaults to "%Y-%m-%d %H:%M:%S %Z%z" (e.g., "2023-10-27 15:30:00 WIB+0700").

        timezone (str, optional): The desired timezone for the output.
            Can be:
            - "local": Uses the system's local timezone.
            - "UTC": Uses Coordinated Universal Time.
            - A valid IANA timezone string (e.g., "America/New_York",
              "Europe/London", "Asia/Jakarta").
            Defaults to "local".

    Returns:
        str: A string representing the current date and time, formatted
             according to the `format_string` and `timezone` parameters.

    Raises:
        pytz.UnknownTimeZoneError: If an invalid IANA timezone string is provided.
        ValueError: If the format_string is invalid.

    Example Usage:
        >>> get_current_time()
        # Example output: '2023-10-27 15:30:00 WIB+0700' (depends on local timezone)

        >>> get_current_time(timezone="UTC")
        # Example output: '2023-10-27 08:30:00 UTC+0000'

        >>> get_current_time(format_string="%H:%M:%S", timezone="America/New_York")
        # Example output: '04:30:00' (if current UTC is 08:30:00)

        >>> get_current_time(format_string="%A, %B %d, %Y %I:%M %p", timezone="Asia/Tokyo")
        # Example output: 'Friday, October 27, 2023 05:30 PM'
    """
    try:
        if timezone == "local":
            now = datetime.now()
            localized_now = now.astimezone()
        elif timezone == "UTC":
            localized_now = datetime.now(UTC)
        else:
            # For specific IANA timezones
            tz = pytz.timezone(timezone)
            localized_now = datetime.now(tz)

        return f"Current time is: {localized_now.strftime(format_string)}"
    except pytz.UnknownTimeZoneError:
        return f"Error: Unknown timezone '{timezone}'. Please provide a valid IANA timezone string."
    except ValueError as e:
        return f"Error: Invalid format string '{format_string}'. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


@tool
def calculate_relative_time(
    delta_value: int,
    delta_unit: str,
    operation: str = "add",
    start_time: Optional[datetime] = None,
    format_string: str = "%Y-%m-%d %H:%M:%S %Z%z",
    timezone: str = "local",
) -> str:
    """
    Calculates a new datetime by adding or subtracting a specified duration
    (hours, minutes, or days) from a given start time.

    This function is designed to be comprehensive for AI use, allowing for
    flexible time adjustments and representation based on various needs.

    Args:
        delta_value (int): The integer value of the duration to add or subtract.
                           Must be a non-negative integer.
        delta_unit (str): The unit of the duration. Valid options are:
                          "hours", "minutes", "days". Case-insensitive.
        operation (str, optional): The operation to perform.
                                   "add" to add the duration (e.g., "5 hours from now").
                                   "subtract" to subtract the duration (e.g., "5 hours ago").
                                   Defaults to "add". Case-insensitive.
        start_time (datetime, optional): The starting datetime object from
            which to calculate the relative time. If None, the current time
            (localized to the specified `timezone`) will be used.
            It is highly recommended to pass timezone-aware datetime objects
            if `start_time` is provided to avoid ambiguity.
        format_string (str, optional): A string representing the desired output
            format for the resulting datetime. This uses standard `strftime` directives.
            (See `get_current_time` docstring for common directives).
            Defaults to "%Y-%m-%d %H:%M:%S %Z%z".
        timezone (str, optional): The desired timezone for the output and for
            localizing the `start_time` if it's None or naive.
            Can be:
            - "local": Uses the system's local timezone.
            - "UTC": Uses Coordinated Universal Time.
            - A valid IANA timezone string (e.g., "America/New_York",
              "Europe/London", "Asia/Jakarta").
            Defaults to "local".

    Returns:
        str: A string representing the calculated date and time, formatted
             according to the `format_string` and `timezone` parameters.

    Raises:
        ValueError: If `delta_value` is negative, `delta_unit` is invalid,
                    `operation` is invalid, or `format_string` is invalid.
        pytz.UnknownTimeZoneError: If an invalid IANA timezone string is provided.

    Example Usage:
        >>> calculate_relative_time(5, "hours")
        # Example output: '2023-10-27 20:30:00 WIB+0700' (5 hours from now)

        >>> calculate_relative_time(12, "hours", operation="subtract")
        # Example output: '2023-10-27 03:30:00 WIB+0700' (12 hours ago)

        >>> calculate_relative_time(3, "days", timezone="Europe/London")
        # Example output: '2023-10-30 09:30:00 BST+0100' (3 days from now in London)

        >>> from datetime import datetime
        >>> specific_time = datetime(2023, 1, 1, 10, 0, 0) # Naive datetime
        >>> calculate_relative_time(2, "hours", start_time=specific_time, timezone="America/New_York")
        # Example output: '2023-01-01 12:00:00 EST-0500' (2 hours from Jan 1, 2023 10 AM in NY)
    """
    if delta_value < 0:
        return "Error: 'delta_value' must be a non-negative integer."

    delta_unit = delta_unit.lower()
    if delta_unit not in ["hours", "minutes", "days"]:
        return f"Error: Invalid 'delta_unit'. Must be 'hours', 'minutes', or 'days'."

    operation = operation.lower()
    if operation not in ["add", "subtract"]:
        return "Error: Invalid 'operation'. Must be 'add' or 'subtract'."

    try:
        # Determine the base time
        if start_time is None:
            if timezone == "local":
                # Get local naive datetime and then localize it
                base_time = datetime.now().astimezone()
            elif timezone == "UTC":
                base_time = datetime.now(UTC)
            else:
                tz = pytz.timezone(timezone)
                base_time = datetime.now(tz)
        else:
            # If start_time is provided, ensure it's timezone-aware if possible,
            # or localize it based on the provided timezone if naive.
            if (
                start_time.tzinfo is None
                or start_time.tzinfo.utcoffset(start_time) is None
            ):
                if timezone == "local":
                    base_time = start_time.astimezone()
                elif timezone == "UTC":
                    base_time = start_time.replace(tzinfo=UTC)
                else:
                    tz = pytz.timezone(timezone)
                    base_time = tz.localize(start_time)
            else:
                base_time = start_time.astimezone(
                    pytz.timezone(timezone)
                    if timezone not in ["local", "UTC"]
                    else None
                )

        # Create the timedelta object
        delta_kwargs = {delta_unit: delta_value}
        time_delta = timedelta(**delta_kwargs)

        # Apply the operation
        if operation == "add":
            result_time = base_time + time_delta
        else:  # operation == "subtract"
            result_time = base_time - time_delta

        return result_time.strftime(format_string)

    except pytz.UnknownTimeZoneError:
        return f"Error: Unknown timezone '{timezone}'. Please provide a valid IANA timezone string."
    except ValueError as e:
        return f"Error: Invalid format string '{format_string}'. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

time_tools = [calculate_relative_time, get_current_time]