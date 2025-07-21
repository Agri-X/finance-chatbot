import logging
from datetime import datetime

import pytz
from langchain.tools import tool
from tabulate import tabulate
from tzlocal import get_localzone

from utils.scheduler import ScheduledTask, TaskScheduler

from chainlit_local import chainlit_global as cl


@tool
async def schedule_task_tool(
    time_to_execute: str,
    prompt: str,
    timezone: str = "local",
):
    """
    Schedule a task to be executed by the AI agent at a specific future time.

    Args:
        time_to_execute: Scheduled time in "YYYY-MM-DD HH:MM:SS" format
        prompt: Main instruction for the agent to execute
        timezone: Timezone for execution time (e.g., 'UTC', 'America/New_York')

    Returns:
        str: Status message about the scheduled task
    """

    task_scheduler = cl.user_session.get("task_scheduler")
    assert isinstance(task_scheduler, TaskScheduler)

    user = cl.user_session.get("user")
    assert isinstance(user, cl.User)

    if not task_scheduler:
        return "Error: Task scheduler not initialized."

    try:
        # Parse timezone
        if timezone == "local":
            target_tz = pytz.timezone(str(get_localzone()))
        else:
            target_tz = pytz.timezone(timezone)

        # Parse execution time
        execution_dt_naive = datetime.strptime(time_to_execute, "%Y-%m-%d %H:%M:%S")
        execution_dt = target_tz.localize(execution_dt_naive)
        now_dt = datetime.now(target_tz)

        if execution_dt < now_dt:
            return f"Error: The time '{time_to_execute}' is in the past."

        # Create task
        task_id = f"task_{int(datetime.now().timestamp())}"
        task = ScheduledTask(
            task_id=task_id,
            execution_time=execution_dt,
            prompt=prompt,
            timezone=timezone,
        )

        # Add to scheduler
        if task_scheduler.add_task(task, user.identifier):
            scheduled_time_str = execution_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            return f"Task '{prompt}' scheduled successfully for {scheduled_time_str}."
        else:
            return "Error: Failed to schedule task."

    except (ValueError, pytz.UnknownTimeZoneError) as e:
        return f"Error scheduling task: {e}"
    except Exception as e:
        logging.exception(f"Unexpected error in schedule_task_tool: {e}")
        return "An unexpected error occurred while scheduling the task."


async def list_scheduled_tasks() -> str:
    """
    List all scheduled tasks with their status and execution times.

    Returns:
        str: Formatted list of all scheduled tasks
    """

    task_scheduler = cl.user_session.get("task_scheduler")
    assert isinstance(task_scheduler, TaskScheduler)

    user = cl.user_session.get("user")
    assert isinstance(user, cl.User)

    if not task_scheduler:
        return "Error: Task scheduler not initialized."

    tasks = task_scheduler.get_all_tasks(user.identifier)

    if not tasks:
        return "No scheduled tasks found."

    # Prepare data for tabulate
    headers = [
        "Status",
        "Task ID",
        "Prompt",
        "Scheduled",
        "Last Executed",
        "Exec Count",
    ]
    table_data = []

    for task in sorted(tasks, key=lambda x: x.execution_time):
        status_emoji = {
            "pending": "⏳",
            "executing": "🔄",
            "completed": "✅",
            "failed": "❌",
        }.get(task.status, "❓")

        scheduled_time = task.execution_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        last_executed_time = (
            task.last_executed.strftime("%Y-%m-%d %H:%M:%S")
            if task.last_executed
            else "N/A"
        )

        table_data.append(
            [
                f"{status_emoji} {task.status}",
                task.task_id,
                f"{task.prompt[:50]}...",
                scheduled_time,
                last_executed_time,
                task.execution_count,
            ]
        )

    # Generate the table
    result = "Scheduled Tasks:\n\n"
    result += tabulate(table_data, headers=headers, tablefmt="github")

    return result


@tool
async def cancel_scheduled_task(task_id: str) -> str:
    """
    Cancel a scheduled task by its ID.

    Args:
        task_id: The ID of the task to cancel

    Returns:
        str: Status message about the cancellation
    """

    task_scheduler = cl.user_session.get("task_scheduler")
    assert isinstance(task_scheduler, TaskScheduler)

    user = cl.user_session.get("user")
    assert isinstance(user, cl.User)

    if not task_scheduler:
        return "Error: Task scheduler not initialized."

    if task_scheduler.remove_task(task_id, user.identifier):
        return f"Task {task_id} has been successfully cancelled."
    else:
        return f"Task {task_id} not found or could not be cancelled."


scheduler_tools = [schedule_task_tool, list_scheduled_tasks, cancel_scheduled_task]
