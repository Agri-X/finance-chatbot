import json
import logging
from typing import Literal

from langgraph.graph.message import MessagesState

from matplotlib import pyplot as plt
import chainlit as cl
import pandas as pd


def get_trailing_tool_messages_with_indices(arr):
    result = []
    for i in range(len(arr) - 1, -1, -1):
        item = arr[i]
        if hasattr(item, "type") and item.type == "tool":
            result.append((item, i))
        else:
            break
    return list(reversed(result))


async def generate_charts_from_embedded_data(chart_configs):
    """
    Generates and sends multiple plots from a configuration file
    that includes embedded data.

    Args:
        chart_configs (list): A list of dictionaries, where each dict
                              defines a chart and contains its own data.
    """
    logging.info("chart drawing called")

    for config in chart_configs:
        logging.info("creating chart for config: %s", config["title"])
        logging.info(config)
        # 1. Create a DataFrame from the embedded data
        df = pd.DataFrame(config["data"])

        # Set the index column (e.g., 'Date') and convert it to datetime
        if "index_column" in config:
            df.set_index(config["index_column"], inplace=True)
            df.index = pd.to_datetime(df.index)

        # 2. Create the plot
        figsize = tuple(config.get("figsize", (20, 10)))
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the primary data lines
        if "plots" in config:
            for plot_info in config["plots"]:
                plot_kwargs = {
                    "label": plot_info.get("label"),
                    "color": plot_info.get("color"),
                    "linestyle": plot_info.get("linestyle"),
                }
                plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}
                ax.plot(df[plot_info["data_key"]], **plot_kwargs)

        # Add any horizontal lines
        if "hlines" in config:
            for hline_info in config["hlines"]:
                ax.axhline(**hline_info)

        # 3. Finalize and send the plot
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        await cl.Message(
            content=config["title"],
            elements=[
                cl.Pyplot(name="plot", figure=fig, display="inline"),
            ],
        ).send()
    return [x["title"] for x in chart_configs if "title" in x]


async def render_chart(state: MessagesState):
    messages = state["messages"]
    trailing_tool_messages = get_trailing_tool_messages_with_indices(messages)

    if not trailing_tool_messages:
        logging.info("No trailing 'tool' messages found for chart generation.")
        return {"messages": messages}

    for tool_message_obj, original_index in trailing_tool_messages:
        try:
            tool_content = json.loads(tool_message_obj.content)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding error at index {original_index}: {e}")
            continue

        if "chart" in tool_content:
            original_message_text = tool_content.get("message", "")
            try:
                charts = await generate_charts_from_embedded_data(tool_content["chart"])
                del tool_content["chart"]
                log_message = (
                    f"LOG: Chart(s) for {', '.join(charts)} have been sent to client."
                )
                tool_content["message"] = (
                    f"{original_message_text}\n{log_message}".strip()
                )
                logging.info(log_message)

            except Exception as e:
                logging.error(f"Chart generation error at index {original_index}: {e}")
                if "chart" in tool_content:
                    del tool_content["chart"]
                error_message = f"Error generating charts: {e}. Please check the chart configuration."
                tool_content["message"] = (
                    f"{original_message_text}\n{error_message}".strip()
                )

        messages[original_index].content = json.dumps(tool_content)

    return {"messages": messages}


def should_render_chart(state: MessagesState) -> Literal["render_chart", "agent"]:
    messages = state["messages"]
    if not messages:
        logging.info("No messages found in state.")
        return "agent"

    trailing_tool_messages = get_trailing_tool_messages_with_indices(messages)

    if not trailing_tool_messages:
        logging.info("No trailing 'tool' messages found for chart generation.")
        return "agent"

    for tool_message_obj, original_index in trailing_tool_messages:
        try:
            tool_content = json.loads(tool_message_obj.content)
            if isinstance(tool_content, dict) and "chart" in tool_content:
                logging.info(
                    f"Chart detected in tool message at index {original_index}."
                )
                return "render_chart"
        except json.JSONDecodeError as e:
            logging.warning(
                f"JSON decoding error in tool message at index {original_index}: {e}. Skipping this message."
            )
        except TypeError as e:
            logging.warning(
                f"Type error for tool message content at index {original_index}: {e}. Content: {tool_message_obj.content}. Skipping this message."
            )
        except Exception as e:
            logging.error(
                f"Unexpected error processing tool message at index {original_index}: {e}. Skipping this message."
            )

    logging.info("No chart content found in any trailing tool messages.")
    return "agent"
