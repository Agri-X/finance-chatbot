import json
import logging
import os
import asyncpg

logger = logging.getLogger(__name__)


async def query_watchlists():
    """
    Connects to a PostgreSQL database using asyncpg and executes a query
    to retrieve user watchlists with stock information aggregated as JSON objects.
    This function is designed to be run asynchronously.
    """
    connection = None
    try:
        connection = await asyncpg.connect(os.environ.get("DATABASE_URL"))
        logger.info("Asynchronous database connection established successfully.")

        sql_query = """SELECT w.user_id, json_agg( json_build_object('stock_symbol', w.stock_symbol, 'brought_price', w.bought_price, 'quantity', w.quantity ) ) AS stocks FROM "watchlists" w GROUP BY w.user_id;"""
        results = await connection.fetch(sql_query)

        res = []
        for row in results:
            user_id = row["user_id"]
            stocks_json = row["stocks"]
            stocks = json.loads(stocks_json)
            for stock in stocks:
                res.append(
                    {
                        "stock_symbol": stock.get("stock_symbol"),
                        "brought_price": stock.get("brought_price"),
                        "quantity": stock.get("quantity"),
                        "email": user_id,
                    }
                )
        return res

    except Exception as error:
        logger.info(f"Error while connecting to or querying PostgreSQL: {error}")
        raise error

    finally:
        if connection:
            await connection.close()
            logger.info("Asynchronous PostgreSQL connection is closed.")
