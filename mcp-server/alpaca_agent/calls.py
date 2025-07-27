import logging
from typing import List, Optional, Union
from datetime import datetime, timedelta
from alpaca.common import APIError
from alpaca.trading import CreateWatchlistRequest, OrderStatus, UpdateWatchlistRequest
from alpaca.trading.requests import GetOptionContractsRequest

from alpaca.trading.client import TradingClient
from alpaca.data import (
    BarSet,
    ContractType,
    CryptoBarsRequest,
    CryptoLatestQuoteRequest,
    OptionHistoricalDataClient,
    OptionSnapshotRequest,
    StockHistoricalDataClient,
    CryptoHistoricalDataClient,
    TimeFrameUnit,
)
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
)
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame
import pandas as pd

from models import (
    AlpacaOrder,
    AlpacaOrderRequest,
    AlpacaPosition,
    AlpacaAccount,
    AlpacaAsset,
    AlpacaQuote,
    AlpacaBar,
    AlpacaOrderType,
    AlpacaWatchlist,
)


def get_account(client: TradingClient):
    """
    Retrieve account details

    :param client: Alpaca trading client
    :return: AlpacaAccount model
    """
    account = client.get_account()
    return AlpacaAccount(**account.__dict__)


def place_order(client: TradingClient, order_details: AlpacaOrderRequest):
    """
    Place an order with flexible order types

    :param client: Alpaca trading client
    :param order_details: Order request details
    :return: Placed AlpacaOrder
    """
    # Map Pydantic model to Alpaca order request based on order type
    if order_details.type == AlpacaOrderType.MARKET:
        order_request = MarketOrderRequest(
            symbol=order_details.symbol,
            qty=order_details.qty,
            side=order_details.side,
            time_in_force=order_details.time_in_force,
        )
    elif order_details.type == AlpacaOrderType.LIMIT:
        if not order_details.limit_price:
            raise ValueError("Limit price is required for limit orders")
        order_request = LimitOrderRequest(
            symbol=order_details.symbol,
            qty=order_details.qty,
            side=order_details.side,
            time_in_force=order_details.time_in_force,
            limit_price=order_details.limit_price,
        )
    elif order_details.type == AlpacaOrderType.STOP:
        if not order_details.stop_price:
            raise ValueError("Stop price is required for stop orders")
        order_request = StopOrderRequest(
            symbol=order_details.symbol,
            qty=order_details.qty,
            side=order_details.side,
            time_in_force=order_details.time_in_force,
            stop_price=order_details.stop_price,
        )
    elif order_details.type == AlpacaOrderType.STOP_LIMIT:
        if not (order_details.stop_price and order_details.limit_price):
            raise ValueError(
                "Both stop and limit prices are required for stop-limit orders"
            )
        order_request = StopLimitOrderRequest(
            symbol=order_details.symbol,
            qty=order_details.qty,
            side=order_details.side,
            time_in_force=order_details.time_in_force,
            stop_price=order_details.stop_price,
            limit_price=order_details.limit_price,
        )
    else:
        raise ValueError(f"Unsupported order type: {order_details.type}")

    # Submit order
    order = client.submit_order(order_request)
    return AlpacaOrder(**order.__dict__)


def get_orders(
    client: TradingClient,
    status: Optional[OrderStatus] = None,
    limit: int = 50,
    after: Optional[datetime] = None,
    until: Optional[datetime] = None,
):
    """
    Retrieve list of orders with optional filtering

    :param client: Alpaca trading client
    :param status: List of order statuses to filter
    :param limit: Maximum number of orders to retrieve
    :param after: Retrieve orders after this timestamp
    :param until: Retrieve orders until this timestamp
    :return: List of AlpacaOrder models
    """

    input = {
        "status": status.value if status else None,
        "limit": limit,
        "after": after,
        "until": until,
    }

    order_request = GetOrdersRequest(**input)
    orders = client.get_orders(order_request)
    return [AlpacaOrder(**order.__dict__) for order in orders]


def get_positions(client: TradingClient):
    """
    Retrieve all open positions

    :param client: Alpaca trading client
    :return: List of AlpacaPosition models
    """
    positions = client.get_all_positions()
    return [AlpacaPosition(**position.__dict__) for position in positions]


def get_position(client: TradingClient, symbol: str):
    """
    Retrieve a specific position by symbol

    :param client: Alpaca trading client
    :param symbol: Stock symbol
    :return: AlpacaPosition model or None
    """
    try:
        position = client.get_open_position(symbol)
        return AlpacaPosition(**position.__dict__)
    except Exception:
        return None


def get_assets(client: TradingClient):
    """
    Retrieve list of assets with optional filtering

    :param client: Alpaca trading client
    :param asset_class: Asset class to filter
    :param exchange: Exchange to filter
    :param status: Status of assets to retrieve
    :return: List of AlpacaAsset models
    """
    assets = client.get_all_assets()
    return [AlpacaAsset(**asset.__dict__) for asset in assets]


def get_asset_by_symbol(client: TradingClient, symbol: str):
    """
    Retrieve asset details by symbol

    :param client: Alpaca trading client
    :param symbol: Stock symbol
    :return: AlpacaAsset model
    """
    asset = client.get_asset(symbol)
    return AlpacaAsset(**asset.__dict__)


def get_latest_quote(
    historical_client: Union[StockHistoricalDataClient, CryptoHistoricalDataClient],
    symbol: str,
):
    """
    Get the latest quote for a given symbol

    :param historical_client: Alpaca historical data client
    :param symbol: Stock symbol
    :return: AlpacaQuote model
    """
    if isinstance(historical_client, CryptoHistoricalDataClient):
        request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = historical_client.get_crypto_latest_quote(request)
    else:
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = historical_client.get_stock_latest_quote(request)

    return AlpacaQuote(**quotes[symbol].__dict__)


def get_historical_bars(
    historical_client: Union[StockHistoricalDataClient, CryptoHistoricalDataClient],
    symbol: str,
    timeframe: TimeFrame = TimeFrame(amount=1, unit=TimeFrameUnit.Day),
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
):
    """
        Get historical price bars for a symbol
    Do make sure you are already clear on what each item is by then
        :param historical_client: Alpaca historical data client
        :param symbol: Stock symbol
        :param timeframe: Time interval for bars
        :param start: Start date for historical data
        :param end: End date for historical data
        :return: List of AlpacaBar models
    """
    if not start:
        start = datetime.now() - timedelta(days=30)
    if not end:
        end = datetime.now()

    if isinstance(historical_client, CryptoHistoricalDataClient):
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end
        )
        bars = historical_client.get_crypto_bars(request)
    else:
        request = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end
        )
        bars = historical_client.get_stock_bars(request)

    return [AlpacaBar(**bar.__dict__) for bar in bars[symbol]]


def get_batched_option_snapshots(
    option_data_client: OptionHistoricalDataClient,
    symbols: List[str],
):
    batch_size = 100
    all_snapshots = {}

    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i : i + batch_size]
        request = OptionSnapshotRequest(symbol_or_symbols=batch_symbols)
        logging.info(f"request: {request}")
        snapshots = option_data_client.get_option_snapshot(request)
        logging.info(f"response: {snapshots}")
        all_snapshots.update(snapshots)
    return all_snapshots


def get_option_contracts(
    trading_client: TradingClient,
    option_data_client: OptionHistoricalDataClient,
    market_data_client: StockHistoricalDataClient,
    request: GetOptionContractsRequest,
    limit: Optional[int] = None,
):
    if not request.underlying_symbols:
        logging.error("No underlying symbols provided in the request.")
        return []

    contracts = []
    next_page_token = None

    logging.info(f"Fetching option contracts for {request.underlying_symbols}...")
    loop_count = 0  # Initialize a loop counter

    while True:
        loop_count += 1  # Increment the counter at the start of each loop

        logging.info(f"request: {request}")
        response = trading_client.get_option_contracts(request)
        logging.info(f"response: {response}")

        if not response or not response.option_contracts:
            break

        contracts.extend(response.option_contracts)
        next_page_token = response.next_page_token

        if not next_page_token:
            break

        if limit and len(contracts) >= limit:
            logging.info(f"Limit of {limit} contracts reached.")
            break

        if loop_count >= 2:  # Add this condition to break after 2 loops
            logging.info("Two loops completed, breaking out.")
            break

    symbols = [c.symbol for c in contracts]
    logging.info(f"Found {len(symbols)} contracts. Fetching snapshots...")

    try:
        snapshots = get_batched_option_snapshots(option_data_client, symbols)
    except Exception as e:
        logging.info(f"Error fetching snapshots: {e}")
        return []

    try:
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)
        stock_bars_request = StockBarsRequest(
            symbol_or_symbols=request.underlying_symbols,
            timeframe=TimeFrame(amount=1, unit=TimeFrameUnit(TimeFrameUnit.Day)),
            start=start_date,
            end=end_date,
            limit=1000,
        )
        bars = market_data_client.get_stock_bars(stock_bars_request)
        if not bars or not isinstance(bars, BarSet):
            raise ValueError("No bars data found or data is not in expected format.")

        bar_data = bars.df
        daily_returns = bar_data["close"].pct_change().dropna()
        hv = daily_returns.std() * (252**0.5)
    except Exception as e:
        logging.info(f"Error calculating historical volatility: {e}")
        hv = None

    rows = []

    for contract in contracts:
        snapshot = snapshots.get(contract.symbol)
        if not snapshot:
            continue

        row_data = {
            "Symbol": contract.symbol,
            # "Underlying": contract.underlying_symbol,
            "Expiration": contract.expiration_date.strftime("%Y-%m-%d"),
            "Strike": contract.strike_price,
            "Type": contract.type.value,
            "Open Interest": contract.open_interest or 0,
            "HV (30D)": hv,
        }

        # Add latest_quote data if available
        if snapshot.latest_quote:
            row_data["Last Price"] = (
                snapshot.latest_quote.ask_price
                if contract.type == ContractType.CALL
                else snapshot.latest_quote.bid_price
            )
            row_data["Bid"] = snapshot.latest_quote.bid_price
            row_data["Ask"] = snapshot.latest_quote.ask_price
        else:
            # If latest_quote is missing, set these to None or a default value
            row_data["Last Price"] = None
            row_data["Bid"] = None
            row_data["Ask"] = None

        # Add greek data if available
        if snapshot.greeks:
            row_data["Delta"] = snapshot.greeks.delta
            row_data["Gamma"] = snapshot.greeks.gamma
            row_data["Rho"] = snapshot.greeks.rho
            row_data["Theta"] = snapshot.greeks.theta
            row_data["Vega"] = snapshot.greeks.vega
        else:
            # If greeks are missing, set these to None or a default value
            row_data["Delta"] = None
            row_data["Gamma"] = None
            row_data["Rho"] = None
            row_data["Theta"] = None
            row_data["Vega"] = None

        # Add implied volatility if available
        row_data["IV"] = (
            snapshot.implied_volatility
            if snapshot.implied_volatility is not None
            else None
        )

        rows.append(row_data)

    return rows


def create_alpaca_watchlist(
    client: TradingClient, name: str, symbols: Optional[List[str]] = None
) -> Optional[str]:  # Changed return type to Optional[str] for error message
    """
    Creates a new watchlist on Alpaca.

    :param client: The Alpaca TradingClient instance.
    :param name: The name of the watchlist.
    :param symbols: An optional list of symbols (e.g., ["AAPL", "GOOG"]) to add initially.
    :return: The ID of the created AlpacaWatchlist object on success, or an error string.
    """
    try:
        watchlist_data = CreateWatchlistRequest(
            name=name, symbols=symbols if symbols else []
        )
        watchlist = client.create_watchlist(watchlist_data=watchlist_data)
        logging.info(f"Watchlist '{name}' created successfully. ID: {watchlist.id}")
        return watchlist.id  # Return the watchlist ID on success
    except APIError as e:
        error_message = f"Error creating watchlist '{name}': {e}"
        logging.error(error_message)
        return error_message
    except Exception as e:
        error_message = (
            f"An unexpected error occurred while creating watchlist '{name}': {e}"
        )
        logging.error(error_message)
        return error_message


def add_symbol_to_alpaca_watchlist(
    client: TradingClient, watchlist_id: str, symbol: str
) -> Optional[str]:  # Changed return type to Optional[str]
    """
    Adds a symbol to an existing Alpaca watchlist.

    :param client: The Alpaca TradingClient instance.
    :param watchlist_id: The ID of the watchlist.
    :param symbol: The symbol to add (e.g., "MSFT").
    :return: The ID of the updated AlpacaWatchlist object on success, or an error string.
    """
    try:
        watchlist = client.add_asset_to_watchlist_by_id(
            watchlist_id=watchlist_id, symbol=symbol
        )
        logging.info(f"Symbol '{symbol}' added to watchlist {watchlist_id}.")
        return watchlist.id  # Return the watchlist ID on success
    except APIError as e:
        error_message = (
            f"Error adding symbol '{symbol}' to watchlist {watchlist_id}: {e}"
        )
        logging.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"An unexpected error occurred while adding symbol '{symbol}' to watchlist {watchlist_id}: {e}"
        logging.error(error_message)
        return error_message


# --- READ Operations ---
def get_all_alpaca_watchlists(
    client: TradingClient,
) -> Optional[
    List[AlpacaWatchlist]
]:  # This function's return type remains a list of objects or None, as per the user's implicit request of only changing error to string, not all of them.
    """
    Retrieves all watchlists for the authenticated Alpaca account.

    :param client: The Alpaca TradingClient instance.
    :return: A list of AlpacaWatchlist objects or None on error.
    """
    try:
        watchlists = client.get_watchlists()
        if watchlists:
            logging.info("--- All Watchlists ---")
            for wl in watchlists:
                logging.info(
                    f"Name: {wl.name}, ID: {wl.id}, Symbols: {[s.symbol for s in wl.assets]}"
                )
            return [AlpacaWatchlist(**wl.__dict__) for wl in watchlists]
        else:
            logging.info("No watchlists found.")
            return []
    except APIError as e:
        logging.error(f"Error retrieving all watchlists: {e}")
        return None
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while retrieving all watchlists: {e}"
        )
        return None


def get_alpaca_watchlist_by_id(
    client: TradingClient, watchlist_id: str
) -> Optional[
    AlpacaWatchlist
]:  # This function's return type remains AlpacaWatchlist object or None.
    """
    Retrieves a specific watchlist by its ID from Alpaca.

    :param client: The Alpaca TradingClient instance.
    :param watchlist_id: The ID of the watchlist.
    :return: The AlpacaWatchlist object or None if not found/error.
    """
    try:
        watchlist = client.get_watchlist_by_id(watchlist_id=watchlist_id)
        if watchlist:
            logging.info(f"--- Watchlist '{watchlist.name}' (ID: {watchlist.id}) ---")
            logging.info(f"Symbols: {[s.symbol for s in watchlist.assets]}")
            return AlpacaWatchlist(**watchlist.__dict__)
        else:
            logging.info(f"Watchlist with ID '{watchlist_id}' not found.")
            return None
    except APIError as e:
        # Alpaca-py raises APIError if watchlist is not found (404)
        if "404" in str(e):
            logging.warning(
                f"Watchlist with ID '{watchlist_id}' not found (API Error 404)."
            )
            return None
        logging.error(f"Error retrieving watchlist by ID '{watchlist_id}': {e}")
        return None
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while retrieving watchlist by ID '{watchlist_id}': {e}"
        )
        return None


def get_alpaca_watchlist_by_name(
    client: TradingClient, name: str
) -> Optional[
    AlpacaWatchlist
]:  # This function's return type remains AlpacaWatchlist object or None.
    """
    Retrieves a specific watchlist by its name (iterates through all watchlists).
    Note: Alpaca API doesn't directly support getting by name, so we iterate.

    :param client: The Alpaca TradingClient instance.
    :param name: The name of the watchlist.
    :return: The AlpacaWatchlist object or None if not found/error.
    """
    watchlists = get_all_alpaca_watchlists(client)
    if watchlists:
        for wl in watchlists:
            if wl.name == name:
                logging.info(f"Found watchlist '{wl.name}' by name. ID: {wl.id}")
                return wl
        logging.info(f"Watchlist with name '{name}' not found.")
    return None


# --- UPDATE Operations ---
def update_alpaca_watchlist_name(
    client: TradingClient, watchlist_id: str, new_name: str
) -> Optional[str]:  # Changed return type to Optional[str]
    """
    Updates the name of an existing Alpaca watchlist.

    :param client: The Alpaca TradingClient instance.
    :param watchlist_id: The ID of the watchlist.
    :param new_name: The new name for the watchlist.
    :return: The ID of the updated AlpacaWatchlist object on success, or an error string.
    """
    try:
        watchlist_data = UpdateWatchlistRequest(name=new_name)
        watchlist = client.update_watchlist_by_id(
            watchlist_id=watchlist_id, watchlist_data=watchlist_data
        )
        logging.info(f"Watchlist {watchlist_id} renamed to '{new_name}'.")
        return watchlist.id  # Return the watchlist ID on success
    except APIError as e:
        error_message = f"Error updating watchlist name for ID '{watchlist_id}': {e}"
        logging.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"An unexpected error occurred while updating watchlist name for ID '{watchlist_id}': {e}"
        logging.error(error_message)
        return error_message


def replace_alpaca_watchlist_symbols(
    client: TradingClient, watchlist_id: str, new_symbols: List[str]
) -> Optional[str]:  # Changed return type to Optional[str]
    """
    Replaces all symbols in an Alpaca watchlist with a new set of symbols.
    Use with caution as this overwrites the entire symbol list.

    :param client: The Alpaca TradingClient instance.
    :param watchlist_id: The ID of the watchlist.
    :param new_symbols: A list of new symbols to replace the existing ones.
    :return: The ID of the updated AlpacaWatchlist object on success, or an error string.
    """
    try:
        watchlist_data = UpdateWatchlistRequest(symbols=new_symbols)
        watchlist = client.update_watchlist_by_id(
            watchlist_id=watchlist_id, watchlist_data=watchlist_data
        )
        logging.info(f"Watchlist {watchlist_id} symbols replaced with {new_symbols}.")
        return watchlist.id  # Return the watchlist ID on success
    except APIError as e:
        error_message = (
            f"Error replacing watchlist symbols for ID '{watchlist_id}': {e}"
        )
        logging.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"An unexpected error occurred while replacing watchlist symbols for ID '{watchlist_id}': {e}"
        logging.error(error_message)
        return error_message


def remove_symbol_from_alpaca_watchlist(
    client: TradingClient, watchlist_id: str, symbol: str
) -> Optional[str]:  # Changed return type to Optional[str]
    """
    Removes a specific symbol from an Alpaca watchlist.

    :param client: The Alpaca TradingClient instance.
    :param watchlist_id: The ID of the watchlist.
    :param symbol: The symbol to remove.
    :return: The ID of the updated AlpacaWatchlist object on success, or an error string.
    """
    try:
        watchlist = client.remove_asset_from_watchlist_by_id(
            watchlist_id=watchlist_id, symbol=symbol
        )
        logging.info(f"Symbol '{symbol}' removed from watchlist {watchlist_id}.")
        return watchlist.id  # Return the watchlist ID on success
    except APIError as e:
        error_message = (
            f"Error removing symbol '{symbol}' from watchlist {watchlist_id}: {e}"
        )
        logging.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"An unexpected error occurred while removing symbol '{symbol}' from watchlist {watchlist_id}: {e}"
        logging.error(error_message)
        return error_message


def delete_alpaca_watchlist(
    client: TradingClient, watchlist_id: str
) -> bool | str:  # Changed return type to bool | str
    """
    Deletes an entire Alpaca watchlist. This is permanent.

    :param client: The Alpaca TradingClient instance.
    :param watchlist_id: The ID of the watchlist to delete.
    :return: True if successful, or an error string otherwise.
    """
    try:
        client.delete_watchlist_by_id(watchlist_id=watchlist_id)
        logging.info(f"Watchlist {watchlist_id} deleted successfully.")
        return True
    except APIError as e:
        error_message = f"Error deleting watchlist {watchlist_id}: {e}"
        logging.error(error_message)
        return error_message
    except Exception as e:
        error_message = (
            f"An unexpected error occurred while deleting watchlist {watchlist_id}: {e}"
        )
        logging.error(error_message)
        return error_message
