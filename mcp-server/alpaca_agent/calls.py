import logging
from typing import List, Optional, Union
from datetime import datetime, timedelta
from alpaca.trading import OrderStatus
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
            timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day),
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


if __name__ == "__main__":
    from alpaca_client import AlpacaClient

    alpaca_client = AlpacaClient()
    trading_client = alpaca_client.trading_client()

    logging.info(get_orders(trading_client))
