from typing import Optional, Union
from datetime import datetime, timedelta
from alpaca.trading import OptionContractsResponse
from alpaca.trading.requests import GetOptionContractsRequest

from alpaca.trading.client import TradingClient
from alpaca.data import (
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
    AlpacaOrderStatus,
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
    status: Optional[AlpacaOrderStatus] = None,
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


def get_option_contracts(
    trading_client: TradingClient,
    option_data_client: OptionHistoricalDataClient,
    market_data_client: StockHistoricalDataClient,
    request: GetOptionContractsRequest,
):
    """
    Retrieve list of option contracts with optional filtering

    :param client: Alpaca trading client
    :param request: GetOptionContractsRequest with filtering options
    :return: List of AlpacaAsset models representing option contracts
    """

    contracts = []
    next_page_token = None
    loop_count = 0

    while True:
        if loop_count >= 3:
            break

        if next_page_token:
            request.page_token = next_page_token

        response = trading_client.get_option_contracts(request)

        if (
            not isinstance(response, OptionContractsResponse)
            or not response
            or not response.option_contracts
        ):
            continue

        contracts.extend(response.option_contracts)

        next_page_token = response.next_page_token

        if not next_page_token:
            break

        loop_count += 1

    symbols = [contract.symbol for contract in contracts]
    print(f"Found {len(symbols)} contracts. Fetching snapshots...")

    # 2. Get snapshots for the contracts to fetch real-time data
    try:
        snapshots = option_data_client.get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=request.underlying_symbols)  # type: ignore
        )
    except Exception as e:
        print(f"Error fetching option snapshots: {e}")
        return pd.DataFrame().to_dict(orient="records")

    if not snapshots:
        print("Could not fetch snapshots for the found contracts.")
        return pd.DataFrame().to_dict(orient="records")

    # 3. Get Historical Volatility for the underlying stock
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        bars_request = StockBarsRequest(
            symbol_or_symbols=request.underlying_symbols[0],  # type: ignore
            timeframe=TimeFrame.Day,  # type: ignore
            start=start_date,
            end=end_date,
        )
        bars = market_data_client.get_stock_bars(bars_request)
        daily_returns = bars[request.underlying_symbols[0]].df["close"].pct_change().dropna()  # type: ignore
        historical_volatility = daily_returns.std() * (252**0.5)  # Annualized
    except Exception as e:
        print(f"Could not calculate Historical Volatility: {e}")
        historical_volatility = None

    # 4. Combine the data
    options_data = []
    for symbol, snapshot in snapshots.items():
        contract_details = next((c for c in contracts if c.symbol == symbol), None)
        if not contract_details or not snapshot.latest_quote or not snapshot.greeks:
            continue

        last_price = (
            snapshot.latest_quote.ask_price
            if request.type.value == "call"  # type: ignore
            else snapshot.latest_quote.bid_price
        )
        # change = last_price - snapshot.daily_bar.close
        # percent_change = (
        #     (change / snapshot.daily_bar.close) * 100
        #     if snapshot.daily_bar.close != 0
        #     else 0
        # )

        options_data.append(
            {
                "Underlying": contract_details.underlying_symbol,
                "Expiration": contract_details.expiration_date.strftime("%Y-%m-%d"),
                "Strike": contract_details.strike_price,
                "Type": contract_details.type.value,
                "Last Price": last_price,
                # "Change": change,
                # "% Change": f"{percent_change:.2f}%",
                "Bid": snapshot.latest_quote.bid_price,
                "Ask": snapshot.latest_quote.ask_price,
                # "Volume": snapshot.daily_bar.volume,
                "Open Interest": contract_details.open_interest or 0,
                "Delta": snapshot.greeks.delta,
                "Gamma": snapshot.greeks.gamma,
                "Rho": snapshot.greeks.rho,
                "Theta": snapshot.greeks.theta,
                "Vega": snapshot.greeks.vega,
                "IV": snapshot.implied_volatility,
                "HV (30D)": historical_volatility,
            }
        )

    return pd.DataFrame(options_data).to_dict(orient="records")


if __name__ == "__main__":
    from alpaca_client import AlpacaClient

    alpaca_client = AlpacaClient()
    trading_client = alpaca_client.trading_client()

    print(get_orders(trading_client))
