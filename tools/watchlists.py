import os
import asyncio
from typing import Optional, List, Dict, Any
import asyncpg
from asyncpg import Pool


class WatchlistsManager:
    """
    A class-based async PostgreSQL manager for handling user watchlists.
    Automatically ensures table exists before any operations.
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the WatchlistsManager.

        Args:
            database_url (str, optional): PostgreSQL connection URL.
                                        If None, uses DATABASE_URL environment variable.
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://user:password@host:5432/database_name"
        )
        self.pool: Optional[Pool] = None
        self._table_created = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """
        Creates and stores the connection pool.
        Automatically ensures the watchlists table exists.
        """
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=10,
                max_queries=50000,
                max_inactive_connection_lifetime=300.0,
                command_timeout=60.0,
                server_settings={"jit": "off"},
            )
            await self._ensure_table_exists()
            print("Connected to database and table verified.")
        except Exception as e:
            print(f"Error creating connection pool: {e}")
            raise

    async def close(self) -> None:
        """Closes the connection pool."""
        if self.pool:
            await self.pool.close()
            print("Connection pool closed.")

    async def _execute_with_retry(self, query: str, *params, retries: int = 3):
        """Execute a query with automatic retry on connection errors."""
        last_exception = None

        for attempt in range(retries + 1):
            try:
                await self._check_connection()
                async with self.pool.acquire() as conn:
                    return await conn.execute(query, *params)
            except (
                asyncpg.exceptions.ConnectionDoesNotExistError,
                asyncpg.exceptions.InterfaceError,
                asyncpg.exceptions.PostgresConnectionError,
            ) as e:
                last_exception = e
                print(f"Connection error (attempt {attempt + 1}/{retries + 1}): {e}")

                if attempt < retries:
                    await self._reconnect()
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    print(f"All retry attempts failed. Last error: {e}")
                    break
            except Exception as e:
                print(f"Non-connection error: {e}")
                raise

    async def _ensure_table_exists(self) -> None:
        """
        Private method to ensure the watchlists table exists.
        Called automatically before any database operations.
        """
        if self._table_created:
            return

        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        async with self.pool.acquire() as conn:
            # First check if table exists and if it has the quantity column
            table_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'watchlists'
                );
                """
            )

            if table_exists:
                # Check if quantity column exists
                qty_column_exists = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'watchlists' AND column_name = 'quantity'
                    );
                    """
                )

                if not qty_column_exists:
                    # Add quantity column to existing table
                    await conn.execute(
                        "ALTER TABLE watchlists ADD COLUMN quantity DECIMAL(15, 6) DEFAULT 0;"
                    )
                    print("Added quantity column to existing watchlists table.")
            else:
                # Create new table with quantity column
                await conn.execute(
                    """
                    CREATE TABLE watchlists (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        stock_symbol VARCHAR(10) NOT NULL,
                        bought_price DECIMAL(10, 2),
                        quantity DECIMAL(15, 6) DEFAULT 0,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT unique_watchlists_item UNIQUE (user_id, stock_symbol)
                    );
                    """
                )
                print("Created new watchlists table with quantity support.")

        self._table_created = True

    async def _fetchval_with_retry(self, query: str, *params, retries: int = 3):
        """Execute a fetchval query with automatic retry on connection errors."""
        last_exception = None

        for attempt in range(retries + 1):
            try:
                await self._check_connection()
                async with self.pool.acquire() as conn:
                    return await conn.fetchval(query, *params)
            except (
                asyncpg.exceptions.ConnectionDoesNotExistError,
                asyncpg.exceptions.InterfaceError,
                asyncpg.exceptions.PostgresConnectionError,
            ) as e:
                last_exception = e
                print(f"Connection error (attempt {attempt + 1}/{retries + 1}): {e}")

                if attempt < retries:
                    await self._reconnect()
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    print(f"All retry attempts failed. Last error: {e}")
                    break
            except Exception as e:
                print(f"Non-connection error: {e}")
                raise

        raise last_exception

    async def _fetch_with_retry(self, query: str, *params, retries: int = 3):
        """Execute a fetch query with automatic retry on connection errors."""
        last_exception = None

        for attempt in range(retries + 1):
            try:
                await self._check_connection()
                async with self.pool.acquire() as conn:
                    return await conn.fetch(query, *params)
            except (
                asyncpg.exceptions.ConnectionDoesNotExistError,
                asyncpg.exceptions.InterfaceError,
                asyncpg.exceptions.PostgresConnectionError,
            ) as e:
                last_exception = e
                print(f"Connection error (attempt {attempt + 1}/{retries + 1}): {e}")

                if attempt < retries:
                    await self._reconnect()
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    print(f"All retry attempts failed. Last error: {e}")
                    break
            except Exception as e:
                print(f"Non-connection error: {e}")
                raise

        raise last_exception
        """
        Private method to ensure the watchlists table exists.
        Called automatically before any database operations.
        """
        if self._table_created:
            return

        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        async with self.pool.acquire() as conn:
            # First check if table exists and if it has the quantity column
            table_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'watchlists'
                );
                """
            )

            if table_exists:
                # Check if quantity column exists
                qty_column_exists = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'watchlists' AND column_name = 'quantity'
                    );
                    """
                )

                if not qty_column_exists:
                    # Add quantity column to existing table
                    await conn.execute(
                        "ALTER TABLE watchlists ADD COLUMN quantity DECIMAL(15, 6) DEFAULT 0;"
                    )
                    print("Added quantity column to existing watchlists table.")
            else:
                # Create new table with quantity column
                await conn.execute(
                    """
                    CREATE TABLE watchlists (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        stock_symbol VARCHAR(10) NOT NULL,
                        bought_price DECIMAL(10, 2),
                        quantity DECIMAL(15, 6) DEFAULT 0,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT unique_watchlists_item UNIQUE (user_id, stock_symbol)
                    );
                    """
                )
                print("Created new watchlists table with quantity support.")

        self._table_created = True

    async def _check_connection(self) -> None:
        """Ensures connection pool is available and table exists."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        # Test the pool connection
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as e:
            print(f"Connection test failed: {e}. Attempting to reconnect...")
            await self._reconnect()

        await self._ensure_table_exists()

    async def _reconnect(self) -> None:
        """Reconnect to the database if connection is lost."""
        try:
            if self.pool:
                await self.pool.close()
            self._table_created = False
            await self.connect()
            print("Successfully reconnected to database.")
        except Exception as e:
            print(f"Failed to reconnect: {e}")
            raise

    async def add_to_watchlists(
        self,
        user_id: str,
        stock_symbol: str,
        bought_price: Optional[float] = None,
        quantity: Optional[float] = None,
    ) -> bool:
        """
        Adds a new stock to a user's watchlists.
        Uses upsert behavior (UPDATE or INSERT).

        Args:
            user_id (str): The unique ID of the user.
            stock_symbol (str): The ticker symbol of the stock.
            bought_price (float, optional): The price at which the stock was bought.
            quantity (float, optional): The quantity of shares bought.

        Returns:
            bool: True if operation was successful, False otherwise.
        """
        try:
            await self._execute_with_retry(
                """
                INSERT INTO watchlists (user_id, stock_symbol, bought_price, quantity)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, stock_symbol) 
                DO UPDATE SET 
                    bought_price = EXCLUDED.bought_price,
                    quantity = EXCLUDED.quantity;
                """,
                user_id,
                stock_symbol.upper(),
                bought_price,
                quantity or 0,
            )
            print(
                f"Successfully added/updated {stock_symbol.upper()} for user {user_id}."
            )
            return True
        except Exception as e:
            print(f"Error adding to watchlists: {e}")
            return False

    async def get_user_watchlists(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all watchlists items for a specific user.

        Args:
            user_id (str): The unique ID of the user.

        Returns:
            List[Dict[str, Any]]: List of watchlists entries as dictionaries.
        """
        try:
            records = await self._fetch_with_retry(
                """
                SELECT id, stock_symbol, bought_price, quantity, created_at
                FROM watchlists
                WHERE user_id = $1
                ORDER BY created_at DESC;
                """,
                user_id,
            )
            return [dict(record) for record in records]
        except Exception as e:
            print(f"Error retrieving watchlists: {e}")
            return []

    async def update_stock_details(
        self,
        user_id: str,
        stock_symbol: str,
        bought_price: Optional[float] = None,
        quantity: Optional[float] = None,
    ) -> bool:
        """
        Updates the bought price and/or quantity for an existing watchlists item.

        Args:
            user_id (str): The unique ID of the user.
            stock_symbol (str): The ticker symbol of the stock to update.
            bought_price (float, optional): The new price to set.
            quantity (float, optional): The new quantity to set.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        if bought_price is None and quantity is None:
            print(
                "No updates provided. Either bought_price or quantity must be specified."
            )
            return False

        try:
            # Build dynamic query based on what's being updated
            updates = []
            params = []
            param_counter = 1

            if bought_price is not None:
                updates.append(f"bought_price = ${param_counter}")
                params.append(bought_price)
                param_counter += 1

            if quantity is not None:
                updates.append(f"quantity = ${param_counter}")
                params.append(quantity)
                param_counter += 1

            # Add user_id and stock_symbol to params
            params.extend([user_id, stock_symbol.upper()])

            query = f"""
                UPDATE watchlists
                SET {', '.join(updates)}
                WHERE user_id = ${param_counter} AND stock_symbol = ${param_counter + 1};
            """

            result = await self._execute_with_retry(query, *params)

            if result.split()[-1] == "1":
                print(f"Successfully updated {stock_symbol.upper()}.")
                return True
            else:
                print(f"Could not update {stock_symbol.upper()}. It may not exist.")
                return False
        except Exception as e:
            print(f"Error updating stock details: {e}")
            return False

    # Keep the old method for backward compatibility
    async def update_bought_price(
        self, user_id: str, stock_symbol: str, new_bought_price: float
    ) -> bool:
        """
        Updates the bought price for an existing watchlists item.
        (Kept for backward compatibility - use update_stock_details for more flexibility)

        Args:
            user_id (str): The unique ID of the user.
            stock_symbol (str): The ticker symbol of the stock to update.
            new_bought_price (float): The new price to set.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        return await self.update_stock_details(
            user_id, stock_symbol, bought_price=new_bought_price
        )

    async def update_quantity(
        self, user_id: str, stock_symbol: str, new_quantity: float
    ) -> bool:
        """
        Updates the quantity for an existing watchlists item.

        Args:
            user_id (str): The unique ID of the user.
            stock_symbol (str): The ticker symbol of the stock to update.
            new_quantity (float): The new quantity to set.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        return await self.update_stock_details(
            user_id, stock_symbol, quantity=new_quantity
        )

    async def add_to_quantity(
        self,
        user_id: str,
        stock_symbol: str,
        additional_quantity: float,
        new_avg_price: Optional[float] = None,
    ) -> bool:
        """
        Adds to existing quantity (useful for accumulating positions).

        Args:
            user_id (str): The unique ID of the user.
            stock_symbol (str): The ticker symbol of the stock.
            additional_quantity (float): Quantity to add to existing position.
            new_avg_price (float, optional): New average price after adding shares.

        Returns:
            bool: True if operation was successful, False otherwise.
        """
        await self._check_connection()

        async with self.pool.acquire() as conn:
            try:
                if new_avg_price is not None:
                    # Update both quantity and average price
                    result = await conn.execute(
                        """
                        UPDATE watchlists
                        SET quantity = quantity + $1, bought_price = $2
                        WHERE user_id = $3 AND stock_symbol = $4;
                        """,
                        additional_quantity,
                        new_avg_price,
                        user_id,
                        stock_symbol.upper(),
                    )
                else:
                    # Just update quantity
                    result = await conn.execute(
                        """
                        UPDATE watchlists
                        SET quantity = quantity + $1
                        WHERE user_id = $2 AND stock_symbol = $3;
                        """,
                        additional_quantity,
                        user_id,
                        stock_symbol.upper(),
                    )

                if result.split()[-1] == "1":
                    print(
                        f"Successfully added {additional_quantity} shares to {stock_symbol.upper()}."
                    )
                    return True
                else:
                    print(f"Could not update {stock_symbol.upper()}. It may not exist.")
                    return False
            except Exception as e:
                print(f"Error adding to quantity: {e}")
                return False

    async def delete_from_watchlists(self, user_id: str, stock_symbol: str) -> bool:
        """
        Removes a stock from a user's watchlists.

        Args:
            user_id (str): The unique ID of the user.
            stock_symbol (str): The ticker symbol of the stock to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        await self._check_connection()

        async with self.pool.acquire() as conn:
            try:
                result = await conn.execute(
                    """
                    DELETE FROM watchlists
                    WHERE user_id = $1 AND stock_symbol = $2;
                    """,
                    user_id,
                    stock_symbol.upper(),
                )
                if result.split()[-1] == "1":
                    print(
                        f"Successfully deleted {stock_symbol.upper()} from user {user_id}'s watchlists."
                    )
                    return True
                else:
                    print(
                        f"{stock_symbol.upper()} was not found on user {user_id}'s watchlists."
                    )
                    return False
            except Exception as e:
                print(f"Error deleting from watchlists: {e}")
                return False

    async def get_watchlists_count(self, user_id: str) -> int:
        """
        Gets the total count of items in a user's watchlists.

        Args:
            user_id (str): The unique ID of the user.

        Returns:
            int: Number of items in the watchlists.
        """
        await self._check_connection()

        async with self.pool.acquire() as conn:
            try:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM watchlists WHERE user_id = $1;",
                    user_id,
                )
                return count or 0
            except Exception as e:
                print(f"Error getting watchlists count: {e}")
                return 0

    async def get_total_portfolio_value(
        self, user_id: str, current_prices: Dict[str, float]
    ) -> float:
        """
        Calculates total portfolio value based on current prices.

        Args:
            user_id (str): The unique ID of the user.
            current_prices (Dict[str, float]): Dictionary mapping stock symbols to current prices.

        Returns:
            float: Total portfolio value.
        """
        watchlists = await self.get_user_watchlists(user_id)
        total_value = 0.0

        for item in watchlists:
            symbol = item["stock_symbol"]
            quantity = float(item["quantity"] or 0)
            current_price = current_prices.get(symbol, 0.0)
            total_value += quantity * current_price

        return total_value

    async def stock_exists_in_watchlists(self, user_id: str, stock_symbol: str) -> bool:
        """
        Checks if a stock already exists in a user's watchlists.

        Args:
            user_id (str): The unique ID of the user.
            stock_symbol (str): The ticker symbol to check.

        Returns:
            bool: True if stock exists in watchlists, False otherwise.
        """
        try:
            exists = await self._fetchval_with_retry(
                "SELECT EXISTS(SELECT 1 FROM watchlists WHERE user_id = $1 AND stock_symbol = $2);",
                user_id,
                stock_symbol.upper(),
            )
            return exists or False
        except Exception as e:
            print(f"Error checking stock existence: {e}")
            return False


# ==============================================================================
# UTILITY WRAPPER FUNCTIONS (Chainlit Integration Ready)
# ==============================================================================


async def add_to_watchlists_util(
    stock_symbol: str,
    bought_price: Optional[float] = None,
    quantity: Optional[float] = None,
) -> bool:
    """
    Utility function to add a stock to user's watchlists with automatic connection management.

    Args:
        stock_symbol (str): The ticker symbol of the stock.
        bought_price (float, optional): The price at which the stock was bought.
        quantity (float, optional): The quantity of shares bought.

    Returns:
        bool: True if operation was successful, False otherwise.
    """
    manager = WatchlistsManager()

    # Import chainlit here to avoid dependency issues if not using chainlit
    try:
        import chainlit as cl

        user = cl.user_session.get("user")
        assert isinstance(user, cl.User)
        user_id = user.identifier
    except (ImportError, AssertionError, AttributeError):
        # Fallback for non-chainlit usage - you can modify this
        user_id = "default_user"

    await manager.connect()
    try:
        result = await manager.add_to_watchlists(
            user_id, stock_symbol, bought_price, quantity
        )
        return result
    finally:
        await manager.close()


async def get_user_watchlists_util() -> List[Dict[str, Any]]:
    """
    Utility function to get user's watchlists with automatic connection management.

    Returns:
        List[Dict[str, Any]]: List of watchlists entries as dictionaries.
    """
    manager = WatchlistsManager()

    try:
        import chainlit as cl

        user = cl.user_session.get("user")
        assert isinstance(user, cl.User)
        user_id = user.identifier
    except (ImportError, AssertionError, AttributeError):
        user_id = "default_user"

    await manager.connect()
    try:
        result = await manager.get_user_watchlists(user_id)
        return result
    finally:
        await manager.close()


async def update_stock_details_util(
    stock_symbol: str,
    bought_price: Optional[float] = None,
    quantity: Optional[float] = None,
) -> bool:
    """
    Utility function to update stock details with automatic connection management.

    Args:
        stock_symbol (str): The ticker symbol of the stock to update.
        bought_price (float, optional): The new price to set.
        quantity (float, optional): The new quantity to set.

    Returns:
        bool: True if update was successful, False otherwise.
    """
    manager = WatchlistsManager()

    try:
        import chainlit as cl

        user = cl.user_session.get("user")
        assert isinstance(user, cl.User)
        user_id = user.identifier
    except (ImportError, AssertionError, AttributeError):
        user_id = "default_user"

    await manager.connect()
    try:
        result = await manager.update_stock_details(
            user_id, stock_symbol, bought_price, quantity
        )
        return result
    finally:
        await manager.close()


# Keep old utility for backward compatibility
async def update_bought_price_util(stock_symbol: str, new_bought_price: float) -> bool:
    """
    Utility function to update bought price with automatic connection management.
    (Kept for backward compatibility)

    Args:
        stock_symbol (str): The ticker symbol of the stock to update.
        new_bought_price (float): The new price to set.

    Returns:
        bool: True if update was successful, False otherwise.
    """
    return await update_stock_details_util(stock_symbol, bought_price=new_bought_price)


async def update_quantity_util(stock_symbol: str, new_quantity: float) -> bool:
    """
    Utility function to update quantity with automatic connection management.

    Args:
        stock_symbol (str): The ticker symbol of the stock to update.
        new_quantity (float): The new quantity to set.

    Returns:
        bool: True if update was successful, False otherwise.
    """
    return await update_stock_details_util(stock_symbol, quantity=new_quantity)


async def add_to_quantity_util(
    stock_symbol: str, additional_quantity: float, new_avg_price: Optional[float] = None
) -> bool:
    """
    Utility function to add to existing quantity with automatic connection management.

    Args:
        stock_symbol (str): The ticker symbol of the stock.
        additional_quantity (float): Quantity to add to existing position.
        new_avg_price (float, optional): New average price after adding shares.

    Returns:
        bool: True if operation was successful, False otherwise.
    """
    manager = WatchlistsManager()

    try:
        import chainlit as cl

        user = cl.user_session.get("user")
        assert isinstance(user, cl.User)
        user_id = user.identifier
    except (ImportError, AssertionError, AttributeError):
        user_id = "default_user"

    await manager.connect()
    try:
        result = await manager.add_to_quantity(
            user_id, stock_symbol, additional_quantity, new_avg_price
        )
        return result
    finally:
        await manager.close()


async def delete_from_watchlists_util(stock_symbol: str) -> bool:
    """
    Utility function to delete from watchlists with automatic connection management.

    Args:
        stock_symbol (str): The ticker symbol of the stock to delete.

    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    manager = WatchlistsManager()

    try:
        import chainlit as cl

        user = cl.user_session.get("user")
        assert isinstance(user, cl.User)
        user_id = user.identifier
    except (ImportError, AssertionError, AttributeError):
        user_id = "default_user"

    await manager.connect()
    try:
        result = await manager.delete_from_watchlists(user_id, stock_symbol)
        return result
    finally:
        await manager.close()


async def get_watchlists_count_util() -> int:
    """
    Utility function to get watchlists count with automatic connection management.

    Returns:
        int: Number of items in the watchlists.
    """
    manager = WatchlistsManager()

    try:
        import chainlit as cl

        user = cl.user_session.get("user")
        assert isinstance(user, cl.User)
        user_id = user.identifier
    except (ImportError, AssertionError, AttributeError):
        user_id = "default_user"

    await manager.connect()
    try:
        result = await manager.get_watchlists_count(user_id)
        return result
    finally:
        await manager.close()


async def get_total_portfolio_value_util(current_prices: Dict[str, float]) -> float:
    """
    Utility function to get total portfolio value with automatic connection management.

    Args:
        current_prices (Dict[str, float]): Dictionary mapping stock symbols to current prices.

    Returns:
        float: Total portfolio value.
    """
    manager = WatchlistsManager()

    try:
        import chainlit as cl

        user = cl.user_session.get("user")
        assert isinstance(user, cl.User)
        user_id = user.identifier
    except (ImportError, AssertionError, AttributeError):
        user_id = "default_user"

    await manager.connect()
    try:
        result = await manager.get_total_portfolio_value(user_id, current_prices)
        return result
    finally:
        await manager.close()


async def stock_exists_in_watchlists_util(stock_symbol: str) -> bool:
    """
    Utility function to check if stock exists in watchlists with automatic connection management.

    Args:
        stock_symbol (str): The ticker symbol to check.

    Returns:
        bool: True if stock exists in watchlists, False otherwise.
    """
    manager = WatchlistsManager()

    try:
        import chainlit as cl

        user = cl.user_session.get("user")
        assert isinstance(user, cl.User)
        user_id = user.identifier
    except (ImportError, AssertionError, AttributeError):
        user_id = "default_user"

    await manager.connect()
    try:
        result = await manager.stock_exists_in_watchlists(user_id, stock_symbol)
        return result
    finally:
        await manager.close()


# Alternative: Context Manager Wrapper Functions (More Efficient for Multiple Operations)
async def with_watchlists_manager(func):
    """
    Decorator/wrapper for functions that need a watchlists manager.
    More efficient when you need multiple operations.

    Usage:
        @with_watchlists_manager
        async def my_function(manager, user_id):
            await manager.add_to_watchlists(user_id, "AAPL", 150.00, 10.0)
            return await manager.get_user_watchlists(user_id)
    """

    async def wrapper(*args, **kwargs):
        async with WatchlistsManager() as manager:
            try:
                import chainlit as cl

                user = cl.user_session.get("user")
                assert isinstance(user, cl.User)
                user_id = user.identifier
            except (ImportError, AssertionError, AttributeError):
                user_id = "default_user"

            return await func(manager, user_id, *args, **kwargs)

    return wrapper


watchlists_tools = [
    add_to_watchlists_util,
    get_user_watchlists_util,
    update_bought_price_util,
    delete_from_watchlists_util,
    get_watchlists_count_util,
    stock_exists_in_watchlists_util,
]
