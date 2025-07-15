import asyncpg
import bcrypt
import os

# Database connection details - Replace with your actual credentials
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://user:password@host:5432/database_name"
)


async def _ensure_table_exists(conn):
    """
    Checks if the user_credentials table exists and creates it if not.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS user_credentials (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL
    );
    """
    try:
        await conn.execute(create_table_sql)
        print("Table 'user_credentials' checked/created successfully.")
    except Exception as e:
        print(f"Error ensuring table exists: {e}")
        raise


async def authenticate_email_user(email: str, password: str) -> bool:
    """
    Authenticates a user based on email and password.

    Args:
        email (str): The user's email address.
        password (str): The plain-text password provided by the user.

    Returns:
        bool: True if authentication is successful, False otherwise.
    """
    conn = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await _ensure_table_exists(conn)

        # Retrieve the hashed password for the given email
        stored_password_hash = await conn.fetchval(
            "SELECT password FROM user_credentials WHERE email = $1", email
        )

        if stored_password_hash:
            # Check if the provided password matches the stored hash
            # bcrypt.checkpw expects bytes for both password and hash
            if bcrypt.checkpw(
                password.encode("utf-8"), stored_password_hash.encode("utf-8")
            ):
                print(f"User '{email}' authenticated successfully.")
                return True
            else:
                print(f"Authentication failed for user '{email}': Incorrect password.")
                return False
        else:
            print(f"Authentication failed for user '{email}': User not found.")
            return False
    except asyncpg.exceptions.PostgresError as e:
        print(f"Database error during authentication: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during authentication: {e}")
        return False
    finally:
        if conn:
            await conn.close()


async def create_user(email: str, password: str) -> bool:
    """
    Creates a new user with the given email and password.
    Hashes the password before storing it.

    Args:
        email (str): The new user's email address.
        password (str): The plain-text password for the new user.

    Returns:
        bool: True if the user was created successfully, False otherwise (e.g., user already exists).
    """
    conn = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await _ensure_table_exists(conn)

        # Check if user already exists
        existing_user_id = await conn.fetchval(
            "SELECT id FROM user_credentials WHERE email = $1", email
        )
        if existing_user_id:
            print(f"User with email '{email}' already exists.")
            return False

        # Hash the password
        # bcrypt.hashpw returns bytes, which asyncpg can store directly as VARCHAR
        hashed_password = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        # Insert the new user
        await conn.execute(
            "INSERT INTO user_credentials (email, password) VALUES ($1, $2)",
            email,
            hashed_password,
        )
        print(f"User '{email}' created successfully.")
        return True
    except asyncpg.exceptions.UniqueViolationError:
        print(
            f"User creation failed: Email '{email}' already exists (unique constraint violation)."
        )
        return False
    except asyncpg.exceptions.PostgresError as e:
        print(f"Database error during user creation: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during user creation: {e}")
        return False
    finally:
        if conn:
            await conn.close()
