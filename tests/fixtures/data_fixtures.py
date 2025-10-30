import datetime
import random
from datetime import timedelta

import duckdb
import pyarrow as pa
import pytest
from faker import Faker


@pytest.fixture(scope="session")
def commerce_data_c1() -> pa.Table:
    Faker.seed(1050)
    n, fake = 1000, Faker()

    name = [fake.name() for _ in range(n)]
    address = [fake.address() for _ in range(n)]
    item = [fake.catch_phrase() for _ in range(n)]
    quantity = [fake.random_int(min=0, max=10_000) for _ in range(n)]
    delivered = [fake.null_boolean() for _ in range(n)]
    price = [fake.pyfloat(min_value=1.0, max_value=10_000) for _ in range(n)]
    tax = [fake.pyfloat(min_value=-100.0, max_value=100.0) for _ in range(n)]
    return pa.Table.from_arrays(
        [name, address, item, quantity, delivered, price, tax],
        names=["name", "address", "item", "quantity", "delivered", "price", "tax"],
    )


@pytest.fixture(scope="session")
def commerce_data_c2() -> pa.Table:
    Faker.seed(2100)
    n, fake = 1000, Faker()

    name = [fake.name() for _ in range(n)]
    address = [fake.address() for _ in range(n)]
    item = [fake.catch_phrase() for _ in range(n)]
    quantity = [fake.random_int(min=0, max=10_000) for _ in range(n)]
    delivered = [fake.null_boolean() for _ in range(n)]
    price = [fake.pyfloat(min_value=1.0, max_value=10_000) for _ in range(n)]
    tax = [fake.pyfloat(min_value=-100.0, max_value=100.0) for _ in range(n)]
    return pa.Table.from_arrays(
        [name, address, item, quantity, delivered, price, tax],
        names=["name", "address", "item", "quantity", "delivered", "price", "tax"],
    )


@pytest.fixture(scope="session")
def commerce_table() -> str:
    return "sales"


class CommercialDataSource:
    """A datasource that generates commercial data and implements SqlDataSource protocol.

    This datasource generates synthetic commerce data with orders distributed
    across the specified date range, and provides date-filtered CTEs for analysis.
    """

    dialect: str = "duckdb"

    def __init__(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        name: str = "commerce",
        records_per_day: int = 30,
        seed: int | None = None,
    ) -> None:
        """Initialize the commercial datasource with generated data.

        Args:
            start_date: First date of data generation
            end_date: Last date of data generation (inclusive)
            name: Name of this dataset
            records_per_day: Average number of records to generate per day
            seed: Random seed for reproducible data
        """
        self._name = name
        self._start_date = start_date
        self._end_date = end_date

        # Generate the data
        arrow_table = self._generate_commerce_data(start_date, end_date, records_per_day, seed)

        # Create DuckDB relation
        self._relation = duckdb.arrow(arrow_table)

        # Create a unique table name for CTE
        # Use object id to ensure uniqueness even with same seed
        self._table_name = f"_commerce_{id(self) % 1000000}"

    @property
    def name(self) -> str:
        """Get the name of this data source."""
        return self._name

    def cte(self, nominal_date: datetime.date) -> str:
        """Return CTE filtering data for specific date.

        Args:
            nominal_date: The date to filter data for

        Returns:
            SQL CTE string that selects only records for the nominal_date
        """
        # Return empty result for out-of-range dates
        if not (self._start_date <= nominal_date <= self._end_date):
            return f"SELECT * FROM {self._table_name} WHERE 1=0"

        date_str = nominal_date.strftime("%Y-%m-%d")
        return f"""
        SELECT *
        FROM {self._table_name}
        WHERE order_date = DATE '{date_str}'
        """

    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        """Execute a query against this data source.

        Args:
            query: The SQL query to execute

        Returns:
            Query results as a DuckDB relation
        """
        return self._relation.query(self._table_name, query)

    def _generate_commerce_data(
        self, start_date: datetime.date, end_date: datetime.date, records_per_day: int, seed: int | None
    ) -> pa.Table:
        """Generate synthetic commerce data with date distribution."""
        if seed is not None:
            random.seed(seed)

        # Calculate total days
        days = (end_date - start_date).days + 1

        # Generate base data
        names = []
        addresses = []
        items = []
        quantities = []
        delivered = []
        prices = []
        taxes = []
        order_dates = []

        # Generate data for each day
        for day_offset in range(days):
            # Create a unique Faker instance for each day to ensure variability
            if seed is not None:
                # Use a combination of seed and day_offset to ensure different data per day
                day_seed = seed + day_offset * 1000
                Faker.seed(day_seed)
                random.seed(day_seed)

            fake = Faker()

            current_date = start_date + timedelta(days=day_offset)
            # Add some randomness to daily record count (+/- 20%)
            daily_records = records_per_day + random.randint(-int(records_per_day * 0.2), int(records_per_day * 0.2))

            # Add a daily variation factor for prices and taxes
            # This ensures each day has different average values
            daily_price_factor = 0.8 + (random.random() * 0.4)  # 0.8 to 1.2
            daily_tax_factor = 0.7 + (random.random() * 0.6)  # 0.7 to 1.3

            for _ in range(daily_records):
                names.append(fake.name())
                addresses.append(fake.address())
                items.append(fake.catch_phrase())
                quantities.append(fake.random_int(min=1, max=100))
                delivered.append(fake.null_boolean())

                # Apply daily variation to prices and taxes
                base_price = fake.pyfloat(min_value=10.0, max_value=1000.0, right_digits=2)
                prices.append(base_price * daily_price_factor)

                base_tax = fake.pyfloat(min_value=0.0, max_value=100.0, right_digits=2)
                taxes.append(base_tax * daily_tax_factor)

                order_dates.append(current_date)

        return pa.Table.from_arrays(
            [
                pa.array(names),
                pa.array(addresses),
                pa.array(items),
                pa.array(quantities),
                pa.array(delivered),
                pa.array(prices),
                pa.array(taxes),
                pa.array(order_dates),
            ],
            names=["name", "address", "item", "quantity", "delivered", "price", "tax", "order_date"],
        )
