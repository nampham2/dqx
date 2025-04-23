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
