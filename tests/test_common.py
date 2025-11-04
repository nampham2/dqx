import datetime as dt

from dqx.common import ResultKey


def test_result_key() -> None:
    tags = {"partner": "baguette", "group_name": "french", "group_size": "small"}
    yyyy_mm_dd = dt.date.fromisoformat("2025-02-09")
    key = ResultKey(yyyy_mm_dd=yyyy_mm_dd, tags=tags)
    assert key.yyyy_mm_dd == yyyy_mm_dd
    assert key.tags == tags

    new_key = ResultKey(yyyy_mm_dd=yyyy_mm_dd, tags=tags)
    assert key == new_key

    different_key = ResultKey(yyyy_mm_dd=yyyy_mm_dd, tags={})
    assert key != different_key

    assert hash(key) == hash(new_key)
    assert hash(key) != hash(different_key)
