from dqx import specs


def test_num_rows() -> None:
    nr = specs.NumRows()
    assert isinstance(nr, specs.MetricSpec)
    nr_1 = specs.NumRows()
    assert nr == nr_1
    assert nr.name == "num_rows()"

def test_average() -> None:
    nr = specs.Average("impressions")
    assert isinstance(nr, specs.MetricSpec)
    nr_1 = specs.Average("impressions")
    assert nr == nr_1
    assert nr.name == "average(impressions)"