from time import sleep

import pytest

from dqx.timer import Metric, Registry, TimeLimitExceededError, TimeLimiting, Timer


def test_tle_fast() -> None:
    with TimeLimiting(1) as timer:
        assert 1 + 1 == 2
    assert timer.elapsed_ms() < 1.0


def test_tle_slow() -> None:
    with pytest.raises(TimeLimitExceededError), TimeLimiting(1) as timer:
        sleep(2)

    assert timer.elapsed_ms() > 1.0

    # Making sure the timer is reset
    with TimeLimiting(1):
        sleep(0.1)
        assert 1 + 1 == 2


def test_timed() -> None:
    metric = Metric("100ms", Registry())

    @Timer.timed(collector=metric)
    def fn_100ms() -> None:
        sleep(0.15)

    fn_100ms()
    assert metric.value is not None
    assert metric.value > 100
