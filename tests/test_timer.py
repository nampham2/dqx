from time import sleep

from dqx.timer import Metric, Registry, TimeLimiting, Timer


def test_tle_fast() -> None:
    with TimeLimiting(1) as timer:
        assert 1 + 1 == 2
    assert timer.elapsed_ms() < 10.0


def test_timed() -> None:
    metric = Metric("100ms", Registry())

    @Timer.timed(collector=metric)
    def fn_100ms() -> None:
        sleep(0.15)

    fn_100ms()
    assert metric.value is not None
    assert metric.value > 100
