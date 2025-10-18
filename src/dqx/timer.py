import functools
import signal
from collections import UserDict
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any, Protocol, Self, runtime_checkable


class TimeLimitExceededError(Exception): ...


def _tle_signal_handler(*_args: object) -> None:
    raise TimeLimitExceededError("Error: TLE")


class TimeLimiting:
    def __init__(self, time_limit_seconds: int | None) -> None:
        self._time_limit = time_limit_seconds

    def __enter__(self) -> Self:
        self.tick = perf_counter()
        if self._time_limit:
            signal.signal(signal.SIGALRM, _tle_signal_handler)
            signal.alarm(self._time_limit)
        return self

    def __exit__(self, *args: object) -> None:
        if self._time_limit:
            signal.alarm(0)
        self.tock = perf_counter()

    def elapsed_ms(self) -> float:
        return (self.tock - self.tick) * 1000


@runtime_checkable
class MetricRegistry(Protocol):
    def collect(self, value: float) -> None: ...


class Timer:
    def __init__(self, registry: MetricRegistry) -> None:
        self.collector = registry

    def __enter__(self) -> Self:
        self.tick = perf_counter()
        self.tock: float | None = None
        return self

    def __exit__(self, *args: object) -> None:
        self.tock = perf_counter()
        self.collector.collect(self.elapsed_ms())

    def elapsed_ms(self) -> float:
        if self.tock is None:
            raise RuntimeError("Timer has not been stopped yet")
        return (self.tock - self.tick) * 1000

    @staticmethod
    def timed(*, collector: MetricRegistry, **kwargs: dict) -> Callable:
        def _do_timing(*args: Sequence, fn: Callable) -> Any:
            tick = perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                tock = perf_counter()
                collector.collect((tock - tick) * 1000)

        def decorator(fn: Callable) -> Callable:
            return functools.wraps(fn)(functools.partial(_do_timing, fn=fn))

        return decorator


class Metric:
    def __init__(self, name: str, collector: UserDict[str, float]) -> None:
        self.name = name
        self.collector = collector

    def collect(self, elapsed: float) -> None:
        self.collector[self.name] = elapsed

    @property
    def value(self) -> float | None:
        return self.collector.get(self.name, None)


class Registry(UserDict[str, float]):
    def __init__(self) -> None:
        super().__init__({})

    def timer(self, name: str) -> Timer:
        return Timer(Metric(name, self))
