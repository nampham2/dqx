"""Demo of Rich logging capabilities in DQX."""

import logging
import time

from dqx import get_logger


def main() -> None:
    """Demonstrate Rich logging features."""
    # Get logger
    logger = get_logger("dqx.demo", level=logging.DEBUG)

    print("=== DQX Rich Logging Demo ===\n")

    # 1. Basic log levels
    print("1. Basic Log Levels:")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    time.sleep(0.5)
    print("\n2. Rich Markup Support:")

    # 2. Rich markup examples
    logger.info("[bold cyan]Bold cyan text[/bold cyan] in log message")
    logger.warning("Contains [yellow]highlighted[/yellow] warning text")
    logger.error("[red]Error[/red] with [bold]emphasis[/bold]")

    time.sleep(0.5)
    print("\n3. Structured Information:")

    # 3. Structured logging
    logger.info("Processing data: [green]✓[/green] Stage 1 complete")
    logger.info("Processing data: [green]✓[/green] Stage 2 complete")
    logger.info("Processing data: [red]✗[/red] Stage 3 failed")

    time.sleep(0.5)
    print("\n4. Exception with Rich Traceback:")

    # 4. Exception logging with rich traceback
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("Failed to perform calculation")

    time.sleep(0.5)
    print("\n5. Long Messages:")

    # 5. Long message handling
    long_message = (
        "This is a very long log message that demonstrates how Rich handles "
        "text wrapping in the console. It should wrap nicely at the terminal "
        "width while maintaining the log level indicator and timestamp alignment."
    )
    logger.info(long_message)

    time.sleep(0.5)
    print("\n6. Data Validation Example:")

    # 6. DQX-specific example
    logger.info("[bold]Starting data quality check[/bold]")
    logger.info("Analyzing dataset: [cyan]sales_data[/cyan]")
    logger.info("  • Checking metric: [green]total_revenue[/green]")
    logger.info("  • Assertion: total_revenue > 0")
    logger.warning("  • Result: [yellow]FAILURE[/yellow] - total_revenue = -100")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
