# Dataset Validation Guide

## Overview

The Dataset Validation feature in DQX ensures data consistency across your verification suites by validating that symbols (metrics) use datasets that are compatible with their parent checks. This prevents configuration errors where symbols reference datasets that aren't available or expected by their containing checks.

## How It Works

### Dataset Hierarchy

DQX follows a hierarchical approach to dataset management:

1. **RootNode** - Receives available datasets from the system
2. **CheckNode** - Can specify which datasets it works with
3. **AssertionNode** - Contains symbols that must use compatible datasets

### Validation Rules

The `DatasetValidator` enforces these rules:

1. **Dataset Mismatch**: If a symbol has a dataset specified, it must be one of the datasets declared by its parent check
2. **Ambiguous Configuration**: If a symbol has no dataset but its parent check has multiple datasets, this is an error (the system cannot determine which dataset to use)
3. **Valid Imputation**: If a symbol has no dataset but its parent check has exactly one dataset, this is valid (the dataset will be imputed)

## Examples

### Example 1: Valid Configuration

```python
@check(name="Price Check", datasets=["production"])
def price_check(mp, ctx):
    # Symbol dataset matches check dataset - VALID
    avg_price = mp.average("price", dataset="production")
    ctx.assert_that(avg_price).where(name="Price is positive").is_positive()
```

### Example 2: Dataset Mismatch

```python
@check(name="Price Check", datasets=["production", "staging"])
def price_check(mp, ctx):
    # Symbol dataset doesn't match any check dataset - ERROR
    avg_price = mp.average("price", dataset="testing")
    ctx.assert_that(avg_price).where(name="Price is positive").is_positive()
```

This will produce an error:
```
Symbol 'average(price)' in assertion 'Price is positive' has dataset 'testing'
which is not in parent check 'Price Check' datasets: ['production', 'staging']
```

### Example 3: Ambiguous Dataset

```python
@check(name="Price Check", datasets=["production", "staging"])
def price_check(mp, ctx):
    # No dataset specified but check has multiple - AMBIGUOUS ERROR
    avg_price = mp.average("price")  # No dataset!
    ctx.assert_that(avg_price).where(name="Price is positive").is_positive()
```

This will produce an error:
```
Symbol 'average(price)' in assertion 'Price is positive' has no dataset specified,
but parent check 'Price Check' has multiple datasets: ['production', 'staging'].
Unable to determine which dataset to use.
```

### Example 4: Valid Imputation

```python
@check(name="Price Check", datasets=["production"])
def price_check(mp, ctx):
    # No dataset specified but check has only one - VALID (will be imputed)
    avg_price = mp.average("price")
    ctx.assert_that(avg_price).where(name="Price is positive").is_positive()
```

### Example 5: No Dataset Validation

```python
@check(name="Price Check")  # No datasets specified
def price_check(mp, ctx):
    # No validation occurs when check doesn't specify datasets
    avg_price = mp.average("price", dataset="any_dataset")
    ctx.assert_that(avg_price).where(name="Price is positive").is_positive()
```

## When Validation Occurs

Dataset validation happens automatically when `suite.build_graph()` or `suite.run()` is called.

If validation fails, a `DQXError` is raised with details about the validation issues.

## Best Practices

1. **Be Explicit**: Always specify datasets on your checks when working with multiple data sources
2. **Match Datasets**: Ensure symbols use datasets that are declared by their parent checks
3. **Single Dataset Simplification**: When a check works with only one dataset, you can omit the dataset parameter on symbols (it will be imputed)
4. **Build Early**: Use `suite.build_graph()` during development to catch configuration errors early

## Integration with Dataset Imputation

Dataset validation works hand-in-hand with dataset imputation:

1. First, validation ensures the configuration is valid
2. Then, imputation fills in missing datasets where unambiguous
3. Finally, analysis runs with the complete, validated dataset configuration

## Error Messages

The validator provides clear error messages to help diagnose issues:

- **Dataset Mismatch**: Shows the symbol, its dataset, and the valid datasets from the parent check
- **Ambiguous Configuration**: Explains why the dataset cannot be determined and lists the available options
- **Multiple Issues**: All validation errors are collected and reported together

## API Reference

### DatasetValidator

The `DatasetValidator` is automatically included when building the graph:

```python
# The validator is used internally by VerificationSuite
suite = VerificationSuite(checks, db, "My Suite")

# Validation happens automatically during build_graph
try:
    suite.build_graph(suite._context, key)
except DQXError as e:
    print(e)  # Shows dataset validation errors
```

### SuiteValidator

The `SuiteValidator` now requires a `MetricProvider` to enable dataset validation:

```python
validator = SuiteValidator()
report = validator.validate(graph, provider)
```

## Troubleshooting

### Common Issues

1. **"Symbol has dataset X which is not in parent check datasets"**
   - Solution: Update the symbol to use one of the check's datasets, or add the dataset to the check

2. **"Unable to determine which dataset to use"**
   - Solution: Specify the dataset explicitly on the symbol, or reduce the check to use only one dataset

3. **"No validation errors but imputation fails"**
   - This shouldn't happen - if you encounter this, please file a bug report

### Debugging Tips

1. Use `suite.build_graph()` to check configuration before running the full suite
2. Look at the node path in error messages to locate the problem
3. Check that datasets are spelled correctly (they're case-sensitive)
4. Ensure parent checks have datasets defined when child symbols use specific datasets
