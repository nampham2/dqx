This is the design document for DQX. Code should follow closely to this design document.

Note: This design document is incomplete.

### Overview

### Dataset specification
Datasets are specified optionally at the assertion level and check level.

In order to run the verification suite, users has to specify datasets in a dictionary: `{"dataset_name" : implementation}`.
The dataset names is then propagated to checks and assertitions and DQX checks the datasets for consistency.
The dataset consistency check runs as follow:
  - A symbol is associated with one and only one dataset.
  - An assertion can be associated with multiple symbols and therefore can have multiple datasets.
  - A check can contain multiple assertions and therefore can have multiple datasets.
  - A symbol can specify a single input datasetname. In case the dataset name is not defined, the dataset name is inferred from the check.
  In this case, the check can only contain a single dataset.
  - An assertion can be specified with a list of datasets, optionally. If it is not specified, the dataset names will be inferred from the parent check.
  The following consistency checks between an assertion and a symbol are performed when the assertion is defined:
    - If a symbol is defined with a dataset and the check is defined with one or more datasets, the symbol's dataset has to be appeared in the check's dataset.
    If the check does not define datasets, infer them from the parent check.
  - The validation suite will not run if ANY of the dataset validation fails.

### Verification Suite
The verification suite is the master object holding all information about a DQX run: context, graph, checks and results.
Users run the verification suite by invoking the `run()` method which does the following in order:
  - Com the dependency graph, this graph is a representation of user's input. It can be stored and retrieved in the future for 
  future evaluation.
  - Impute the datasets, fail if there exists a inconsistency in dataset specification.
  - 

#### How to get pending metrics

Pending metrics are metrics to be evaluated, if a metric has already been computed in the DB we either skip the computation
or recompute and persist to the database.

When do we recompute a metric:
  - Users force recomputation of all metrics.
  - A metric is computed long time ago and needs refresh, a metrics needs a TTL attribute.

For past metrics, for example yesterdays' metrics or last week's metrics, it's not going to be recomputed unless specified by users.

To compute past metrics, the datasource must implement a new cte functions that takes into account the lag.

Metric compute & recompute strategies:
  - Always (default): always recompute metrics, including past metrics.
  - Expired: only recompute expired metrics, including past metrics.
  - New: only compute new metrics.


#### Datasets
Having datasets defined on multiple levels allows check to be performed on different datasets
so they can be shared between validation suites easier.
The datasets can be defined on both the `@check` annotation, and on the `assertions`. Finally,
users have to provide the datasets to the validation suites.

### Output data structure
While the problem is encoded as a graph, the output is stored in a different format suitable
for presentation.
For the users to comprehend the result, it's easier to present it in a tabular format.

suite | check | assertion | metric | result | debug

The result could be Success(value) or Failure(Message)

How to construct the Failure(Message): It's hard to translate the failure messages to a meaningful text.
It's best to collect failure messages in structured format: {"symbol label": "Error message", "nominal_result_key": ["date", "tags"]}
Essentially, it's SymbolicMetric data + an error message.

### Analyzer
Will take care of the metric (re)calculation, preventing the metric from being stale.

Metric rentention time
  - Metric creation time is stored
  - Metric retention time is the current time stamp - stored creation time stamp > threshold

Metric recalculation policy:
  - ALL: recalculate all relevant metrics regardless of the retention
  - STALE: recalculate relevant metrics
  - NONE: do not recalculate metrics

If analyzer fails, the subsequent stages will not be executed. Users will receive an error message
with the failed SQL for troubleshooting.

### Metrics Provider
The provider creates two kinds of metrics:
- Simple metrics: directly corresponds to a metric in the MetricDB
- Extended metrics: metrics based on more than 1 data point in the MetricDB.

Assumptions:
  - Simple metrics won't fail unless there's a DB issue which will raise a RuntimeError.
  - Extended metric can fail with an error. For example dod metric can fails because of divide by zero.


The consequence is that there can be more than 1 symbol failure in the evaluation of a metric.