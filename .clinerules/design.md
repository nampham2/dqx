This is the design document for DQX. Code should follow closely to this design document.

Note: This design document is incomplete.

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
