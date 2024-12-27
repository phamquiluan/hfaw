# Creating new RCA datasets and Developing new RCA methods

## Creating new RCA datasets
To create new RCA datasets, follow these steps:

1. **System Setup**: Deploy the target microservice system in a controlled environment, such as a Kubernetes cluster, and configure it to generate telemetry data (metrics, logs, traces).

2. **Fault Injection**: Identify the fault types to include (e.g., resource, network, code-level faults). Use tools like `stress-ng` for resource faults, `tc` for network faults, and manual code modifications for code-level faults.

3. **Data Collection**: 
   - **Metrics**: Use tools like Prometheus and cAdvisor to gather system metrics.
   - **Logs**: Employ log aggregators like Fluent Bit or Loki to collect and structure logs.
   - **Traces**: Use tracing tools like Jaeger to capture distributed traces.

4. **Fault Annotation**: Annotate the collected data with labels for the injected faults, including:
   - The time of fault injection.
   - The root cause service.
   - Specific root cause indicators (e.g., a metric, log entry, or trace span).

5. **Data Processing**: Format the telemetry data into a structured format like CSV or JSON. Ensure consistency by including columns for timestamps, service names, and telemetry data values.

6. **Validation**: Engage domain experts to validate the dataset for accuracy and completeness.

7. **Documentation**: Provide a README or similar file with details about the dataset, including:
   - The systems used.
   - Fault types included.
   - Instructions for downloading and using the dataset.

## Developing new RCA methods
To develop new RCA methods and integrate them into RCAEval, follow these steps:

1. **Define the Approach**:
   - Decide on the type of RCA method (metric-based, trace-based, multi-source).
   - Determine the algorithm or technique to use (e.g., statistical analysis, causal inference, machine learning).

2. **Implement the Method**:
   - Create a new Python file in the `RCAEval/e2e/` directory, naming it appropriately (e.g., `new_method.py`).
   - Implement the method as a Python function with the following signature:
     ```python
     def new_method(data, inject_time=None, dataset=None, sli=None, anomalies=None, **kwargs):
         # Method logic here
         return {
             "ranks": ranked_root_causes,
         }
     ```

3. **Preprocess the Data**:
   - Use existing utilities from `RCAEval.io.time_series` to preprocess the input telemetry data, such as `preprocess`, `drop_constant`, or `select_useful_cols`.

4. **Analyze the Data**:
   - Implement the core logic for root cause analysis.
   - Rank the root cause candidates based on their likelihood of causing the failure.

5. **Test the Method**:
   - Write unit tests in `tests/test_new_method.py` to ensure correctness and reproducibility.
   - Use sample datasets available in RCAEval to validate the method.

6. **Integrate with RCAEval**:
   - Add the method to `RCAEval/e2e/__init__.py` for seamless import.
   - Update the `main.py` evaluation script to include the new method by adding it to the `--method` options.

7. **Document the Method**:
   - Provide usage examples in the README or a dedicated tutorial notebook in the `docs/` folder.
   - Include a description of the method, its assumptions, and limitations.

8. **Contribute Back**:
   - Submit a pull request to the RCAEval repository with the new method and associated documentation.
