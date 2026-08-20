# PySpark / AWS EMR / Glue reconstruction audit

## Source inventory

Reviewed artifacts:

- `DATA_445_pysparkWeek5_tutorial.ipynb`
- `DATA445_pyspark_Lliben_Week5.ipynb`
- `DATA445_EMR_Glue_PySpark_Assignment_3_FINAL ForJupyter(1).ipynb`
- `Data445PysparkAssign3AWSlliben_Workspace_1.ipynb`
- legacy GitHub Pages `projects/EMR_Glue_PySpark.ipynb`

No embedded AWS access keys, secret keys, tokens, S3 URLs, or ARNs were found in the recovered notebook text.

## Week 5 provenance

The first Markdown cell identifies the notebook as a UMGC DATA 445 PySpark tutorial adapted from Syam Kakarla's beginner PySpark article and adapted by UMGC adjunct faculty.

The student's Week 5 notebook has the same 65-cell structure as the tutorial copy. Source differences are minor, including:

- changing the CSV filename to `Data445Week5stocks_price_final.csv`,
- showing 20 rows instead of 10 in one cell,
- whitespace/formatting edits.

It is useful learning evidence but not a strong standalone original portfolio project.

## EMR/Glue source reconciliation

The 4,173-byte `...FINAL ForJupyter(1).ipynb` and the 46,348-byte AWS workspace export contain the same nine code cells. The size difference comes from saved live EMR/Livy outputs in the workspace export.

The legacy GitHub Pages notebook also contains the same nine-cell source without execution output.

## Catalog-name debugging

The workspace first executes:

```python
df = spark.sql("select * from hc_application_csv")
```

and records a Spark `TABLE_OR_VIEW_NOT_FOUND` error.

The next cell changes the source to:

```python
df = spark.sql("select * from hc_applications")
```

and the subsequent schema/count/first/describe operations succeed.

This is a valid debugging step and is preserved as part of the project history.

## Embedded CSV header defect

The successful DataFrame reports:

- total Spark row count: **307,512**;
- numeric `describe()` counts such as `sk_id_curr` and `target`: **307,511**;
- string-column counts such as `name_contract_type` and `code_gender`: **307,512**.

The first row contains values such as:

- `name_contract_type='NAME_CONTRACT_TYPE'`,
- `code_gender='CODE_GENDER'`,
- `occupation_type='OCCUPATION_TYPE'`,
- while numeric fields such as `sk_id_curr` and `target` are null.

This is the CSV header represented as a data record. A successful read therefore masked an ingestion-quality problem.

The reconstruction adds a first-row/column-name header detector and explicit row-count/schema contracts.

## Old portfolio claim correction

The legacy website describes the notebook as a "Data pipeline example using AWS EMR, Glue, and PySpark integration." The notebook demonstrates integration and validation, but it does not implement a full pipeline with extraction orchestration, transformations, writes, partitioning, retries, lineage, or scheduled execution.

The reconstructed title and README use the more precise phrase **AWS Glue + EMR PySpark ingestion validation**.

## Current Spark context

The historical tutorial pins `pyspark==3.0.1`. During reconstruction, Apache Spark 4.2.0 is the current stable Spark line listed by Apache (July 2026). The portfolio keeps PySpark optional and targets the current 4.2 line for new local integrations, while avoiding a claim that the original EMR environment used that version.

## Better production validation sequence

1. Resolve database/table names from an approved catalog inventory.
2. Read through one declared table/source contract.
3. Validate expected key columns and types.
4. Detect accidental header rows and malformed records.
5. Reconcile expected source count versus Spark count.
6. Check null rates and duplicate primary/business keys.
7. Record partition/source metadata.
8. Only then run transformations or analytics.
