# Pipeline

> *Note:* This is a `README.md` boilerplate generated using `Kedro 0.18.6`.

## Overview

This pipeline:
1. Loads data from a csv file (`_load_data` function).
2. Drops unnecessary columns and renames the rest (`_drop_columns` function).
3. Changes the data type of a column (`_change_dtypes` function).
4. Filters the data and aggregates it (`_prepare_data` function).
5. Calculates the price elasticity for each product (`_calculate_price_elasticity` function).
6. Simulates the impact of price changes on demand and revenue (`simulate_elasticity` function).
7. Generates a report based on the simulation results (`make_simulation_report` function).
8. Displays the results in a Streamlit app (`run_simulation_tab` function).

## Pipeline inputs

### `df_ready.csv`

|      |                    |
| ---- | ------------------ |
| Type | `.csv file` |
| Description | Data containing product information |

## Pipeline intermediate outputs

No explicit intermediate outputs are saved to disk.

## Pipeline outputs

### `Streamlit App`

|      |                    |
| ---- | ------------------ |
| Type | `Streamlit App` |
| Description | An application that allows the user to interactively view the results of price elasticity calculations and price change simulations |
