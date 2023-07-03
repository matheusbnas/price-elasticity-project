"""
This is a boilerplate pipeline
generated using Kedro 0.18.6
"""
from typing import Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from price_elasticity import make_price_elasticity

st.set_page_config(layout="wide")


def _load_data() -> pd.DataFrame:
    df_raw = pd.read_csv("../../data/01_raw/df_ready.csv")
    return df_raw


def _drop_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        "Unnamed: 0",
        "Cluster",
        "condition",
        "Disc_percentage",
        "isSale",
        "Imp_count",
        "p_description",
        "currency",
        "dateAdded",
        "dateSeen",
        "dateUpdated",
        "imageURLs",
        "shipping",
        "sourceURLs",
        "weight",
        "Date_imp_d.1",
        "Zscore_1",
        "price_std",
    ]

    df_raw = df_raw.drop(columns=columns_to_drop)
    df_raw.columns = [
        "date_imp",
        "date_imp_d",
        "category_name",
        "name",
        "price",
        "disc_price",
        "merchant",
        "brand",
        "manufacturer",
        "day_n",
        "month",
        "month_n",
        "day",
        "week_number",
    ]
    return df_raw


def _change_dtypes(df1: pd.DataFrame) -> pd.DataFrame:
    df1["date_imp_d"] = pd.to_datetime(df1["date_imp_d"])
    return df1


def _prepare_data(df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_best = df2.loc[
        (df2["category_name"] == "laptop, computer")
        & (df2["merchant"] == "Bestbuy.com")
    ]
    df_agg = (
        df_best.groupby(["name", "week_number"])
        .agg({"disc_price": "mean", "date_imp": "count"})
        .reset_index()
    )

    x_price = df_agg.pivot(index="week_number", columns="name", values="disc_price")

    y_demand = df_agg.pivot(index="week_number", columns="name", values="date_imp")

    median_price = np.round(x_price.median(), 2)
    x_price.fillna(median_price, inplace=True)
    y_demand.fillna(0, inplace=True)
    return x_price, y_demand


def _calculate_price_elasticity(
    x_price: pd.DataFrame, y_demand: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate price elasticity for each product.

    Parameters
    ----------
    x_price : pd.DataFrame
        The DataFrame of product prices.
    y_demand : pd.DataFrame
        The DataFrame of product demands.

    Returns
    -------
    pd.DataFrame
        The DataFrame with price elasticity results for each product.
    """
    price_elasticity_laptop = {
        "name": [],
        "price_elastity": [],
        "price_mean": [],
        "quantity_mean": [],
        "intercept": [],
        "slope": [],
        "rsquared": [],
        "p_value": [],
    }

    for column in x_price.columns:
        column_points = [
            {
                "x_price": x_price.reset_index(drop=True)[column][i],
                "y_demand": y_demand.reset_index(drop=True)[column][i],
            }
            for i in range(len(x_price[column]))
        ]

        list_price = [point["x_price"] for point in column_points]
        list_demand = [point["y_demand"] for point in column_points]

        X = sm.add_constant(list_price)
        model = sm.OLS(list_demand, X)
        results = model.fit()

        if results.f_pvalue < 0.05:
            intercept, slope = results.params

            price_elasticity_laptop["name"].append(column)
            price_elasticity_laptop["price_elastity"].append(
                slope * (np.mean(list_price) / np.mean(list_demand))
            )
            price_elasticity_laptop["rsquared"].append(results.rsquared)
            price_elasticity_laptop["p_value"].append(results.f_pvalue)
            price_elasticity_laptop["intercept"].append(intercept)
            price_elasticity_laptop["slope"].append(slope)
            price_elasticity_laptop["price_mean"].append(
                np.round(np.mean(list_price), 2)
            )
            price_elasticity_laptop["quantity_mean"].append(
                np.round(np.mean(list_demand), 2)
            )

    return pd.DataFrame(price_elasticity_laptop)


def simulate_elasticity(
    percentual: float, y_demand: pd.DataFrame, df_elasticity: pd.DataFrame, option: str
) -> Union[pd.DataFrame, None]:
    """
    Simulate price elasticity with a given percentage change.

    Parameters
    ----------
    percentual : float
        The percentage of price change.
    y_demand : pd.DataFrame
        The DataFrame of product demands.
    df_elasticity : pd.DataFrame
        The DataFrame of price elasticity.
    option : str
        The option of price change, can be either "Desconto" or "Aumento de Preço".

    Returns
    -------
    Union[pd.DataFrame, None]
        The DataFrame of simulated results. Returns None if the percentual is 0.
    """
    if percentual == 0:
        simulate_result = None
    else:
        result_revenue = {
            "name": [],
            "faturamento_atual": [],
            "faturamento_novo": [],
            "variacao_faturamento": [],
            "variacao_percentual": [],
        }

        if option == "Desconto":
            percentual = -percentual

        for i in range(len(df_elasticity)):
            current_price_mean = df_elasticity["price_mean"][i]
            current_demand = y_demand[df_elasticity["name"][i]].sum()

            if percentual < 0:
                price_change = current_price_mean * (1 - ((percentual * (-1)) / 100))
            else:
                price_change = (current_price_mean * percentual) + current_price_mean

            demand_increase = (percentual / 100) * df_elasticity["price_elastity"][i]
            new_demand = demand_increase * current_demand

            current_revenue = round(current_price_mean * current_demand, 2)
            new_revenue = round(price_change * new_demand, 2)

            revenue_variation = round(new_revenue - current_revenue, 2)
            percentage_variation = round(
                (new_revenue - current_revenue) / current_revenue, 2
            )

            result_revenue["name"].append(df_elasticity["name"][i])
            result_revenue["faturamento_atual"].append(current_revenue)
            result_revenue["faturamento_novo"].append(new_revenue)
            result_revenue["variacao_faturamento"].append(revenue_variation)
            result_revenue["variacao_percentual"].append(percentage_variation)

        simulate_result = pd.DataFrame(result_revenue)

    return simulate_result


def _generate_product_report(
    final: pd.DataFrame, op: str, number: int, intro_relatorio: str
) -> str:
    produtos = []

    for i in range(len(final)):
        produto = final["name"][i]
        faturamento_atual = final["faturamento_atual"][i]
        faturamento_novo = final["faturamento_novo"][i]
        acao = "Aumento" if op == "Aumento de Preço" else "Diminuição"
        acao2 = "Aumento" if faturamento_novo > faturamento_atual else "Diminuição"

        produto_limitado = produto if len(produto) <= 50 else produto[:50]

        produtos.append(produto_limitado)
        relatorio_produto = (
            f"- {acao} {number}% no produto {produto_limitado}:"
            f" {acao2} do faturamento em R${abs(faturamento_novo)}\n"
        )
        intro_relatorio += relatorio_produto
    total_produtos_analisados = len(produtos)

    return total_produtos_analisados, intro_relatorio


def _generate_general_report(
    final: pd.DataFrame, total_produtos_analisados: int, number: int
) -> str:
    soma_faturamento_novo = final["faturamento_novo"].sum()
    soma_faturamento_atual = final["faturamento_atual"].sum()
    variacao_faturamento = soma_faturamento_novo - soma_faturamento_atual

    impacto = "AUMENTA" if variacao_faturamento > 0 else "DIMINUI"
    relatorio_geral = (
        "\n## **Impacto no faturamento e na demanda no negócio como um todo:**\n"
    )
    relatorio_geral += f"- Total de produtos analisados: {total_produtos_analisados}\n"
    relatorio_geral += (
        f"- Com um desconto de {number}% o faturamento do seu negócio "
        f"{impacto}, podendo fazer com que o faturamento potencial do "
        f"seu negócio possa atingir {round(soma_faturamento_novo,2)}. "
        f"Isso representa um valor de {round(abs(variacao_faturamento),2)}"
        f"{'a mais' if impacto == 'AUMENTA' else 'a menos'} do que você "
        f"fatura atualmente.\n"
    )
    relatorio_geral += (
        f"- Variação percentual no faturamento: {final['variacao_percentual'].sum()}%\n"
    )
    return relatorio_geral


def make_simulation_report(final: pd.DataFrame, op: str, number: int) -> str:
    """
    Generate a simulation report.

    Parameters
    ----------
    final : pd.DataFrame
        The DataFrame of simulation results.
    op : str
        The option of price change.
    number : int
        The percentage of price change.

    Returns
    -------
    str
        The simulation report.
    """
    intro_relatorio = (
        "### **Nosso modelo de Inteligência Artificial gerou um "
        "relatório personalizado simulando os efeitos que essa alteração "
        "de preço pode causar na Demanda e Faturamento:**\n\n"
    )
    total_produtos_analisados, intro_relatorio = _generate_product_report(
        final, op, number, intro_relatorio
    )

    relatorio_geral = _generate_general_report(final, total_produtos_analisados, number)

    relatorio_final = intro_relatorio + relatorio_geral
    return relatorio_final


def prepare_data_and_calculate_elasticity() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_raw = _load_data()
    df1 = _drop_columns(df_raw)

    df2 = _change_dtypes(df1)

    x_price, y_demand = _prepare_data(df2)

    df_elasticity = _calculate_price_elasticity(x_price, y_demand)

    return df_elasticity, y_demand


def run_simulation_tab(df_elasticity: pd.DataFrame, y_demand: pd.DataFrame) -> None:
    """
    Run simulation and display results on Streamlit.

    Parameters
    ----------
    df_elasticity : pd.DataFrame
        The DataFrame of price elasticity.
    y_demand : pd.DataFrame
        The DataFrame of product demands.
    """
    col1, col2 = st.columns((1, 1))
    with col1:
        st.markdown(
            (
                "<h2 style='text-align: center;'>Você gostaria de aplicar um "
                "desconto ou um aumento de preço nos produtos?</h2>"
            ),
            unsafe_allow_html=True,
        )
        option = st.selectbox("", ("Aumento de Preço", "Aplicar Desconto"))
        if option == "Aumento de Preço":
            op = "Aumento de Preço"
        else:
            op = "Desconto"

    with col2:
        st.markdown(
            '<h2 style="text-align: center;">Qual o percentual de '
            + op
            + " que você deseja aplicar?</h2>",
            unsafe_allow_html=True,
        )
        number = st.number_input("")

    if number != 0:
        final = simulate_elasticity(number, y_demand, df_elasticity, op)
        final2 = final.copy()
        final2.columns = [
            "Produto",
            "Faturamento Atual",
            "Faturamento Previsto IA",
            "Variação de Faturamento",
            "Percentual de Variação",
        ]
        st.dataframe(final2, use_container_width=True)

        relatorio = make_simulation_report(final, op, number)
        st.markdown(relatorio)


if __name__ == "__main__":
    st.header("Price Elasticity Project")

    df_elasticity, y_demand = prepare_data_and_calculate_elasticity()

    tab1, tab2 = st.tabs(["Elasticidade de Preço", "Simule Cenários"])

    with tab1:
        make_price_elasticity()
    with tab2:
        run_simulation_tab(df_elasticity, y_demand)
