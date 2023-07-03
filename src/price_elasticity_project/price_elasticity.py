import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def load_data():
    df_elasticity = pd.read_excel("../../data/04_feature/elasticity.xlsx")
    df_bp = pd.read_excel("../../data/04_feature/resultado.xlsx")
    df_c = pd.read_excel("../../data/04_feature/result.xlsx")
    return df_elasticity, df_bp, df_c


def plot_elasticity(df_elasticity):
    st.header("Price Elasticity - Graphs")
    df_elasticity["ranking"] = (
        df_elasticity.loc[:, "price_elasticity"].rank(ascending=False).astype(int)
    )
    df_elasticity = df_elasticity.reset_index(drop=True)

    fig, ax = plt.subplots()
    ax.hlines(
        y=df_elasticity["ranking"],
        xmin=0,
        xmax=df_elasticity["price_elasticity"],
        alpha=0.5,
        linewidth=3,
    )

    for name, p in zip(df_elasticity["name"], df_elasticity["ranking"]):
        ax.text(4, p, name)

    for x, y, s in zip(
        df_elasticity["price_elasticity"],
        df_elasticity["ranking"],
        df_elasticity["price_elasticity"],
    ):
        ax.text(
            x,
            y,
            round(s, 2),
            horizontalalignment="right" if x < 0 else "left",
            verticalalignment="center",
            fontdict={"color": "red" if x < 0 else "green", "size": 10},
        )

    ax.grid(linestyle="--")

    st.pyplot(fig)


def make_price_elasticity():
    tab1, tab2, tab3 = st.tabs(
        ["Price Elasticity", "Business Performance", "Cross Price Elasticity"]
    )
    df_elasticity, df_bp, df_c = load_data()

    with tab1:
        tab4, tab5 = st.tabs(
            ["Price Elasticity - Graphs", "Price Elasticity - Dataframe"]
        )
        with tab4:
            plot_elasticity(df_elasticity)
        with tab5:
            st.header("Price Elasticity - Dataframe")
            df_order_elasticity = (
                df_elasticity[["ranking", "name", "price_elasticity"]]
                .sort_values(by="price_elasticity", ascending=True)
                .set_index("ranking")
            )
            st.dataframe(df_order_elasticity, use_container_width=True)

    with tab2:
        st.header("Business Performance")
        df_bp = df_bp.set_index("name")
        st.dataframe(df_bp, use_container_width=True)
    with tab3:
        st.header("Cross Price Elasticity")
        df_c = df_c.set_index("name")
        st.dataframe(df_c, use_container_width=True)
