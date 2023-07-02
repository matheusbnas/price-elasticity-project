"""
This is a boilerplate pipeline
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_streamlit


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_streamlit,
                inputs=["price_elasticity", "business_performance", "cross_price_elasticity"],
                outputs=None,
                name="run_streamlit",
            ),
        ]
    )
