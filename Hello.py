

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Dev Support",
        page_icon="🛠️",
    )

    st.write("# Welcome to Dev Support 👋")

    st.markdown(
        """
        Dev Supportは社内の人材を効率的に選別する目的で開発されたアプリケーションです。
        社内エンジニアの検索機能と携わるプロジェクトを入力してスキルの類似度が高いエンジニアをレコメンドする機能を備えています。
        サイドバーから機能を試してみましょう！
    """
    )


if __name__ == "__main__":
    run()
