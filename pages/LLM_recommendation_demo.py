import os
import streamlit as st
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.title("LLM Recommendation Demo 🤖" )
st.write("""このデモはDev Supportにおけるチャット形式での情報検索を試して頂くものです。""")

# Set openai api key
user_api_key = st.text_input(
    label="OpenAI API key 👇",
    placeholder="OpenAIのAPIキーをペーストしてください",
    type="password"
)

os.environ['OPENAI_API_KEY'] = user_api_key

#upload CSV
uploaded_file = st.file_uploader("upload", type="csv")

# Make a simple sequential chain
def sequential_chain(user_input):

    llm = OpenAI(model = "text-davinci-003")

    prompt_1 = PromptTemplate(
        input_variables=["project"],
        template="Question: What is necessary skills for {project} ? \nAnswer :"
    )
    chain_1 = LLMChain(llm=llm, prompt=prompt_1)

    prompt_2 = PromptTemplate(
        input_variables = ["skills"],
        template = "Recommend me 3 {skills} to achieve the project"
    )
    chain_2 = LLMChain(llm=llm, prompt=prompt_2)

    overall_chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=True)

    return overall_chain.run(user_input)


# Load CSV file and process the data
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    raw_csv = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})

    dataset = raw_csv.load()

    text_splitter_dataset = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    documents = text_splitter_dataset.split_documents(dataset)

    database = Chroma.from_documents(documents, OpenAIEmbeddings())

    with st.form(key='my_form', clear_on_submit=True):

        "### Describe your project"

        name = st.text_input('プロジェクト名の入力:')
        overview = st.text_area('プロジェクト概要の入力')
        complete_by = st.date_input('プロジェクト完了の日付を選択')
        goals = st.text_area('プロジェクトのゴールを入力')
        roles = st.multiselect('必要な役職を選択', skills_retrieve(EMP_PATH))
        desired_outcomes = st.text_area('理想的なプロジェクトの結果を入力', placeholder="""The successful completion of this project will result in a cutting-edge recommendation engine integrated into the online streaming platform. The engine will deliver accurate and personalized content suggestions to users, ultimately enhancing their viewing experience, increasing engagement, and contributing to the platform's business success.
        """)
        match_button = st.form_submit_button('Match employees to this project!')



        if match_button and name and overview and complete_by and goals and roles and desired_outcomes:
            response = sequential_chain(user_input)
            output = database.similarity_search(response)
            st.write("Here's necessary skills suggested from the project based on your input👇")
            st.write(response)
            st.write("Here's an inhouse engineer recommended based on the input dataset👇")
            st.write(output[0].page_content)