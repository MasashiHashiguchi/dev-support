import os
import streamlit as st
import tempfile
import pandas as pd
import datetime
import openai
import chromadb
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.title("Dev Support 🛠" )

# Set openai api key
user_api_key = st.text_input(
    label="OpenAI API key 👇",
    placeholder="OpenAIのAPIキーをペーストしてください",
    type="password"
)

os.environ['OPENAI_API_KEY'] = user_api_key

#upload CSV
uploaded_file = st.file_uploader("upload", type="csv")

# Define templates
template_1 = """
You are a project manager and professional in team building. 
Based on the [PROJECT CONTEXT] below summarize the project into a comprehensive description in 200 words covering all the categories below:

[PROJECT CONTEXT]
[Project Name]
{name_input}

[Project Overview]
{overview_input}

[Project Duration]
{duration_input}

[Project Goals]
{goals_input}

[Team position]
{position_input}

[Desired Outcome]
{outcome_input}
[/PROJECT CONTEXT]
"""

template_2 ="""
You are a project manager and professional in team building. 
Based on the [PROJECT CONTEXT] below summarize the project into a comprehensive description in 200 words covering all the categories below:

[PROJECT CONTEXT]
[Project Name]
{name_input}

[Project Overview]
{overview_input}

[Project Duration]
{duration_input}

[Project Goals]
{goals_input}

[Team position]
{position_input}

[Desired Outcome]
{outcome_input}
[/PROJECT CONTEXT]

SUMMARY:
"""

template_2 = """
You are a project manager and professional in team building. You can build a team and 
match employees for a specific project the most efficient way. Also you are a professional in understanding what 
people's roles and skills are required based on the [PROJECT SUMMARY AND SPECIFIC SEQUENTIAL STEPS]. Your answer 
should not include any annotations and abstract besides the output structure in the following example.
[TASK] Define key roles for the following project, for each role define necessary skills, both technical and soft-skills, tools. Write down a list that includes specific role, role's skills and tools. [/TASK]

[OUTPUT STRUCTURE]
[STRUCTURE FORMAT]
[Role]
[Technical skills]
[Soft Skills]
[Tools]
[/Role]

[/STRUCTURE FORMAT]
[EXAMPLE]
Team Roles, Skills, and Tools for the Project:

Project Manager
Technical Skills: Project management methodologies, communication, risk management.
Soft Skills: Leadership, communication, problem-solving.
Tools: Project management software (e.g., Jira, Trello), communication tools (e.g., Slack, Microsoft Teams).

Data Scientist
Technical Skills: Machine Learning, data analysis, feature engineering, algorithm development.
Soft Skills: Critical thinking, attention to detail, analytical mindset.
Tools: Python, Jupyter Notebook, pandas, scikit-learn, TensorFlow or PyTorch.

Machine Learning Engineer
Technical Skills: Machine Learning algorithms, model training, evaluation, optimization.
Soft Skills: Collaboration, teamwork, problem-solving.
Tools: Python, scikit-learn, TensorFlow or PyTorch, model evaluation metrics.
[/EXAMPLE]
[/OUTPUT STRUCTURE]

[PROJECT SUMMARY AND SPECIFIC SEQUENTIAL STEPS]
{agent_1_output}
[/PROJECT SUMMARY AND SPECIFIC SEQUENTIAL STEPS]
"""

# Make a simple sequential chain
def sequential_chain(name, overview, completed_by, goals, positions, desired_outcomes):
    llm = OpenAI(model="text-davinci-003")
    prompt_1 = PromptTemplate(
        input_variables=["name_input", "overview_input", "duration_input", "goals_input", "position_input", "outcome_input"],
        template=template_1
    )
    chain_1 = LLMChain(llm=llm, prompt=prompt_1)

    prompt_2 = PromptTemplate(
        input_variables=["agent_1_output"],
        template=template_2
    )
    chain_2 = LLMChain(llm=llm, prompt=prompt_2)

    inputs = {'name_input': name, 'overview_input': overview, 'duration_input': completed_by, 'goals_input': goals,
              'position_input': positions, 'outcome_input': desired_outcomes}

    output_first = chain_1.run(inputs)

    output_second = chain_2.run(output_first)

    return output_second


# Load CSV file and process the data
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    raw_csv = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})

    dataset = raw_csv.load()

    text_splitter_dataset = CharacterTextSplitter(chunk_size= 1000, chunk_overlap= 50)

    documents = text_splitter_dataset.split_documents(dataset)

    database = Chroma.from_documents(documents, OpenAIEmbeddings())

    def position_retrieve(path):
        employees = pd.read_csv(path)
        positions = list(employees['Position']) 
        positions = set(','.join(positions).split(sep=","))
        return positions

    with st.form(key='my_form', clear_on_submit=True):

        "### Describe your project"

        #user_input = st.text_input('プロジェクト名の入力:')
        name = st.text_input('Input project name:')
        overview = st.text_area('Input project overview')
        completed_by = st.date_input('Choose project completion date')
        goals = st.text_area('Input project goals')
        positions = st.multiselect('Choose required positions', position_retrieve(tmp_file_path))
        desired_outcomes = st.text_area('Input project desired outcomes')
        match_button = st.form_submit_button(label="Submit")

        duration = str(completed_by - datetime.date.today())
        positions = ', '.join(positions)

        if match_button:
            response = sequential_chain(name, overview, duration, goals, positions, desired_outcomes)
            output = database.similarity_search(query=response)
            st.write("Here's the skills based on your inputs👇")
            st.write(response)
            st.write("Here's an inhouse engineer recommendation based on the input dataset👇")


            result = str(output[0].page_content)
            result_list = result.splitlines()
            for i in range(len(result_list)):
                index = int(i)
                st.write(result_list[index])