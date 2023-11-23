import os
import streamlit as st
import tempfile
import pandas as pd
from datetime import timedelta
import datetime
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
Based on the [PROJECT CONTEXT] below summarize the project into a comprehensive description covering all the categories below:

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

template_2 = """
<<SYSTEM>> You are a project manager and professional in team building. You can build a team and 
    match employees for a specific project the most efficient way. Also you are a professional in understanding what 
    people's roles and skills are required based on the [PROJECT SUMMARY AND SPECIFIC SEQUENTIAL STEPS]. Your answer 
    should not include any annotations and abstract besides the output structure in the following example. <</SYSTEM>>

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

    [TASK] Define key roles for the following project, for each role define necessary skills, both technical and 
    soft-skills, tools. Write down a list that includes specific role, role’s skills and tools. [/TASK]

    [PROJECT SUMMARY AND SPECIFIC SEQUENTIAL STEPS]
    {agent_1_output}
    [/PROJECT SUMMARY AND SPECIFIC SEQUENTIAL STEPS]
"""

# Make a simple sequential chain
def sequential_chain():
    llm = OpenAI(model="text-davinci-003")
    prompt_1 = PromptTemplate(
        input_variables=["name_input", "overview_input", "duration_input", "goals_input", "position_input", "output_input"],
        template=template_1
    )
    chain = LLMChain(llm=llm, prompt=prompt_1)


    inputs = {'name_input': name, 'overview_input': overview, 'duration_input': duration, 'goals_input': goals,
              'position_input': positions, 'outcome_input': desired_outcomes}

    output = chain.run(inputs)

    return output


# Load CSV file and process the data
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    raw_csv = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})

    dataset = raw_csv.load()

    text_splitter_dataset = CharacterTextSplitter(chunk_size=512, chunk_overlap=24)

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
        name = st.text_input('Input project name:', placeholder='Personalized Content Recommendation Engine for Online Streaming Platform')
        overview = st.text_area('Input project overview', placeholder="""In this project description, we present the details of an IT-company project focused on leveraging Machine Learning (ML) and Data Science to develop an innovative recommendation engine for an online streaming platform. The project aims to enhance user experience, increase engagement, and optimize content recommendations based on individual preferences.
        """)
        complete_by = st.date_input('Choose project completion date')
        goals = st.text_area('Input project goals', placeholder="""The primary objective of this IT-company project is to design, build, and deploy a sophisticated recommendation engine that utilizes Machine Learning and Data Science techniques to:
        1. Enhance User Engagement: Provide users with tailored content recommendations that match their viewing history, preferences, and behaviors.
        2. Improve Content Discovery: Facilitate the discovery of new content by suggesting relevant movies, TV shows, and genres that align with users' interests.
        3. Optimize Viewing Experience: Increase user satisfaction by reducing the time spent searching for content and improving the relevance of recommendations.
        4. Boost Platform Retention: Encourage users to spend more time on the platform by consistently delivering compelling and personalized content.
        5. Drive Business Value: Translate improved user engagement into higher viewer retention rates, increased subscriptions, and enhanced brand loyalty.
        """)
        positions = st.multiselect('Choose required positions', position_retrieve(tmp_file_path))
        desired_outcomes = st.text_area('Input project desired outcomes', placeholder="""The successful completion of this project will result in a cutting-edge recommendation engine integrated into the online streaming platform. The engine will deliver accurate and personalized content suggestions to users, ultimately enhancing their viewing experience, increasing engagement, and contributing to the platform's business success.
        """)
        match_button = st.form_submit_button(label="Submit")

        duration = str(complete_by - datetime.date.today())
        positions = ', '.join(positions)

        inputs = {'name_input': name, 'overview_input': overview, 'duration_input': duration, 'goals_input': goals,
              'position_input': positions, 'outcome_input': desired_outcomes}

        if match_button:
            response = sequential_chain()
            output = database.similarity_search(response)
            st.write("Here's the summary of the project based on your input👇")
            st.write(response)
            st.write("Here's an inhouse engineer recommended based on the input dataset👇")
            st.write(output[0].page_content)