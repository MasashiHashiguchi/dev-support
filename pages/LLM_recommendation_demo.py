import streamlit as st

st.title("LLM Recommendation Demo 🤖" )
st.write("""このデモはDev Supportにおけるチャット形式での情報検索を試して頂くものです。""")

# Set openai api key
user_api_key = st.text_input(
    label="OpenAI API key 👇",
    placeholder="OpenAIのAPIキーをペーストしてください",
    type="password"
)

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

        user_input = st.text_input("Query:", placeholder="Write a project title or summary here", key='input')
        submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            response = sequential_chain(user_input)
            output = database.similarity_search(response)
            st.write("Here's necessary skills suggested from the project based on your input👇")
            st.write(response)
            st.write("Here's an inhouse engineer recommended based on the input dataset👇")
            st.write(output[0].page_content)