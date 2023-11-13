import streamlit as st

st.write("# Dev Support for search with chat")

# Set openai api key
user_api_key = st.text_input(
    label="OpenAI API key 👇",
    placeholder="OpenAIのAPIキーをペーストしてください",
    type="password"
)

#upload CSV
uploaded_file = st.file_uploader("upload", type="csv")

# Build chain with uploaded files
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
        'delimiter': ','})

    data = loader.load()

    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo', openai_api_key=user_api_key),retriever=vectors.as_retriever())

# activate a search function
search_qlist = ["Can you make a list of Frontend Engineer", "Can you make a list of Backend Engineer", "Can you make a list of DevOps Engineer", "Can you make a list of AI and Data Scientist"]

select_q_search = st.selectbox("Select a question from the dropdown", search_qlist)

def conversational_chat(prompt):
    response = chain({"question": prompt, "chat_history": st.session_state['history']})
    st.session_state['history'].append((prompt, response["answer"]))

    return response["answer"]

if 'hisotry' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# container for the chat history
response_container = st.container()

if st.button('SEARCH'):
    get_ans = conversational_chat(select_q_search)

    st.session_state['past'].append(select_q_search)
    st.session_state['generated'].append(get_ans)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
            #with st.chat_message("user"):
            #   st.write(st.session_state["past"][i])
            #with st.chat_message("assistant"):
            #    st.write(st.session_state["generated"][i])