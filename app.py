import streamlit as st
import functions as mts
import time
import redis
import json


class DocumentEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'metadata'):
            return {
                'metadata': obj.metadata,
                'page_content': obj.page_content
            }
        return super().default(obj)
    
redis_client = redis.Redis(
    host='10.114.204.147',
    port=6379,
    socket_timeout=5,
    socket_connect_timeout=5,
    socket_keepalive=True,
    retry_on_timeout=True,
    max_connections=10,
    ssl=False,
    db=0,
    decode_responses=True
)

CACHE_TTL = 3600

def get_cached_response(prompt):
    """Get cached response for a prompt"""
    return redis_client.get(f"chat:prompt:{prompt}")

def get_source_path(doc):
    """Extract source path from document structure"""
    if isinstance(doc, dict):
        return doc["source"]
    return doc.metadata["source"]

def document_to_dict(doc):
    """Convert Document object to serializable dictionary"""
    if isinstance(doc, dict):
        return doc
    return {
        'chunk_text': doc.page_content,
        'source': {
            'source': doc.metadata['source']
        }
    }



def cache_response(prompt, response):
    """Cache the response with serializable data"""
    serializable_response = {
        'answer_with_citations': response.answer_with_citations,
        'cited_chunks': [document_to_dict(doc) for doc in response.cited_chunks]
    }
    
    redis_client.setex(
        f"chat:prompt:{prompt}",
        CACHE_TTL,
        json.dumps(serializable_response, cls=DocumentEncoder)
    )

docs = None

st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 42% !important; # Set the width to your desired value
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Initialize session state keys if they don't exist
for key in [
    "show_upload_button",
    "show_upload_form",
    "show_success_message_upload",
    "show_success_message_update",
    "success_message",
    "show_table",
    "show_update_button",
    "show_update_form",
]:
    if key not in st.session_state:
        st.session_state[key] = (
            True if key in ["show_upload_button", "show_table", "show_update_button"] else False
        )

# Function to reset the success message and related state variables
def reset_state(
    show_upload_button=True,
    show_upload_form=False,
    show_success_message_upload=False,
    show_success_message_update=False,
    show_table=True,
    show_update_button=True,
    show_update_form=False,
):
    st.session_state.update(
        {
            "show_upload_button": show_upload_button,
            "show_upload_form": show_upload_form,
            "show_table": show_table,
            "show_success_message_upload": show_success_message_upload,
            "show_success_message_update": show_success_message_update,
            "show_update_button": show_update_button,
            "show_update_form":show_update_form,
        }
    )
    st.rerun()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "username" not in st.session_state:
    st.session_state["username"] = None

if st.session_state["authenticated"] and st.session_state["username"] != None:
    if st.button("Logout", type="primary"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        reset_state()

    st.header("Knowledge Management for GenAI ✨")
    st.success(
        "Manage you Chatbot's Document Knowledge with ease!",
        icon="🔥",
    )
    # st.markdown(
    #     """
    # This streamlit app showcases the usage of the new custom streamlit component streamlit-token-craft
    # For more information, please [click here](https://github.com/stavrostheocharis/streamlit-token-craft).
    # """
    # )

    st.divider()
    with st.sidebar:
        if st.session_state["show_upload_button"]:
            if st.button("Upload New Document"):
                reset_state(show_upload_button=False, show_update_button=False, show_upload_form=True, show_table=False)

        if st.session_state["show_upload_form"]:
            with st.form(key="upload_key_form"):
                file_names = st.file_uploader(
                    "Upload your files here!",
                    accept_multiple_files=True,
                    disabled=st.session_state["show_success_message_upload"]
                )

                col1, col2 = st.columns([3, 10])
                with col1:
                    submit_button = st.form_submit_button(
                        label="Upload Files",
                    )
                with col2:
                    cancel_button = st.form_submit_button(label="Cancel", type="primary")

                if submit_button:
                    if file_names:  # Check if the name is not empty
                        with st.spinner('Uploading'):
                            mts.upload_files(file_names, st.session_state["username"])
                        st.session_state[
                            "success_message"
                        ] = "New files successfully uploaded!"
                        st.session_state["show_success_message_upload"] = True
                        reset_state(
                            show_upload_button=False,
                            show_update_button=False,
                            show_success_message_upload=True,
                            show_table=False,
                        )
                    else:
                        st.warning("Please select the files first!.")
                elif cancel_button:
                    reset_state()

        if st.session_state["show_update_form"]:
            with st.form(key="update_key_form"):
                st.text(
                    "Do you really want to update the knowledge?",
                    # disabled=st.session_state["show_success_message_update"]
                )

                col1, col2 = st.columns([3.2, 10])
                with col1:
                    submit_button = st.form_submit_button(
                        label="Update Knowledge",
                    )
                with col2:
                    cancel_button = st.form_submit_button(label="Cancel", type="primary")

                if submit_button:
                        with st.spinner('Updating Knowledge. Please wait a moment.'):
                            # mts.refresh_vertex_search(st.session_state["username"])
                            st.session_state["messages"] = [{"role": "assistant", "content": "Halo, aku akan menjawab semua pertanyaan kamu terkait informasi personal. Silahkan ditanyakan ya! Aku senang banget kalau bisa membantu 😊"}]
                            mts.clear_docai_output(st.session_state["username"])
                            docs = mts.get_docs(st.session_state["username"])
                            vector_store = mts.update_vector_store()
                            mts.add_data(vector_store, docs)
                            retriever = mts.retriever(vector_store)
                            create_answer = mts.create_answer()
                            output_parser = mts.output_parser()
                            # chat = mts.initialize_chat_bot()
                            # parameters = mts.initialize_parameter_bot()
                        st.session_state[
                            "success_message"
                        ] = "Success Update the knowledge!"
                        st.session_state["show_success_message_update"] = True
                        reset_state(
                            show_upload_button=False,
                            show_update_button=False,
                            show_success_message_update=True,
                            show_table=False,
                        )
                elif cancel_button:
                    reset_state()


        # Show a success message if a new key was added
        if st.session_state["show_success_message_upload"]:
            container = st.container(border=True)
            container.write(
                "Your files have been uploaded! Please kindly reconfigure the knowledge of your chatbot and chatbot will become smarter! ✨"
            )
            container.success(st.session_state["success_message"])
            # 'OK' button to reset the success state
            if container.button("OK"):
                reset_state(show_success_message_upload=False)

        if st.session_state["show_success_message_update"]:
            container = st.container(border=True)
            container.write(
                "The knowledge has upgraded. You can try to your chatbot immediately! ✨🧙‍♂️"
            )
            container.success(st.session_state["success_message"])
            # 'OK' button to reset the success state
            if container.button("OK"):
                reset_state(show_success_message_update=False)

        if st.session_state["show_table"]:
            # Display the keys in the table with the hashed version
            with st.spinner('Preparing Table'):
                rendered_data = mts.list_files_in_bucket(st.session_state["username"])
                st.table(rendered_data)
            

            # Check for updates from the React component
            if rendered_data is not None:
                needs_rerun = False

                if needs_rerun:
                    st.rerun()
        
        if st.session_state["show_update_button"]:
            if st.button("Submit Document"):
                reset_state(show_update_button=False, show_upload_button=False, show_update_form=True, show_table=False)

    docs = mts.get_docs_from_bucket(st.session_state["username"])
    vector_store = mts.update_vector_store()
    retriever = mts.retriever(vector_store)
    create_answer = mts.create_answer()
    output_parser = mts.output_parser()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Halo, aku akan menjawab semua pertanyaan kamu terkait informasi personal. Silahkan ditanyakan ya! Aku senang banget kalau bisa membantu 😊"}]

    # Display existing chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        cached_response = get_cached_response(prompt)
        if cached_response:
            response_data = json.loads(cached_response)
            st.session_state.messages.append({"role": "assistant", "content": response_data['answer_with_citations']})
            st.chat_message("assistant").write(response_data['answer_with_citations'])
            
            # Show source chunks for cached response
            with st.expander("View Source Chunks"):
                for idx, doc in enumerate(response_data['cited_chunks']):
                    with st.container(border=True):
                        st.markdown(f"**Source #{idx}**")
                        st.markdown("**Content:**")
                        st.markdown(doc["chunk_text"])
                        st.markdown("**Source:**")
                        filename = doc["source"].split('/')[-1]
                        st.markdown(f"File: {filename}")
        else:
            with st.spinner('Preparing'):
                response = mts.qa_with_check_grounding.invoke({
                            "query": prompt,
                            "create_answer": create_answer,
                            "retriever": retriever,
                            "output_parser": output_parser,
                            "docs": docs
                        })
                cache_response(prompt, response)
                st.session_state.messages.append({"role": "assistant", "content": response.answer_with_citations})
                st.chat_message("assistant").write(response.answer_with_citations)
                with st.expander("View Source Chunks"):
                    for idx, doc in enumerate(response.cited_chunks):
                        with st.container(border=True):
                            # Display chunk number
                            st.markdown(f"**Source #{idx}**")
                            
                            # Display chunk text
                            st.markdown("**Content:**")
                            st.markdown(doc["chunk_text"])
                            
                            # Display source information
                            st.markdown("**Source:**")
                            if isinstance(doc, dict):
                                filename = doc["source"].split('/')[-1]
                            else:
                                filename = doc.metadata["source"].split('/')[-1]
                            st.markdown(f"File: {filename}")
else:
    st.header("Knowledge Management for GenAI ✨")
    st.divider()
    with st.form(key="login"):
        username = st.text_input(
            label="Username",
            placeholder="Input your username",
            disabled=st.session_state["authenticated"],
        )

        password = st.text_input(
            label="Password",
            placeholder="Input your password",
            type="password",
            disabled=st.session_state["authenticated"],
        )

        if st.form_submit_button(
            label="Login",
            disabled=st.session_state["authenticated"],
            type="primary",
        ):
            with st.spinner("Logging In. Please wait a moment"):
                user_result = mts.authentication(username, password)

            if len(user_result) > 0:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success("Login Success!")
                with st.spinner("Please wait a moment."):
                    time.sleep(2)
                st.rerun()
            else:
                st.error("Login Failed")