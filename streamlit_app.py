import streamlit as st
<<<<<<< HEAD
# from pages import main, stock, sample

st.set_page_config(initial_sidebar_state="expanded")
=======
from pages import main, stock, sample

>>>>>>> 5b57e653a901ec57c1a098cd4e5ffd5a546f0cc4
state = st.session_state
state.setdefault("is_authenticated", False)
state.setdefault("auth_name", "")

<<<<<<< HEAD
# pages = {
#     "Main Page 🥑": main.page,
#     "Stock Page 💧": stock.page,
#     "Sample Page": sample.page,
# }       
=======
pages = {
    "Main Page 🥑": main.page,
    "Stock Page 💧": stock.page,
    "Sample Page": sample.page,
}       
>>>>>>> 5b57e653a901ec57c1a098cd4e5ffd5a546f0cc4

def login():
    username = state.auth_name
    print("username:", username)
    is_valid_username = username in ['ycwong'] 
    state.is_authenticated = is_valid_username       
    state.auth_name = username 
    return state.is_authenticated

def logout():
    state.is_authenticated = False    
    state.auth_name = ""

if not state.is_authenticated:            
    with st.form(key="login_form"):        
        st.header("🎈Welcome to QuickView 🎈")
        st.markdown("""---""")
        st.text_input(label="Username", key="auth_name")
        submit = st.form_submit_button("Login", on_click=login)
        if not state.is_authenticated and submit:
            st.error("Invalid username")
        elif state.is_authenticated:
            st.success("Redirecting to home page...")
else:    
    st.sidebar.write(f"🎈 Welcome back, {state.get('auth_name')} 🎈")
    st.write(state)
    st.sidebar.markdown("""---""")
    st.sidebar.button("Logout", on_click=logout)        
    st.sidebar.markdown("""---""")
<<<<<<< HEAD
    # for i, (page_name, page_callback) in enumerate(pages.items(), start=1):
    #     st.sidebar.button(f"{i}) {page_name}", on_click=page_callback)  
=======
    for i, (page_name, page_callback) in enumerate(pages.items(), start=1):
        st.sidebar.button(f"{i}) {page_name}", on_click=page_callback)  
>>>>>>> 5b57e653a901ec57c1a098cd4e5ffd5a546f0cc4
