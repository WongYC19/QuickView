import streamlit as st

state = st.session_state
state.setdefault("is_authenticated", False)
state.setdefault("auth_name", "")

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