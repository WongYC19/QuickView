import streamlit as st
from streamlit_utils.config import config_page, get_state, login, logout
import os

for (root,dirs,files) in os.walk(topdown=True):
  print(root)
  print(dirs)
  print(files)
  print('--------------------------------')
      
# config_page()
# state = get_state()

# if not state.is_authenticated:            
#     with st.form(key="login_form"):        
#         st.header("🎈Welcome to QuickView 🎈")
#         st.markdown("""---""")
#         st.text_input(label="Username", key="auth_name")
#         submit = st.form_submit_button("Login", on_click=login)
#         if not state.is_authenticated and submit:
#             st.error("Invalid username")
#         elif state.is_authenticated:
#             st.success("Redirecting to home page...")
# else:    
#     st.sidebar.write(f"🎈 Welcome back, {state.get('auth_name')} 🎈")
#     st.write(state)
#     st.sidebar.markdown("""---""")
#     st.sidebar.button("Logout", on_click=logout)        
#     st.sidebar.markdown("""---""")
