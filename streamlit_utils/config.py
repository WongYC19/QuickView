import os
import pathlib

import PIL
import streamlit as st

current_folder = pathlib.Path(__file__).parent.resolve()
icon_file = os.path.join(current_folder, "quickview_logo.png")
page_icon = PIL.Image.open(icon_file)
state = st.session_state

def config_page():
    # Config page title and icon    
    st.set_page_config(page_title="Quick View", page_icon=page_icon, layout="wide")

    # Minimalize the default features
    hide_menu_style = """
        <style>
             MainMenu, header, footer {visibility: hidden;}         
            .block-container { padding-top: 3rem; }
        </style>
    """ 

    st.markdown(hide_menu_style, unsafe_allow_html=True)

def get_state():    
    state.setdefault("is_authenticated", False)
    state.setdefault("auth_name", "")
    return state

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
    
    
    