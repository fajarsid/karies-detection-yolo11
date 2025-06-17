import streamlit as st

def show_login_page():
    st.title("🔐 Login Aplikasi Deteksi Karies")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if username == "admin" and password == "admin123":
            st.success("Login berhasil!")
            st.session_state["logged_in"] = True
        else:
            st.error("Username atau password salah.")
