import streamlit as st

def show_login_page():
    # Inisialisasi session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    
    # Jika sudah login, tidak perlu tampilkan form
    if st.session_state.get("logged_in"):
        st.success("Anda sudah login!")
        return

    # Membuat container di tengah
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🔐 Login")
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Spacer
        
        with st.container(border=True):
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Masukkan username")
                password = st.text_input("Password", type="password", placeholder="Masukkan password")
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)  # Spacer
                submit = st.form_submit_button("Login", type="primary", use_container_width=True)

        if submit:
            if not username or not password:
                st.error("Username dan password harus diisi")
            elif username == "admin" and password == "admin123":
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success("Login berhasil!")
                st.rerun()
            else:
                st.error("Username atau password salah")