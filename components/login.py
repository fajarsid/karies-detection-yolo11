import streamlit as st

def show_login_page():
    # Inisialisasi session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    
    # Jika sudah login, tidak perlu tampilkan form
    if st.session_state.get("logged_in"):
        st.success("Anda sudah login!")
        return

    st.title("🔐 Login Aplikasi Deteksi Karies")

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Masukkan username")
        password = st.text_input("Password", type="password", placeholder="Masukkan password")
        submit = st.form_submit_button("Login", type="primary")

    if submit:
        if not username or not password:
            st.error("Username dan password harus diisi")
        elif username == "admin" and password == "admin123":
            st.session_state["logged_in"] = True
            st.session_state["username"] = username  # Simpan username
            st.success("Login berhasil! Mengarahkan...")
            st.rerun()  # Refresh halaman
        else:
            st.error("Username atau password salah")