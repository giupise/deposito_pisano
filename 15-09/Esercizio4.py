from openai import AzureOpenAI
import streamlit as st

st.title("ChatGPT-like clone but better")

# parametri che verranno poi valorizzati dagli input dell'utente
defaults = {
    "api_version": st.session_state.get("api_version", ""),
    "azure_endpoint": st.session_state.get("azure_endpoint", ""),
    "api_key": st.session_state.get("api_key", ""),
    "azure_model": st.session_state.get("azure_model", "gpt-4o"),
    "configured": st.session_state.get("configured", False),
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "messages" not in st.session_state:
    st.session_state.messages = []

# costruisco il client
@st.cache_resource(show_spinner=False)
def build_client(azure_endpoint: str, api_version: str, api_key: str):
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
    )
    return client

# validazione endpoint
def validate_endpoint(url: str) -> bool:
    return url.startswith("https://")

# validazione credenziali
def validate_credentials(client: AzureOpenAI, model_name: str) -> bool:
    try:
        _ = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return True
    except Exception as e:
        st.error(f"Credenziali o configurazione non valide: {e}")
        return False

# memorizzo le informazioni del form
if not st.session_state.configured:
    st.subheader("Configura Azure OpenAI")
    with st.form("azure_setup", clear_on_submit=False):
        api_version = st.text_input("API Version", placeholder="es. 2024-02-15-preview", value=st.session_state.api_version)
        azure_endpoint = st.text_input("Azure Endpoint", placeholder="https://your-resource.openai.azure.com", value=st.session_state.azure_endpoint)
        api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
        model_name = st.text_input("Model / Deployment name", placeholder="es. gpt-4o, gpt-4o-mini, etc.", value=st.session_state.azure_model)

        submitted = st.form_submit_button("Salva configurazione")
        if submitted:
            errors = []
            # con strip() rimuovo eventuali spazi
            if not api_version.strip(): 
                errors.append("API Version mancante.")
            if not azure_endpoint.strip() or not validate_endpoint(azure_endpoint.strip()):
                errors.append("Azure Endpoint non valido (deve iniziare con https://).")
            if not api_key.strip():
                errors.append("API Key mancante.")
            if not model_name.strip():
                errors.append("Model/Deployment name mancante.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                # creo client con i dati inseriti
                client = build_client(azure_endpoint.strip(), api_version.strip(), api_key.strip())
                # validazione credenziali con chiamata di test
                if validate_credentials(client, model_name.strip()):
                    st.session_state.api_version = api_version.strip()
                    st.session_state.azure_endpoint = azure_endpoint.strip()
                    st.session_state.api_key = api_key.strip()
                    st.session_state.azure_model = model_name.strip()
                    st.session_state.configured = True
                    st.success("Credenziali validate")
                    st.rerun()

# se la configurazione è confermata, mostra chat
if st.session_state.configured:
    with st.sidebar:
        st.caption("Configurazione Azure")
        st.text_input("API Version", value=st.session_state.api_version, disabled=True)
        st.text_input("Endpoint", value=st.session_state.azure_endpoint, disabled=True)
        st.text_input("Model", value=st.session_state.azure_model, disabled=True)
        if st.button("Modifica configurazione"):
            st.session_state.configured = False
            st.rerun()

    # build client
    client = build_client(
        st.session_state.azure_endpoint,
        st.session_state.api_version,
        st.session_state.api_key,
    )

    # mostra storico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # input utente
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # risposta assistant
        with st.chat_message("assistant"):
            try:
                stream = client.chat.completions.create(
                    model=st.session_state["azure_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                    max_tokens=4096,
                    temperature=1.0,
                    top_p=1.0,
                )
                response = st.write_stream(stream)
            except Exception as e:
                st.error(f"Errore durante la chiamata al modello: ricpntrolla le credenziali inserite")
                response = "Mi dispiace, c'è stato un errore durante l'inferenza."

        st.session_state.messages.append({"role": "assistant", "content": response})
