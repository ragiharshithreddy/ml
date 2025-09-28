# app.py
import streamlit as st
import pandas as pd
import json
import os
import zipfile
from io import BytesIO
from PIL import Image
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="ML No-Code Studio", layout="wide")

# --- simple local user store (NOT production) -------------
USERS_DB = "users.json"
JOBS_DB = "jobs.json"
UPLOADS_DIR = "uploads"
SCREENSHOTS_DIR = "screenshots"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, obj):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

users = load_json(USERS_DB, {})
jobs = load_json(JOBS_DB, {})

# ----------------- AUTH UI (simple) -----------------------
st.sidebar.title("Account")
auth_mode = st.sidebar.radio("Choose", ["Sign up", "Log in", "Guest"])

current_user = None
if auth_mode == "Sign up":
    username = st.sidebar.text_input("Choose username")
    password = st.sidebar.text_input("Choose password", type="password")
    allow_create = st.sidebar.button("Create account")
    if allow_create:
        if not username or not password:
            st.sidebar.warning("Enter username & password.")
        elif username in users:
            st.sidebar.error("Username exists.")
        else:
            users[username] = {"password": password, "role": "user", "created": str(datetime.utcnow())}
            save_json(USERS_DB, users)
            st.sidebar.success("Account created — please switch to Log in.")
elif auth_mode == "Log in":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Log in"):
        if username in users and users[username]["password"] == password:
            current_user = username
            st.sidebar.success(f"Logged in as {username}")
        else:
            st.sidebar.error("Invalid credentials")
else:
    if st.sidebar.button("Continue as Guest"):
        current_user = "guest"
        st.sidebar.info("You are using a guest session (ephemeral).")

# If user logged in previously during session, show it
if current_user is None and "username" in st.session_state:
    current_user = st.session_state.get("username")

if current_user:
    st.session_state["username"] = current_user

# -------------- Header / Tabs -----------------------------
st.title("ML No-Code Studio")
st.subheader("Upload, explore, configure preprocessing, choose models, and open training in Colab/Kaggle/HF (user-run).")

tabs = st.tabs(["Upload & EDA", "Preprocess & Pipeline", "Modeling & Run", "Jobs", "Help & Screenshots"])
# ---------------- Tab 1: Upload & EDA ----------------------
with tabs[0]:
    st.header("Upload Dataset (CSV or Images ZIP)")
    st.write("Supported: CSV (tabular) or ZIP containing images organized in folders per label.")
    upload_type = st.selectbox("Upload type", ["Tabular CSV", "Image ZIP"], index=0)
    uploaded = st.file_uploader("Select file", type=["csv","zip"], accept_multiple_files=False)

    if uploaded is not None:
        # Save
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        save_name = f"{current_user or 'anon'}_{ts}_{uploaded.name}"
        save_path = os.path.join(UPLOADS_DIR, save_name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved: {save_path}")

        if upload_type == "Tabular CSV":
            try:
                df = pd.read_csv(save_path)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                df = None
            if df is not None:
                st.subheader("Preview")
                st.write(df.head())
                st.write(df.describe(include='all'))
                st.write("Columns:", df.columns.tolist())

                st.subheader("Quick Charts")
                cols = df.columns.tolist()
                if len(cols) >= 2:
                    x = st.selectbox("X-axis", cols, index=0)
                    y = st.selectbox("Y-axis", cols, index=1)
                    kind = st.selectbox("Chart type", ["scatter","line","bar"])
                    if st.button("Plot"):
                        fig = px.scatter(df, x=x, y=y) if kind=="scatter" else (px.line(df, x=x, y=y) if kind=="line" else px.bar(df, x=x, y=y))
                        st.plotly_chart(fig, use_container_width=True)

        else:  # Image ZIP
            try:
                z = zipfile.ZipFile(save_path)
                top_list = z.namelist()[:50]
                st.write(f"Found {len(z.namelist())} items. Showing first 50 entries.")
                st.write(top_list)
                st.subheader("Sample images")
                shown = 0
                for name in z.namelist():
                    if name.endswith("/") or shown >= 12:
                        continue
                    try:
                        data = z.read(name)
                        img = Image.open(BytesIO(data))
                        st.image(img, caption=name, width=160)
                        shown += 1
                    except Exception:
                        continue
            except Exception as e:
                st.error(f"ZIP error: {e}")

# ---------------- Tab 2: Preprocess & Pipeline ----------------------
with tabs[1]:
    st.header("Manual Preprocessing Builder")
    st.write("Select column roles, imputations, encodings and basic options. This builds a `pipeline_config.json` you can run later.")

    st.info("First: choose an uploaded CSV file from the uploads folder (or upload in the Upload tab).")
    # list CSVs
    uploaded_files = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".csv")]
    chosen = st.selectbox("Choose uploaded CSV", ["-- none --"] + uploaded_files)

    if chosen and chosen != "-- none --":
        df = pd.read_csv(os.path.join(UPLOADS_DIR, chosen))
        st.write(df.head())

        st.subheader("Column roles")
        col_roles = {}
        cols = df.columns.tolist()
        for c in cols:
            role = st.selectbox(f"Role for `{c}`", ["feature","target","ignore"], index=0, key=f"role_{c}")
            col_roles[c] = role

        st.subheader("Imputation")
        impute = {}
        for c in cols:
            if df[c].isnull().any():
                strategy = st.selectbox(f"Impute `{c}`", ["none","mean","median","mode","constant"], index=0, key=f"imp_{c}")
                if strategy != "none":
                    val = None
                    if strategy == "constant":
                        val = st.text_input(f"Constant value for `{c}`", key=f"const_{c}")
                    impute[c] = {"strategy": strategy, "fill_value": val}

        st.subheader("Encoding & Scaling")
        encodes = {}
        for c in cols:
            if df[c].dtype == object or df[c].nunique() < 20:
                enc = st.selectbox(f"Encode `{c}`", ["none","onehot","ordinal","target"], key=f"enc_{c}")
                encodes[c] = enc

        st.subheader("Train/Test Split")
        test_size = st.slider("Test size (%)", 5, 50, 20)
        random_state = st.number_input("Random seed", 0, 99999, 42)

        st.subheader("Save Pipeline Config")
        pipeline_name = st.text_input("Pipeline name", value=f"pipeline_{chosen}_{ts}" if 'ts' in locals() else f"pipeline_{chosen}")
        if st.button("Save pipeline config"):
            config = {
                "source_csv": chosen,
                "col_roles": col_roles,
                "impute": impute,
                "encoding": encodes,
                "test_size": float(test_size)/100.0,
                "random_state": int(random_state),
                "created_by": current_user or "anonymous",
                "created_at": str(datetime.utcnow())
            }
            cfg_name = pipeline_name if pipeline_name.endswith(".json") else pipeline_name + ".json"
            cfg_path = os.path.join(UPLOADS_DIR, cfg_name)
            save_json(cfg_path, config)
            st.success(f"Saved pipeline config to {cfg_path}")
            st.json(config)

# ---------------- Tab 3: Modeling & Run ----------------------
with tabs[2]:
    st.header("Modeling & Run")
    st.write("Choose a saved pipeline (from Preprocess) and pick a model. Then generate a notebook to run in Colab/Kaggle/HF or run locally if you host a worker.")

    pipeline_files = [f for f in os.listdir(UPLOADS_DIR) if f.endswith(".json")]
    selected_cfg = st.selectbox("Select pipeline config", ["-- none --"] + pipeline_files)
    if selected_cfg and selected_cfg != "-- none --":
        config = load_json(os.path.join(UPLOADS_DIR, selected_cfg), {})
        st.write("Pipeline preview:")
        st.json(config)

        st.subheader("Model choices (preset)")
        model_type = st.selectbox("Model family", ["scikit-learn (tabular)", "PyTorch CNN (images)", "HuggingFace Transformer (text)"])
        if model_type.startswith("scikit"):
            algo = st.selectbox("Algorithm", ["RandomForest", "LogisticRegression", "XGBoost"])
            n_estimators = st.slider("n_estimators (RF/XGB)", 10, 500, 100)
            lr = st.number_input("learning_rate (unused for RF)", 0.0001, 1.0, 0.01, format="%.4f")
            hp = {"algo": algo, "n_estimators": n_estimators, "lr": lr}
        elif "PyTorch" in model_type:
            epochs = st.slider("Epochs", 1, 200, 10)
            batch = st.selectbox("Batch size", [8,16,32,64])
            hp = {"epochs": epochs, "batch": batch}
        else:
            hp = {"model": "distilbert", "epochs": 3, "batch": 8}

        st.subheader("Where to run")
        run_choice = st.selectbox("Compute backend", ["Open in Colab (user account)", "Open in Kaggle (user)", "Hugging Face (user)", "Run on Server (your worker)"])
        email_notify = st.text_input("Notification email (optional)")

        if st.button("Create Job"):
            job_id = f"job_{int(datetime.utcnow().timestamp())}"
            job = {
                "job_id": job_id,
                "owner": current_user or "anonymous",
                "pipeline_config": selected_cfg,
                "model_spec": hp,
                "run_choice": run_choice,
                "status": "created",
                "created_at": str(datetime.utcnow()),
                "notify": email_notify
            }
            jobs[job_id] = job
            save_json(JOBS_DB, jobs)
            st.success(f"Job {job_id} created. Go to Jobs tab for actions.")
            st.json(job)

        st.markdown("---")
        st.write("Generate notebook link (for user-run backends):")
        repo_user = st.text_input("GitHub username/org (where you uploaded notebook template)", value="your-username")
        repo_name = st.text_input("GitHub repo name", value="ml-no-code-starter")
        notebook_path = st.text_input("Notebook path (in repo)", value="pipeline_templates/training_template.ipynb")

        if st.button("Generate Open-in-Colab link"):
            url = f"https://colab.research.google.com/github/{repo_user}/{repo_name}/blob/main/{notebook_path}"
            st.markdown(f"[Open in Colab]({url})")
            st.info("Open the notebook; edit dataset path or mount Drive and run. This will execute in the USER's account (their quotas apply).")

# ---------------- Tab 4: Jobs ----------------------
with tabs[3]:
    st.header("Jobs Dashboard")
    st.write("Created jobs and their simple statuses (this is meta — you must run the generated notebook in user Colab or run jobs on your worker).")
    if jobs:
        for jid, job in sorted(jobs.items(), key=lambda x: x[0], reverse=True):
            with st.expander(f"Job {jid} — {job.get('status')}"):
                st.write("Owner:", job.get("owner"))
                st.write("Pipeline config:", job.get("pipeline_config"))
                st.write("Model spec:", job.get("model_spec"))
                st.write("Run choice:", job.get("run_choice"))
                if st.button(f"Mark {jid} as completed", key=f"done_{jid}"):
                    jobs[jid]["status"] = "completed"
                    save_json(JOBS_DB, jobs)
                    st.experimental_rerun()
                if st.button(f"Mark {jid} as failed", key=f"fail_{jid}"):
                    jobs[jid]["status"] = "failed"
                    save_json(JOBS_DB, jobs)
                    st.experimental_rerun()
    else:
        st.info("No jobs yet.")

# ---------------- Tab 5: Help & Screenshots ----------------------
with tabs[4]:
    st.header("Help Assistant (choice-based)")
    topic = st.selectbox("Choose topic", [
        "Uploading Dataset",
        "Preprocessing Options",
        "Model Selection",
        "Run Options (Colab/Kaggle/HF)",
        "Error Handling"
    ])
    if topic == "Uploading Dataset":
        st.write("Steps to upload: 1) Select Upload & EDA tab. 2) Choose CSV or ZIP. 3) Click upload. For images, use ZIP with folders for labels.")
        img_path = os.path.join(SCREENSHOTS_DIR, "upload.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Upload example")
    elif topic == "Preprocessing Options":
        st.write("Define column roles, imputations, encodings. Save a pipeline config and then pick it in Modeling.")
        img_path = os.path.join(SCREENSHOTS_DIR, "preprocessing.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Preprocessing UI")
    elif topic == "Model Selection":
        st.write("Pick a model family, set simple hyperparameters, and create a job. Use 'Open in Colab' to run in your own account.")
    elif topic == "Run Options (Colab/Kaggle/HF)":
        st.write("Use the generated notebook link to open in your account. Mount your Drive in Colab to read large datasets and store checkpoints.")
    else:
        st.write("Common errors: wrong file type, missing target column, too-large dataset for Colab free quota. Save checkpoints to Drive and resume.")

# ---------------- Footer ------------------------
st.sidebar.markdown("---")
st.sidebar.write("⚠️ This app is a learning/starter scaffold. Do not use the built-in auth for production. For production, use OAuth/Firebase/Auth0 and secure storage.")
