# ML No-Code Studio — Starter

A student-friendly Streamlit app to upload datasets, build preprocessing pipelines via UI, choose models, and open training notebooks in Colab/Kaggle/Hugging Face (user-run).

## Quick setup
1. Create a GitHub repo and add these files (app.py, requirements.txt, pipeline_templates/, screenshots/).
2. Deploy on Streamlit Cloud:
   - Connect GitHub → New app → Select repo and `app.py`.
3. Open the app, create an account (or Guest), and upload a CSV/ZIP.

## How it works
- **Upload & EDA**: preview CSV or sample images.
- **Preprocess & Pipeline**: manual choices build a `pipeline_config.json` saved in `uploads/`.
- **Modeling & Run**: select a saved pipeline and model preset, create a job, and generate “Open in Colab” link to run training in your own account.
- **Jobs**: store minimal job metadata (for tracking).
- **Help**: add screenshots to `screenshots/` and they appear in the help tab.

## Training
- Heavy libraries (PyCaret, PyTorch, Transformers) are installed in the Colab notebook (user-run). This avoids heavy installs on Streamlit hosting.

## Notes & security
- This starter uses a **simple file-backed auth** for learning/demo only. For real apps use OAuth, Firebase Auth, or Auth0.
- Uploaded files are stored in the `uploads/` folder on the server — treat as temporary. For production use S3/Google Cloud Storage.
