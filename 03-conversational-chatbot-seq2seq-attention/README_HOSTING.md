# Streamlit Community Cloud Hosting Guide

Streamlit Community Cloud is recommended because the application has one Python entry point and uses
a compact NumPy inference artifact rather than TensorFlow or Keras during startup.

## Community Cloud settings

```text
Repository: unit-mole/lstm-projects
Branch: main
Main file path: 03-conversational-chatbot-seq2seq-attention/app/streamlit_app.py
Python version: 3.12
```

The deployment dependency file is located beside the entry point:

```text
03-conversational-chatbot-seq2seq-attention/app/requirements.txt
```

## Deployment steps

1. Push Project 03 and its workflow to GitHub.
2. Open Streamlit Community Cloud and choose **Create app**.
3. Select `unit-mole/lstm-projects` and branch `main`.
4. Enter the Project 03 main file path shown above.
5. Select Python 3.12 in Advanced settings.
6. Deploy and monitor the build logs.
7. Test canonical prompts, out-of-domain fallback, attention, metrics, and clear-chat behavior.
8. Add the final live URL to the project and root README files.

## Maintenance

The GitHub repository is the application source. Pushing code changes updates the app. Changing
`app/requirements.txt` triggers dependency reinstallation.
