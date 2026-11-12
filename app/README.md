---
title: Where do these tweets come from?
emoji: 🍁
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Guess a Canadian Twitter user's home metro from their tweets
---

# Web demo

Paste a few tweets and see which of 41 Canadian metros the author most likely lives in, with a probability map and the words that drove the guess.

The model is `logreg_2026` from the main project (word and hashtag TF-IDF + logistic regression), fitted on all 420 users of the thesis dataset. It holds no @handles, only terms used by at least 5 different people, and is about 5 MB.

## Run locally

```bash
pip install -e ".[app]"
geoloc train-demo      # writes app/model.joblib and app/metros.json (needs the data, see main README)
python app/app.py      # http://localhost:7860
```

## Deploy to Hugging Face Spaces

Create a Gradio Space and upload this folder's `app.py`, `requirements.txt`, `README.md`, `model.joblib` and `metros.json`. The app runs on its own when `geoloc` isn't installed.
