# Resume-JD Comparison Agent

This is a Streamlit application that compares a resume (uploaded as a PDF) against a job description (JD) fetched from a provided URL. The app uses Agno's AI tools and a PostgreSQL vector database for knowledge representation and retrieval.

---

## Features

- Upload a resume in PDF format.
- Enter a job description URL.
- Extracts and embeds resume content into a pgvector database.
- Scrapes JD content using ScrapeGraphTools.
- Uses Groq LLM to generate a structured comparison between the resume and the JD.

---

## Requirements

Install dependencies with:

pip install -r requirements.txt

a .env file with these keys:

DB_URL=your_postgres_url
GEMINI_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key

## Running the App

To start the app:

streamlit run app.py

This will launch the app in your browser.


---

## Notes

- Temporary uploaded files are cleaned up automatically after processing.
- The database must be accessible and support pgvector extensions.
