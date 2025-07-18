import streamlit as st
import os
from agno.agent import Agent
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.embedder.google import GeminiEmbedder
from agno.models.groq import Groq
from agno.tools.scrapegraph import ScrapeGraphTools
from dotenv import load_dotenv


load_dotenv()
db_url = os.getenv("DB_URL", "postgresql://neondb_owner:npg_cNOFYky9t2XZ@ep-misty-mode-adb61k7o-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")


gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not gemini_api_key:
    st.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()
if not db_url:
    st.error("DB_URL not found in environment variables. Please set it or check your hardcoded URL.")
    st.stop()


def get_resume_comparison_agent(uploaded_resume_path: str):
    """
    Initializes and returns the Agno Agent configured for resume and JD comparison.
    The resume is loaded into a knowledge base, and the agent is equipped with web scraping tools.
    """
  
    resume_kb = PDFKnowledgeBase(
        path=uploaded_resume_path,
        vector_db=PgVector(
            table_name="resume_documents", # Dedicated table for resume embeddings
            db_url=db_url,
            embedder=GeminiEmbedder(api_key=gemini_api_key),
        ),
    )

    knowledge_base = CombinedKnowledgeBase(
        sources=[resume_kb],
        vector_db=PgVector(
            table_name="combined_documents", 
            db_url=db_url,
            embedder=GeminiEmbedder(api_key=gemini_api_key),
        ),
    )
    st.info("Loading resume into knowledge base (this may take a moment on first run)...")
    try:
        resume_kb.load(recreate=False)  #it won't drop existing data.`recreate=False`
        knowledge_base.load(recreate=False) # Ensure combined KB is updated
        st.success("Resume loaded successfully into knowledge base.")
    except Exception as e:
        st.error(f"Error loading resume into knowledge base: {e}. Please check your DB connection and credentials.")
        st.stop()


   
    agent = Agent(
        model=Groq(id="qwen/qwen3-32b", api_key=groq_api_key),
        knowledge=knowledge_base,
        search_knowledge=True,
        tools=[ScrapeGraphTools(smartscraper=True)], 
        show_tool_calls=True
    )
    return agent

def main():
    st.set_page_config(page_title="Resume-JD Comparison Agent", layout="centered")

    html_temp = """
    <div style="background-color:#28a745;padding:10px;border-radius:5px;">
    <h2 style="color:white;text-align:center;margin-bottom:0;">Resume-JD Comparison Agent</h2>
    <p style="color:white;text-align:center;font-size:0.9em;">Powered by Agno, Groq, Gemini & Neon DB</p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.markdown("""
        <style>
            .stFileUploader label {
                font-size: 16px;
                font-weight: bold;
                color: #007bff; /* Blue color for labels */
            }
            div.stButton > button:first-child {
                background-color: #DD3300; /* Red-orange */
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 8px;
                border: none;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            div.stButton > button:first-child:hover {
                background-color: #CC2200; /* Darker red-orange on hover */
            }
            .stTextInput label {
                font-size: 16px;
                font-weight: bold;
                color: #007bff;
            }
            .stSuccess, .stInfo, .stWarning, .stError {
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            .stSuccess { background-color: #d4edda; color: #155724; border-color: #c3e6cb; }
            .stInfo { background-color: #d1ecf1; color: #0c5460; border-color: #bee5eb; }
            .stWarning { background-color: #fff3cd; color: #856404; border-color: #ffeeba; }
            .stError { background-color: #f8d7da; color: #721c24; border-color: #f5c6cb; }
            h3 { color: #333333; }
            h4 { color: #555555; }
        </style>
    """, unsafe_allow_html=True)

    st.write("---")

    # Resume Upload
    uploaded_file = st.file_uploader("Upload Your Resume (PDF):", type="pdf")

    # Job Description URL Input
    jd_url = st.text_input("Enter the URL of the Job Description:")

    st.write("---") 
    if st.button("Compare Resume and JD"):
        if uploaded_file is None:
            st.warning("Please upload a resume PDF.")
            st.stop()

        if not jd_url:
            st.warning("Please enter a Job Description URL.")
            st.stop()

        # Saving uploaded file to a temporary location later ive done cleanup
        temp_dir = "temp_uploaded_resumes"
        os.makedirs(temp_dir, exist_ok=True)
        uploaded_resume_path = os.path.join(temp_dir, uploaded_file.name)
        with open(uploaded_resume_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.info(f"Resume '{uploaded_file.name}' uploaded successfully. Starting analysis...")

        try:
            agent = get_resume_comparison_agent(uploaded_resume_path)
            query = f"""
            Analyze the provided resume (available in my knowledge base) against the job description found at this URL: {jd_url}.
            Your task is to provide a comprehensive compatibility analysis in a structured Markdown format.

            **Crucial Instruction:** You *must* use the ScrapeGraphTools to get the content from the provided JD URL.
            Then, compare that scraped content with the information you can retrieve from the resume knowledge base.

            Provide the comparison in the following structured format:

            ---
            ### **Resume-Job Description Compatibility Analysis**

            #### **1. Overall Compatibility Score (out of 10):**
            [Provide a numerical score from 1 to 10. Higher is better.]

            #### **2. Key Matches & Strengths:**
            * Highlight specific skills, experiences, and qualifications from the resume that directly align with the job description.
            * Mention any standout achievements or experiences that are particularly relevant.

            #### **3. Gaps & Areas for Improvement:**
            * Identify skills, experiences, or keywords mentioned in the job description that are missing or not strongly represented in the resume.
            * Point out any discrepancies or areas where the resume could be stronger to match the JD.

            #### **4. Actionable Resume Improvement Suggestions:**
            * Suggest specific changes or additions to the resume to better target this job role.
            * Include advice on how to rephrase existing bullet points, add new sections, or highlight certain experiences to bridge the identified gaps.
            * Recommend incorporating specific keywords or phrases from the JD.

            ---
            """

            st.info("Agent is processing... This may take a few moments as it scrapes the JD, reads your resume, and performs the detailed analysis.")
            results = agent.run(query, markdown=True)
            st.success('Comparison Results:')
            st.markdown(results.content)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.exception(e) 

        finally:
            if uploaded_resume_path and os.path.exists(uploaded_resume_path):
                os.remove(uploaded_resume_path)
                st.info(f"Cleaned up temporary file: {uploaded_resume_path}")
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
                st.info(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == '__main__':
    main()