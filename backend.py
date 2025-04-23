#%%
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='C:/Users/DELL/PycharmProjects/DataQuestHackathon/insight_env/env')
groq_api_key = os.getenv("GROQ_API_KEY")

print("GROQ key loaded:", groq_api_key is not None)
print(groq_api_key)
#%%
from openai import RateLimitError
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_csv_agent
from langchain.agents import AgentExecutor
from collections import deque
import speech_recognition as sr
import sys
import os
sys.path.append(os.getcwd())
from DataCleaning import dataCleanPipeLine
import pandas as pd
import pyttsx3
import time
from DataVisualizationPipeline import dataVisualizationPipeline
import spacy
from Summary import summary
import globals
import tempfile
from InsightsPipeline import insightsGenerationPipeline
#%%
# === Setup: Load Model ===
model_name = "facebook/bart-large-mnli"
cache_path = "./models"

model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_path)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

print("Model downloaded and ready!")

def initialize_agent():
    if globals.global_df is not None:
       llm = ChatGroq(
         model_name="llama-3.3-70b-versatile",
         temperature=0.7
       )
       globals.llm = llm
       try:
           # Save the DataFrame to a temporary CSV file
           temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
           globals.global_df.to_csv(temp_file.name, index=False)

           # Pass the temp file path instead of DataFrame
           globals.agent = create_csv_agent(
               llm,
               temp_file.name,  # 👈 This is the fix!
               verbose=False,
               allow_dangerous_code=True
           )
           print("✅ Agent initialized successfully.")
           return "Agent initialized successfully."

       except Exception as e:
           print(f"❌ Agent initialization failed: {e}")
           return f"Agent initialization failed: {str(e)}"
    else:
      print("⚠️ No dataset uploaded yet.")
      return "No dataset uploaded yet."

memory = deque(maxlen=15)
#%%
import pandas as pd
import os



"""
def load_dataset(file=None, default_path: str = "Hotel_Reviews.csv") -> pd.DataFrame:
    try:
        if file is not None:
            # Check if this is a file-like object (e.g. from Gradio or Streamlit)
            if hasattr(file, "read"):
                if file.name.lower().endswith((".xls", ".xlsx")):
                    return pd.read_excel(file)
                elif file.name.lower().endswith(".csv"):
                    return pd.read_csv(file)
                else:
                    raise ValueError(f"Unsupported file type: {file.name}")
            elif isinstance(file, str) and os.path.exists(file):
                # If an actual file path was passed
                if file.lower().endswith((".xls", ".xlsx")):
                    return pd.read_excel(file)
                elif file.lower().endswith(".csv"):
                    return pd.read_csv(file)
                else:
                    raise ValueError(f"Unsupported file type: {file}")
            else:
                raise ValueError("Invalid file input or file does not exist.")

        # If no file, fallback to default
        if default_path.lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(default_path)
        elif default_path.lower().endswith(".csv"):
            return pd.read_csv(default_path)
        else:
            raise ValueError(f"Unsupported file type for default path: {default_path}")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")

def load_dataset(file) -> pd.DataFrame:
    try:
        if file is None:
            raise ValueError("No file uploaded.")

        filename = file.name.lower()
        if filename.endswith(".csv"):
            return pd.read_csv(file)  # Gradio already provides a file-like object
        elif filename.endswith((".xls", ".xlsx")):
            return pd.read_excel(file)
        else:
            raise ValueError("Unsupported file type. Please upload a .csv, .xls, or .xlsx file.")
    except Exception as e:
        raise ValueError(f"Error reading the file: {str(e)}")

"""""


def load_dataset(file) -> pd.DataFrame:
    try:
        if file is None:
            raise ValueError("No file uploaded.")

        filename = file.lower()
        if filename.endswith(".csv"):
            try:
                return pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(file, encoding='ISO-8859-1')
        elif filename.endswith((".xls", ".xlsx")):
            return pd.read_excel(file)
        else:
            raise ValueError("Unsupported file type. Please upload a .csv, .xls, or .xlsx file.")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")
#%%
intent_groups = {
    "data cleaning": [
        "clean data", "fix columns", "remove duplicates", "handle missing values", "drop nulls",
        "fill missing values", "fix nulls", "remove outliers", "data preprocessing",
        "impute with mean", "impute with median", "impute with mode", "impute with knn",
        "impute with linear", "clean", "fix", "drop", "fill", "remove", "preprocess",
        "impute", "nulls", "missing", "outliers"
    ],
    "eda visualization": [
        "exploratory data analysis", "eda", "visualize data", "plot histogram", "box plot",
        "draw scatter", "show charts", "plot data", "visuals", "histogram", "scatter plot",
        "box plot", "bar plot", "hist", "box", "bar", "scatter", "visualize", "plot",
        "chart", "graph", "vis", "charts", "graphs"
    ],
    "insights": [
        "get insights", "what insights", "detect trends", "show patterns", "get correlations",
        "interesting insights", "understand data", "insights", "trends", "patterns",
        "correlations", "analyze", "find trends", "spot patterns", "interesting",
        "understand", "gain insights", "key findings"
    ],
    "anomalies": [
        "find anomalies", "spot outliers", "detect outliers", "unusual data", "abnormal entries",
        "anomalies", "outliers", "unusual", "abnormal", "detect", "spot", "find",
        "odd values", "strange data", "unexpected"
    ],
    "summary": [
        "data summary", "summarize dataset", "overview", "mean median mode", "describe dataset",
        "profile dataset", "summary", "summarize", "overview", "describe", "profile",
        "stats", "statistics", "basic info", "dataset info", "dataset summary"
    ],
    "describe columns": [
        "column details", "explain columns", "describe fields", "what do columns mean",
        "feature descriptions", "describe columns", "explain", "details", "columns",
        "fields", "features", "column meaning", "feature meaning", "column info",
        "field info", "what are columns", "what are features"
    ],

}

candidate_labels = list(intent_groups.keys())

# === Intent Classifier ===
def classify_intent(user_query):
    result = classifier(user_query, candidate_labels)
    return result['labels'][0]
#%%
def extract_cleaning_method(query):
    query = query.lower()
    if "mean" in query:
        return "mean"
    elif "median" in query:
        return "median"
    elif "mode" in query:
        return "mode"
    elif "linear" in query:
        return "linear"
    elif "knn" in query:
        return "knn"
    else:
        return None
#%%
nlp = spacy.load("en_core_web_sm")

def extract_eda_info(message):
    plot_types = ["scatter", "bar", "box", "hist"]
    message_lower = message.lower()
    
    # Identify plot type using simple match
    plot_type = next((ptype for ptype in plot_types if ptype in message_lower), None)

    columns = extract_column_names(message)

    return plot_type if plot_type else None, columns if columns else None
#%%
def data_cleaning(method):
    cleaned_df, logs = dataCleanPipeLine(globals.global_df, method)
    cleaned_df.to_csv("cleaned_dataset.csv", index=False)
    return "✅ Data cleaned successfully! Columns fixed, duplicates dropped, missing values handled."

def eda_visualization(method,cols):
    results = dataVisualizationPipeline(globals.global_df, method, cols)
    logs = results["logs"]
    plots = results["plots"]  # List of image paths
    return {
        "message": "✅ Created visualizations such as histograms, box plots, and scatter plots.",
        "plots": plots
    }

def insights(df,clean_df,columns):
    dataset_description_prompt = """
        Give me a brief, insight-focused summary of this dataset in 2 to 3 lines, highlighting key patterns, trends, and any notable relationships between variables
        """
    dataset_description = globals.llm.predict(dataset_description_prompt)
    insights_payload = insightsGenerationPipeline(globals.global_df,clean_df,columns)
    combined_payload = {
        "dataset_description": dataset_description,
        "insights": insights_payload
    }
    textual_insights_prompt = f"""
     Based on the dataset description and the numerical insights (like feature importance, outlier detection, profiling, and clustering results), generate a well-structured, easy-to-understand summary of key findings. The insights should be understandable to both technical and non-technical audiences, avoiding heavy jargon but still conveying important patterns, trends, and implications.
     ### Dataset Description:
    {combined_payload['dataset_description']}
    ### Numerical insights:
    {combined_payload['insights']}
     """
    textual_insights = globals.llm.predict(textual_insights_prompt)
    return textual_insights

def anomalies():
    prompt="Scan the dataset for any unusual or outlier values. Mention what they are and in which column they appear. Explain why they might be important."
    return  globals.agent.run(prompt)

def summary_func():

    summary_data= summary(globals.global_df)
    result = f"📊 **Dataset Summary** 📊\n"
    result += f"• Rows: {summary_data['num_rows']}\n"
    result += f"• Columns: {summary_data['num_columns']}\n\n"

    result += "🧠 **Data Types:**\n"
    for dtype, cols in summary_data['types_of_data'].items():
      result += f"  - {dtype}: {', '.join(cols)}\n"

    if summary_data['missing_or_unusual']:
      result += "\n⚠️ **Missing or Unusual Values:**\n"
      for col, issues in summary_data['missing_or_unusual'].items():
         issue_str = ", ".join(f"{k}: {v}" for k, v in issues.items())
         result += f"  - {col}: {issue_str}\n"
    else:
      result += "\n✅ No missing or unusual values found."

    return result

def describe_columns():
    prompt = "Please describe each column in the dataset, including its data type and meaning if possible."
    return globals.agent.run(prompt)
#%%
def enhanced_query(query):
    try:
        history = "\n".join([f"User: {q}\nBot: {r}" for q, r in memory])
        full_prompt = f"Given the context below, answer the following question:\n{history}\n\nUser: {query}"
        response = globals.agent.run(full_prompt)
    except Exception as e:
        print(f"Error processing the query: {e}")
        response = "Sorry, there was an error processing your query. Please try again."
    return response
#%%
def store_in_memory(user_query, response):
    memory.append((user_query, response))
#%%
def initialize_data_insights():
    print("\n🧠 [Insight Agent Initialized: Profiling and Anomaly Detection]\n")

    print("🔍 Profiling your dataset...")
    try:
        profile_summary = globals.agent.run(
            """
            Give a simple summary of the dataset:
            - Number of rows and columns
            - Types of features (categorical, numeric)
            - Any missing values
            - Initial observations (like weird values or patterns)
            Respond in bullet points, simple and clean.
            """
        )
    except Exception as e:
        profile_summary = f"Failed to get profile summary. Error: {e}"

    print("🚨 Scanning for anomalies...")
    try:
        anomalies = globals.agent.run(
            "Are there any unusual values or outliers in the dataset? Mention which column and why they matter."
        )
    except Exception as e:
        anomalies = f"Failed to detect anomalies. Error: {e}"

    return profile_summary, anomalies

def insight_agent_interactive(user_query=None, profile_summary=None, anomalies=None):
    print("\n🧠 [Insight Agent Activated: Interactive Mode]\n")

    if user_query and "re-analyze" in user_query.lower():
        print("🔄 Re-running data profiling and anomaly detection...")
        profile_summary, anomalies = initialize_data_insights()

    if user_query:
        print("\n💬 Checking if the query is clear...\n")
        try:
            clarity_check = globals.agent.run(
                input=f"""
                A user asked: '{user_query}'.
                Is this question clear enough to answer from the dataset?
                Respond only with: Yes or No
                """,
                handle_parsing_errors=False  # <-- important
            ).strip().lower()
        except Exception as e:
            clarity_check = "no"
            print(f"Error during clarity check: {e}")

        print(f"Model response: {clarity_check}")

        if "yes" in clarity_check:
            try:
                insight = globals.agent.run(f"Find insights based on this question: {user_query}")
            except Exception as e:
                insight = f"Could not generate insight. Error: {e}"

            try:
                context_explanation = globals.agent.run("Explain why this insight matters to a non-technical audience.")
            except Exception as e:
                context_explanation = f"Could not explain context. Error: {e}"

            try:
                visual_suggestion = globals.agent.run("Suggest simple visuals to explain the insight.")
            except Exception as e:
                visual_suggestion = f"Could not suggest visuals. Error: {e}"

            try:
                action_plan = globals.agent.run("What are 2–3 actions someone could take based on this?")
            except Exception as e:
                action_plan = f"Could not suggest actions. Error: {e}"

            try:
                next_questions = globals.agent.run(
                    "Give 2–3 next questions a non-technical person might ask based on this insight. Use bullet points."
                )
            except Exception as e:
                next_questions = f"Could not generate follow-up questions. Error: {e}"

            return f"""
📌 USER QUERY: {user_query}

🔍 INSIGHT:
{insight}

💡 WHY IT MATTERS:
{context_explanation}

📈 SUGGESTED VISUALS:
{visual_suggestion}

✅ ACTION PLAN:
{action_plan}

🤔 WOULD YOU LIKE TO:
{next_questions}

🚨 ANOMALIES:
{anomalies}
"""

#%%
def get_voice_input():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    print("🎤 Listening... Speak your query after the beep!")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Beep!")
        audio = recognizer.listen(source)
    
    try:
        query = recognizer.recognize_google(audio)
        print(f"🗣️ You said: {query}")
        return query
    except sr.UnknownValueError:
        print("❌ Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"❌ Speech recognition error: {e}")
        return None

#%%
def get_user_input():
    # Get user input from the console
    return input("Please enter your query: ")



def extract_column_names(query):
    uploaded_data=globals.global_df
    df_columns = uploaded_data.columns.tolist()
    doc = nlp(query.lower())
    # Extract nouns and proper nouns as potential column names
    potential_columns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    # Match potential columns with actual dataset columns
    matched_columns = [col for col in df_columns if col.lower() in potential_columns]
    return matched_columns if matched_columns else None
#%%
def process_query(query,mode="basic"):
    print(query+" "+mode)
    if globals.global_df is None:
        return "Please upload a dataset first."
    # Start timer
    start_time = time.time()
    try:
        # Basic Mode: Handle simple static queries
        if mode == "basic":
            intent = classify_intent(query)
            print(intent)
            # Handle the different intents in basic mode
            if intent == "data cleaning":
                print(1)
                method = extract_cleaning_method(query)
                print(2)
                response = data_cleaning(method=method)  # Pass df to data_cleaning
            elif intent == "eda visualization":
                method, cols = extract_eda_info(query)
                vis_result = eda_visualization(method=method, cols=cols)
                response = (vis_result["message"], vis_result["plots"])  # Pass df to eda_visualization
            elif intent == "insights":
                uploaded_data=globals.global_df
                clean_df, logs = dataCleanPipeLine(globals.global_df, None)
                response = insights(uploaded_data, clean_df, columns=extract_column_names(query))
            elif intent == "anomalies":
                response = anomalies()  # Pass df to anomalies
            elif intent == "summary":
                response = summary_func()  # Pass df to summary
            elif intent == "describe columns":
                response = describe_columns()  # Pass df to describe_columns
            else:
                response = enhanced_query(query)

        # Insight Agent Mode: Handle dynamic, interactive queries
        elif mode == "insight_agent":
            response = insight_agent_interactive(user_query=query)  # Always interactive for insight agent

        else:
            response = "❌ Invalid mode selected. Use 'basic' or 'insight_agent'."

    except RateLimitError as e:
        print(f"❌ Rate limit reached. Please try again in 9 minutes. Error: {e}")
        response = "❌ Rate limit reached. Please try again in 9 minutes."

    # End timer
    end_time = time.time()
    print(f"Response generation time: {end_time - start_time} seconds")

    # Store the current user query and agent response in memory
    store_in_memory(query, response)

    return response

#%%

# Simulated session state (normally this would be handled by gr.State or a real session manager)
session_state = {
    "profile_summary": None,
    "anomalies": None
}

# Example to test the Insight Agent mode
def test_query(user_query):
    """"
    global session_state

    # Run profiling and anomaly detection only once
    if session_state["profile_summary"] is None or session_state["anomalies"] is None:
        session_state["profile_summary"], session_state["anomalies"] = initialize_data_insights()

    # Use the insight function
    response = insight_agent_interactive(
        user_query=user_query,
        profile_summary=session_state["profile_summary"],
        anomalies=session_state["anomalies"]
    )
    
    print(response)
    """""


    """""
#test_query("Clean the data and impute with knn")
response=process_query("visualize scatter plot")
print(response)
"""""
# Generate HTML report
from reportlab.lib.units import inch

import os
import pandas as pd
from jinja2 import Template
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from PIL import Image
import base64
import io




# Function to convert image to Base64 string
def image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


# Function to generate visualizations and save them as images
def generate_html_visualizations(uploaded_data: pd.DataFrame):
    output_dir = "visualizations/"
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations using dataVisualizationPipeline (replace with actual pipeline)
    result = dataVisualizationPipeline(uploaded_data)

    # Access the plots from the returned dictionary
    plots = result["plots"]

    # List to store Base64 encoded images
    base64_images = []
    for i, plot in enumerate(plots):
        if isinstance(plot, str) and os.path.exists(plot):  # Check if it's a valid file path
            # Convert image to Base64
            base64_image = image_to_base64(plot)
            base64_images.append(base64_image)
            print(f"Converted plot {i} to Base64.")
        else:
            print(f"Skipping non-image object: {plot}")

    return base64_images


def generate_html_report(uploaded_data, summary: str, insights: str, base64_images: list, output_path: str):
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hotel Reviews Data Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px auto;
                max-width: 900px;
                line-height: 1.6;
                color: #333;
                padding: 20px;
            }
            h1, h2 {
                color: #2c3e50;
                text-align: center;
            }
            img {
                max-width: 100%;
                margin-bottom: 20px;
                display: block;
                margin-left: auto;
                margin-right: auto;
            }
            p {
                text-align: justify;
            }
            .summary, .insights {
                margin-bottom: 40px;
            }
            .visualization {
                text-align: center;
            }
        </style>
    </head>
    <body>
        <h1>Hotel Reviews Data Report</h1>

        <!-- Summary Section -->
        <div class="summary">
            <h2>Summary</h2>
            <p>{{ summary }}</p>
        </div>

        <!-- Insights Section -->
        <div class="insights">
            <h2>Insights</h2>
            <p>{{ insights }}</p>
        </div>

        <h2>Visualizations</h2>
        {% for base64_image in base64_images %}
            <div class="visualization">
                <img src="data:image/png;base64,{{ base64_image }}" alt="Visualization">
            </div>
        {% endfor %}
    </body>
    </html>
    """

    # Render the HTML template with the summary, insights, and Base64 images
    template = Template(html_template)
    rendered_html = template.render(summary=summary, insights=insights, base64_images=base64_images)

    # Save the HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)
    print(f"HTML report saved at {output_path}")


def generate_pdf_visualizations(uploaded_data: pd.DataFrame):
    output_dir = "visualizations/"
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations using dataVisualizationPipeline (replace with actual pipeline)
    result = dataVisualizationPipeline(uploaded_data)

    # Access the plots from the returned dictionary
    plots = result["plots"]

    # List to store image file paths
    image_paths = []
    for i, plot in enumerate(plots):
        if isinstance(plot, str) and os.path.exists(plot):  # Check if it's a valid file path
            # Open the image using PIL
            img = Image.open(plot)
            image_path = os.path.join(output_dir, f"plot_{i}.png")
            img.save(image_path)  # Save the image as PNG
            image_paths.append(image_path)
            print(f"Saved plot: {image_path}")
        else:
            print(f"Skipping non-image object: {plot}")

    return image_paths


# Function to generate PDF report with summary and insights
def generate_pdf_report(uploaded_data, summary: str, insights: str, visualization_images: list, output_path: str):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    margin = inch
    y = height - margin

    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2.0, y, "Hotel Reviews Data Report")
    y -= 30

    # Add summary
    c.setFont("Helvetica", 12)
    c.drawString(margin, y, "Summary:")
    y -= 20
    c.drawString(margin, y, summary)
    y -= 40

    # Add insights
    c.drawString(margin, y, "Insights:")
    y -= 20
    c.drawString(margin, y, insights)
    y -= 40

    # Add visualizations (charts), one per page with proper sizing
    for image_path in visualization_images:
        if os.path.exists(image_path):
            print(f"Adding image to PDF: {image_path}")
            c.showPage()  # Start a new page for each chart
            img = Image.open(image_path)
            img_width, img_height = img.size

            # Resize the image proportionally to fit within the page, maintaining aspect ratio
            max_width = width - 2 * margin
            max_height = height - 2 * margin

            # Scale image to fit within the max width/height while maintaining aspect ratio
            scale_factor = min(max_width / img_width, max_height / img_height)
            img_width = int(img_width * scale_factor)
            img_height = int(img_height * scale_factor)

            # Center the image horizontally and vertically
            x_offset = (width - img_width) / 2
            y_offset = (height - img_height) / 2

            # Draw the image on the canvas
            c.drawImage(image_path, x_offset, y_offset, img_width, img_height)
            y -= img_height + margin  # Adjust the y position for the next image

    # Save the PDF
    c.save()
    print(f"PDF report saved at {output_path}")


# Main report export function
def export_report( format: str, output_path: str):
    summary_prompt = """
    Provide a beginner-friendly summary of this dataset, including:
    - Number of rows and columns
    - Types of data (e.g., text, numbers, categories)
    - Any columns with missing or unusual values
    Present it clearly using bullet points.
    """
    insights_prompt = """
    Analyze this hotel reviews dataset and provide useful insights in simple terms:
    - Identify trends and patterns
    - Highlight any outliers or anomalies
    - Mention correlations between features if any
    - Suggest practical actions based on the data
    Keep it concise and in bullet points.
    """

    summary = globals.agent.run(summary_prompt)
    insights = globals.agent.run(insights_prompt)
    _, visualization_images = dataVisualizationPipeline(globals.global_df)

    if format.lower() == "html":
        base64_images = generate_html_visualizations(globals.global_df)
        # Generate and save the HTML report
        generate_html_report(globals.global_df, summary, insights, base64_images, output_path)
        # generate_html_report(uploaded_dataset, summary, insights, visualization_images, output_path)
    elif format.lower() == "pdf":
        visualization_images = generate_pdf_visualizations(globals.global_df)
        generate_pdf_report(globals.global_df, summary, insights, visualization_images, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Choose 'pdf' or 'html'.")