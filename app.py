import os
from dotenv import load_dotenv

load_dotenv()
from langtrace_python_sdk import langtrace
#langtrace_api_key = 
langtrace.init(api_key=os.getenv('LANGTRACE_API_KEY'))

from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel
import os
from flask import Flask, request, jsonify
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai_tools import DirectoryReadTool, FileReadTool, WebsiteSearchTool
from crewai.tools import BaseTool
from pydantic import Field
from dotenv import load_dotenv  # Import dotenv
from exa_py import Exa
from datetime import datetime, timedelta
import uuid



# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)

# Retrieve the API key from the environment
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

#tavily_api_key = os.getenv('TAVILY_API_KEY')

tool = WebsiteSearchTool(
    config=dict(
        llm=dict(
            provider="groq", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="groq/gemma2-9b-it",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="ollama", # or openai, ollama, ...
            config=dict(
                model="all-minilm:latest",
                #task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)


# Initialize the language model (Groq) with the proper settings
llm = ChatOpenAI(model='gpt-4o-mini') 




class SearchAndContents(BaseTool):
    name: str = "Search and Contents Tool"
    description: str = (
        "Searches the web based on a search query for the latest results. Results are only from the last week. Uses the Exa API. This also returns the contents of the search results."
    )

    def _run(self, search_query: str) -> str:

        exa = Exa(api_key=os.getenv("EXA_API_KEY"))

        one_week_ago = datetime.now() - timedelta(days=7)
        date_cutoff = one_week_ago.strftime("%Y-%m-%d")

        search_results = exa.search_and_contents(
            query=search_query,
            use_autoprompt=True,
            start_published_date=date_cutoff,
            text={"include_html_tags": False, "max_characters": 8000},
        )

        return search_results



docs_tool = DirectoryReadTool(directory='./newsletter-posts')
file_tool = FileReadTool()
search_tool = SearchAndContents()
web_rag_tool = tool

# Agents
researcher = Agent(
    llm=llm,
    role="Partnership Researcher",
    goal="Identify noteworthy partnerships from startups, enterprises, and AI sectors based on the user input: {user_input}.",
    backstory="An expert researcher tracking global partnerships to highlight the most impactful collaborations based on the user input: {user_input}. "
              "Your role is to provide comprehensive research that helps inform decision-making and serves as the foundation for the subsequent agents. "
              "You are detail-oriented and provide clear, concise summaries with reliable sources, including the significance of each partnership and its broader industry impact.",
    allow_delegation=False,
    tools=[search_tool],
    verbose=True
)


partnership_expert = Agent(
    llm=llm,
    role="Partnership Expert",
    goal="Add bespoke and unique partnership insights to the research provided by the Partnership Researcher.",
    backstory="An experienced partnership strategist with deep knowledge of the art and science of building collaborations. "
              "You provide analysis and insights into the nuances of partnerships: the legal frameworks, business development strategies, win-win-win models, and the complexities of design, training, and setup. "
              "Your expertise adds depth to the partnership summaries, emphasizing why each collaboration works, the challenges overcome, and its significance in the broader context of partnerships.",
    allow_delegation=False,
    tools=[search_tool],
    verbose=True
)



writer = Agent(
    llm=llm,
    role="Newsletter Content Creator",
    goal="You transform the insights from the Partnership Expert into reader-friendly content"
         "Write engaging and concise partnership highlights for the newsletter about the user input: {user_input}.",
    backstory="A creative writer skilled in crafting concise, informative, and compelling newsletter sections. "
              "You transform the insights from the Partnership Expert into reader-friendly content. "
              "You provide analysis and insights into the nuances of partnerships: the legal frameworks, business development strategies, win-win-win models, and the complexities of design, training, and setup. "
              "Your expertise adds depth to the partnership summaries, emphasizing why each collaboration works, the challenges overcome, and its significance in the broader context of partnerships."
              "Your writing highlights the significance of each partnership, focusing on its innovation, relevance, and potential impact while keeping the tone professional, engaging, and aligned with the newsletter's goals.",
    allow_delegation=False,
    tools=[],
    verbose=True
)



editor = Agent(
    llm=llm,
    role="Newsletter Editor",
    goal="Proofread, refine, and structure the newsletter to ensure it is ready for publication.",
    backstory="A meticulous editor responsible for reviewing and polishing the newsletter content from the Newsletter Content Creator. "
              "You provide analysis and insights into the nuances of partnerships: the legal frameworks, business development strategies, win-win-win models, and the complexities of design, training, and setup. "
              "Your expertise adds depth to the partnership summaries, emphasizing why each collaboration works, the challenges overcome, and its significance in the broader context of partnerships."
              "Your focus is on improving readability, ensuring error-free copy, enhancing structure, and aligning the tone with the newsletter's vision. "
              "You ensure the newsletter engages the audience, flows logically, and highlights key insights effectively."
              "Include valid website url links to each partnerships that was done by the Partnership Researcher.",
    allow_delegation=False,
    tools=[docs_tool, file_tool],
    verbose=True
)


# Tasks
research_task = Task(
    description=(
        "Research recent partnerships across industries with a focus on startups, enterprises, "
        "and AI-based collaborations. Identify the 'Partnership of the Week'."
    ),
    expected_output="A list of noteworthy partnerships with sources, categorized as 'Startup Partnerships', "
                    "'Enterprise Partnerships', and 'AI in Partnerships'.",
    agent=researcher
)


partnership_insights_task = Task(
    description=(
        "Analyze the research provided by the Researcher and add bespoke insights into the art and science of partnerships. "
        "Focus on highlighting the strategic, legal, and operational complexities of each partnership, including win-win-win models, "
        "business development strategies, and training or setup considerations."
    ),
    expected_output="A detailed analysis of each partnership with unique insights on the factors that contributed to its success, "
                    "the challenges overcome, and its broader implications for the industry. Include commentary on the partnership's "
                    "design, legal framework, and strategic impact.",
    agent=partnership_expert
)


write_task = Task(
    description=(
        "Create concise and engaging content for each partnership category, including a short summary for: \n"
        "- Partnership of the Week\n"
        "- Startup Partnerships\n"
        "- Enterprise Partnerships\n"
        "- AI in Partnerships"
    ),
    expected_output="A word-formatted newsletter with categorized sections and summaries for each partnership.",
    agent=writer
)

edit_task = Task(
    description="Proofread and structure the newsletter to ensure it is publication-ready."
                "Should have a opening intro blurb, ensure it always say Hi to readers."
                "Interesting things that happened in the week/tech ecosystem"
                "short wrapup of the week in general"
                "not just partnerships but what's big changes and developents that have happened"
                "Very long. it should be more than 3/4 sentences"
                "Include valid website url links to each partnerships that was done by the Partnership Researcher."
                "It should always start with Welcome to QTA weekly digest on AI Partnerships"
                "It should end with"
                "Written by Ebuka",
    expected_output="A finalized newsletter, ready for weekly publication."
                    "Each section should be captivating and have 5 paragraphs",
    agent=editor,
    output_file=f'newsletter-posts/new_post_{uuid.uuid4().hex}.md'
)

# Crew Initialization
crew = Crew(
    agents=[researcher, partnership_expert, writer, editor],
    tasks=[research_task, partnership_insights_task, write_task, edit_task],
    verbose=True,
    planning=True
)

# API Endpoint for processing user input
@app.route('/process_partnerships', methods=['POST'])
def process_partnerships():
    try:
        # Get the user input from the request body
        user_input = request.json.get('user_input', '')

        # Ensure input is not empty
        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        # Trigger the crew tasks with the provided input
        crew_output = crew.kickoff(inputs={"user_input": user_input})

        # Convert the crew output to a dictionary
        output_dict = crew_output.to_dict()

        # Return the result as JSON
        return jsonify({"message": "Task successfully processed", "result": output_dict}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)