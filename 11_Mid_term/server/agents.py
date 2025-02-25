from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.tools import ElevenLabsText2SpeechTool
import tiktoken
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv
from datetime import datetime
import logging
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agents.log'),
        logging.StreamHandler()
    ]
)

# Create loggers for each agent
extractor_logger = logging.getLogger('ExtractorAgent')
skeptic_logger = logging.getLogger('SkepticAgent')
believer_logger = logging.getLogger('BelieverAgent')
supervisor_logger = logging.getLogger('SupervisorAgent')
podcast_logger = logging.getLogger('PodcastProducerAgent')

# Load environment variables
load_dotenv()

# Get API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not eleven_api_key:
    raise ValueError("ELEVEN_API_KEY not found in environment variables")

# Initialize the base LLM
base_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=openai_api_key
)

class ExtractorOutput(BaseModel):
    content: str = Field(description="The extracted and refined query")
    key_points: List[str] = Field(description="Key points extracted from the query")

class AgentResponse(BaseModel):
    content: str = Field(description="The agent's response")
    chunks: Optional[List[Dict[str, str]]] = Field(description="Relevant context chunks used", default=None)

class SupervisorOutput(BaseModel):
    content: str = Field(description="The supervisor's analysis")
    chunks: Dict[str, List[str]] = Field(description="Quadrant-based chunks of the analysis")

class PodcastOutput(BaseModel):
    title: str = Field(description="Title of the podcast episode")
    description: str = Field(description="Description of the episode")
    script: str = Field(description="The podcast script")
    summary: str = Field(description="A brief summary of the episode")
    duration_minutes: int = Field(description="Estimated duration in minutes")

class ExtractorAgent:
    def __init__(self, tavily_api_key: str):
        self.search_tool = TavilySearchResults(
            api_key=tavily_api_key,
            max_results=5
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert information extractor. Your role is to:
            1. Extract relevant information from search results
            2. Organize the information in a clear, structured way
            3. Focus on factual, verifiable information
            4. Cite sources when possible"""),
            ("human", "{input}")
        ])
        self.chain = self.prompt | base_llm

    async def __call__(self, query: str) -> Dict[str, Any]:
        try:
            # Log the incoming query
            extractor_logger.info(f"Processing query: {query}")
            
            try:
                # Search using Tavily
                search_results = await self.search_tool.ainvoke(query)
                extractor_logger.debug(f"Search results: {json.dumps(search_results, indent=2)}")
            except Exception as e:
                extractor_logger.error(f"Error in Tavily search: {str(e)}", exc_info=True)
                raise Exception(f"Tavily search failed: {str(e)}")
            
            # Format the results
            if isinstance(search_results, list):
                formatted_results = f"Search results for: {query}\n" + "\n".join(
                    [str(result) for result in search_results]
                )
            else:
                formatted_results = f"Search results for: {query}\n{search_results}"
            
            try:
                # Generate response using the chain
                response = await self.chain.ainvoke({"input": formatted_results})
                extractor_logger.info(f"Generated response: {response.content}")
            except Exception as e:
                extractor_logger.error(f"Error in LLM chain: {str(e)}", exc_info=True)
                raise Exception(f"LLM chain failed: {str(e)}")
            
            return {
                "type": "extractor",
                "content": response.content,
                "raw_results": search_results
            }
        except Exception as e:
            extractor_logger.error(f"Error in ExtractorAgent: {str(e)}", exc_info=True)
            raise Exception(f"Error in extractor: {str(e)}")

class SkepticAgent:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a critical thinker engaging in a thoughtful discussion. While maintaining a balanced perspective, you should:
            - Analyze potential challenges and limitations
            - Consider real-world implications
            - Support arguments with evidence and examples
            - Maintain a respectful and constructive tone
            - Raise important considerations
            
            If provided with context information, use it to inform your response while maintaining your analytical perspective.
            Focus on examining risks and important questions from the context.
            
            Keep your responses concise and focused on the topic at hand."""),
            ("human", """Context information:
            {chunks}
            
            Question/Topic:
            {input}""")
        ])
        self.chain = self.prompt | base_llm

    async def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        skeptic_logger.info(f"Processing input: {input_data['content']}")
        chunks = input_data.get("chunks", [])
        chunks_text = "\n".join(chunks) if chunks else "No additional context provided."
        
        response = await self.chain.ainvoke({
            "input": input_data["content"],
            "chunks": chunks_text
        })
        skeptic_logger.info(f"Generated response: {response.content}")
        return {"type": "skeptic", "content": response.content}

class BelieverAgent:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an optimistic thinker engaging in a thoughtful discussion. While maintaining a balanced perspective, you should:
            - Highlight opportunities and potential benefits
            - Share innovative solutions and possibilities
            - Support arguments with evidence and examples
            - Maintain a constructive and forward-thinking tone
            - Build on existing ideas positively
            
            If provided with context information, use it to inform your response while maintaining your optimistic perspective.
            Focus on opportunities and solutions from the context.
            
            Keep your responses concise and focused on the topic at hand."""),
            ("human", """Context information:
            {chunks}
            
            Question/Topic:
            {input}""")
        ])
        self.chain = self.prompt | base_llm

    async def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        believer_logger.info(f"Processing input: {input_data['content']}")
        chunks = input_data.get("chunks", [])
        chunks_text = "\n".join(chunks) if chunks else "No additional context provided."
        
        response = await self.chain.ainvoke({
            "input": input_data["content"],
            "chunks": chunks_text
        })
        believer_logger.info(f"Generated response: {response.content}")
        return {"type": "believer", "content": response.content}

class SupervisorAgent:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a balanced supervisor. Your role is to:
            1. Analyze inputs from all agents
            2. Identify key points and insights
            3. Balance different perspectives
            4. Synthesize a comprehensive view
            5. Provide clear, actionable conclusions
            
            Organize your response into these sections:
            - Opportunities: Key possibilities and positive aspects
            - Risks: Important challenges and concerns
            - Questions: Critical questions to consider
            - Solutions: Potential ways forward
            
            Focus on creating a balanced, well-reasoned synthesis of all viewpoints.
            Keep your words to 100 words or less."""),
            ("human", "Analyze the following perspectives:\n\nExtractor: {extractor_content}\n\nSkeptic: {skeptic_content}\n\nBeliever: {believer_content}")
        ])
        self.chain = self.prompt | base_llm

    async def __call__(self, agent_responses: Dict[str, Any]) -> Dict[str, Any]:
        supervisor_logger.info("Processing agent responses:")
        supervisor_logger.info(f"Extractor: {agent_responses['extractor']['content']}")
        supervisor_logger.info(f"Skeptic: {agent_responses['skeptic']['content']}")
        supervisor_logger.info(f"Believer: {agent_responses['believer']['content']}")
        
        # Process supervisor's analysis
        response = await self.chain.ainvoke({
            "extractor_content": agent_responses["extractor"]["content"],
            "skeptic_content": agent_responses["skeptic"]["content"],
            "believer_content": agent_responses["believer"]["content"]
        })
        
        supervisor_logger.info(f"Generated analysis: {response.content}")
        
        # Parse the response into sections
        content = response.content
        sections = {
            "opportunities": [],
            "risks": [],
            "questions": [],
            "solutions": []
        }
        
        # Simple parsing of the content into sections
        current_section = None
        for line in content.split('\n'):
            line = line.strip()
            if line.lower().startswith('opportunities:'):
                current_section = "opportunities"
            elif line.lower().startswith('risks:'):
                current_section = "risks"
            elif line.lower().startswith('questions:'):
                current_section = "questions"
            elif line.lower().startswith('solutions:'):
                current_section = "solutions"
            elif line and current_section:
                sections[current_section].append(line)
        
        return {
            "type": "supervisor",
            "content": response.content,
            "chunks": sections
        }

class PodcastProducerAgent:
    def __init__(self):
        podcast_logger.info("Initializing PodcastProducerAgent")
        
        # Initialize the agent with a lower temperature for more consistent output
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=openai_api_key
        )
        
        # Create audio storage directory if it doesn't exist
        self.audio_dir = os.path.join(os.path.dirname(__file__), "audio_storage")
        os.makedirs(self.audio_dir, exist_ok=True)
        podcast_logger.info(f"Audio directory: {self.audio_dir}")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert podcast producer. Create a single, cohesive podcast script that:
            1. Introduces the topic clearly
            2. Presents a balanced debate between perspectives
            3. Incorporates key insights from the supervisor's analysis:
               - Opportunities and positive aspects
               - Risks and challenges
               - Key questions to consider
               - Potential solutions
            4. Prioritizes content based on quadrant analysis:
               - Important & Urgent: Address first and emphasize
               - Important & Not Urgent: Cover thoroughly but with less urgency
               - Not Important & Urgent: Mention briefly if relevant
               - Not Important & Not Urgent: Include only if adds value
            5. Maintains natural conversation flow with clear speaker transitions
            6. Concludes with actionable takeaways
            
            Keep the tone professional but conversational. Format the script with clear speaker indicators and natural pauses."""),
            ("human", """Create a podcast script from this content:

            Topic: {user_query}

            Debate Content:
            {debate_content}

            Supervisor's Analysis:
            {supervisor_content}

            Quadrant Analysis:
            Important & Urgent:
            {important_urgent}

            Important & Not Urgent:
            {important_not_urgent}

            Not Important & Urgent:
            {not_important_urgent}

            Not Important & Not Urgent:
            {not_important_not_urgent}""")
        ])
        self.chain = self.prompt | llm

        # Metadata prompt for categorization
        self.metadata_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the debate and provide:
            1. A category (single word: technology/science/society/politics/economics/culture)
            2. A short description (3-4 words) of the main topic
            Format: category|short_description"""),
            ("human", "{content}")
        ])
        self.metadata_chain = self.metadata_prompt | llm

    async def __call__(self, debate_history: list, supervisor_notes: list, user_query: str, supervisor_chunks: dict, quadrant_analysis: dict) -> Dict[str, Any]:
        try:
            podcast_logger.info("Starting podcast production")
            
            # Format the debate content
            debate_content = "\n\n".join([
                f"{entry['speaker']}: {entry['content']}" 
                for entry in debate_history
            ])
            
            # Get the latest supervisor analysis
            supervisor_content = supervisor_notes[-1] if supervisor_notes else ""
            
            # Format quadrant content
            important_urgent = "\n".join(quadrant_analysis.get("important_urgent", []))
            important_not_urgent = "\n".join(quadrant_analysis.get("important_not_urgent", []))
            not_important_urgent = "\n".join(quadrant_analysis.get("not_important_urgent", []))
            not_important_not_urgent = "\n".join(quadrant_analysis.get("not_important_not_urgent", []))
            
            # Generate the podcast script
            script_response = await self.chain.ainvoke({
                "user_query": user_query,
                "debate_content": debate_content,
                "supervisor_content": supervisor_content,
                "important_urgent": important_urgent,
                "important_not_urgent": important_not_urgent,
                "not_important_urgent": not_important_urgent,
                "not_important_not_urgent": not_important_not_urgent
            })
            
            # Get metadata for the podcast
            metadata_response = await self.metadata_chain.ainvoke({
                "content": script_response.content
            })
            category, description = metadata_response.content.strip().split("|")
            
            # Clean up filename components
            clean_query = user_query.lower().replace(" ", "_")[:30]
            clean_description = description.lower().replace(" ", "_")
            clean_category = category.lower().strip()
            
            try:
                # Create a single filename with hyphens separating main components
                filename = f"{clean_query}-{clean_description}-{clean_category}.mp3"
                filepath = os.path.join(self.audio_dir, filename)
                
                # Generate audio file
                from gtts import gTTS
                tts = gTTS(text=script_response.content, lang='en')
                tts.save(filepath)
                
                podcast_logger.info(f"Successfully saved audio file: {filepath}")
                
                return {
                    "type": "podcast",
                    "content": script_response.content,
                    "audio_file": filename,
                    "category": clean_category,
                    "description": description,
                    "title": f"Debate: {description.title()}"
                }
            except Exception as e:
                podcast_logger.error(f"Error in audio generation: {str(e)}", exc_info=True)
                return {
                    "type": "podcast",
                    "content": script_response.content,
                    "error": f"Audio generation failed: {str(e)}"
                }
                
        except Exception as e:
            podcast_logger.error(f"Error in podcast production: {str(e)}", exc_info=True)
            return {
                "type": "podcast",
                "error": f"Podcast production failed: {str(e)}"
            }

def create_agents(tavily_api_key: str) -> Dict[str, Any]:
    # Initialize all agents
    extractor = ExtractorAgent(tavily_api_key)
    believer = BelieverAgent()
    skeptic = SkepticAgent()
    supervisor = SupervisorAgent()
    podcast_producer = PodcastProducerAgent()
    
    return {
        "extractor": extractor,
        "believer": believer,
        "skeptic": skeptic,
        "supervisor": supervisor,
        "podcast_producer": podcast_producer
    } 