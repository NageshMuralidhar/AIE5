from typing import Dict, Any, List, Annotated, TypedDict, Union, Optional
from langgraph.graph import Graph, END
from agents import create_agents
import os
from dotenv import load_dotenv
import json
import uuid

# Load environment variables
load_dotenv()

# Create transcripts directory if it doesn't exist
TRANSCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "transcripts")
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
TRANSCRIPTS_FILE = os.path.join(TRANSCRIPTS_DIR, "podcasts.json")

def save_transcript(podcast_script: str, user_query: str) -> None:
    """Save podcast transcript to JSON file."""
    # Create new transcript entry
    transcript = {
        "id": str(uuid.uuid4()),
        "podcastScript": podcast_script,
        "topic": user_query
    }
    
    try:
        # Load existing transcripts
        if os.path.exists(TRANSCRIPTS_FILE):
            with open(TRANSCRIPTS_FILE, 'r') as f:
                transcripts = json.load(f)
        else:
            transcripts = []
        
        # Append new transcript
        transcripts.append(transcript)
        
        # Save updated transcripts
        with open(TRANSCRIPTS_FILE, 'w') as f:
            json.dump(transcripts, f, indent=2)
            
    except Exception as e:
        print(f"Error saving transcript: {str(e)}")

class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    current_agent: str
    debate_turns: int
    extractor_data: Dict[str, Any]
    debate_history: List[Dict[str, Any]]
    supervisor_notes: List[str]
    supervisor_chunks: List[Dict[str, List[str]]]
    final_podcast: Dict[str, Any]
    agent_type: str
    context: Optional[Dict[str, Any]]

def create_workflow(tavily_api_key: str):
    # Initialize all agents
    agents = create_agents(tavily_api_key)
    
    # Create the graph
    workflow = Graph()
    
    # Define the extractor node function
    async def run_extractor(state: AgentState) -> Dict[str, Any]:
        query = state["messages"][-1]["content"]
        print(f"Extractor processing query: {query}")
        
        try:
            response = await agents["extractor"](query)
            print(f"Extractor response: {response}")
            
            # Update state
            state["extractor_data"] = response
            
            # Get initial supervisor analysis
            supervisor_analysis = await agents["supervisor"]({
                "extractor": response,
                "skeptic": {"content": "Not started"},
                "believer": {"content": "Not started"}
            })
            print(f"Initial supervisor analysis: {supervisor_analysis}")
            
            state["supervisor_notes"].append(supervisor_analysis["content"])
            state["supervisor_chunks"].append(supervisor_analysis.get("chunks", {}))
            
            # Move to debate phase
            state["current_agent"] = "debate"
            return state
        except Exception as e:
            print(f"Error in extractor: {str(e)}")
            raise Exception(f"Error in extractor: {str(e)}")

    # Define the debate node function
    async def run_debate(state: AgentState) -> Dict[str, Any]:
        print(f"Debate turn {state['debate_turns']}")
        
        try:
            if state["debate_turns"] == 0:
                # First turn: both agents respond to extractor
                print("Starting first debate turn")
                
                # If we have context, use it to inform the agents' responses
                context = state.get("context", {})
                agent_chunks = context.get("agent_chunks", []) if context else []
                
                # Create context-aware input for agents
                context_input = {
                    "content": state["extractor_data"]["content"],
                    "chunks": agent_chunks
                }
                
                skeptic_response = await agents["skeptic"](context_input)
                believer_response = await agents["believer"](context_input)
                
                state["debate_history"].extend([
                    {"speaker": "skeptic", "content": skeptic_response["content"]},
                    {"speaker": "believer", "content": believer_response["content"]}
                ])
                print(f"First turn responses added: {state['debate_history'][-2:]}")
            else:
                # Alternating responses based on agent type if specified
                if state["agent_type"] in ["believer", "skeptic"]:
                    current_speaker = state["agent_type"]
                else:
                    # Default alternating behavior
                    last_speaker = state["debate_history"][-1]["speaker"]
                    current_speaker = "believer" if last_speaker == "skeptic" else "skeptic"
                
                print(f"Processing response for {current_speaker}")
                
                # Create context-aware input
                context = state.get("context", {})
                agent_chunks = context.get("agent_chunks", []) if context else []
                context_input = {
                    "content": state["debate_history"][-1]["content"],
                    "chunks": agent_chunks
                }
                
                response = await agents[current_speaker](context_input)
                
                state["debate_history"].append({
                    "speaker": current_speaker,
                    "content": response["content"]
                })
                print(f"Added response: {state['debate_history'][-1]}")
            
            # Add supervisor note and chunks
            supervisor_analysis = await agents["supervisor"]({
                "extractor": state["extractor_data"],
                "skeptic": {"content": state["debate_history"][-1]["content"]},
                "believer": {"content": state["debate_history"][-2]["content"] if len(state["debate_history"]) > 1 else "Not started"}
            })
            print(f"Supervisor analysis: {supervisor_analysis}")
            
            state["supervisor_notes"].append(supervisor_analysis["content"])
            state["supervisor_chunks"].append(supervisor_analysis.get("chunks", {}))
            
            state["debate_turns"] += 1
            print(f"Debate turn {state['debate_turns']} completed")
            
            # End the workflow after 2 debate turns
            if state["debate_turns"] >= 2:
                state["current_agent"] = "podcast"
                print("Moving to podcast production")
            
            return state
        except Exception as e:
            print(f"Error in debate: {str(e)}")
            raise Exception(f"Error in debate: {str(e)}")

    async def run_podcast_producer(state: AgentState) -> Dict[str, Any]:
        print("Starting podcast production")
        
        try:
            # Create podcast from debate
            podcast_result = await agents["podcast_producer"](
                state["debate_history"],
                state["supervisor_notes"],
                state["messages"][-1]["content"],  # Pass the original user query
                state["supervisor_chunks"],
                {}  # Empty quadrant analysis since we removed storage manager
            )
            print(f"Podcast production result: {podcast_result}")
            
            # Save transcript to JSON file
            save_transcript(
                podcast_script=podcast_result["content"],
                user_query=state["messages"][-1]["content"]
            )
            
            # Store the result
            state["final_podcast"] = podcast_result
            
            # End the workflow
            state["current_agent"] = END
            return state
        except Exception as e:
            print(f"Error in podcast production: {str(e)}")
            raise Exception(f"Error in podcast production: {str(e)}")

    # Add nodes to the graph
    workflow.add_node("extractor", run_extractor)
    workflow.add_node("debate", run_debate)
    workflow.add_node("podcast", run_podcast_producer)

    # Set the entry point
    workflow.set_entry_point("extractor")

    # Add edges
    workflow.add_edge("extractor", "debate")
    
    # Add conditional edges for debate
    workflow.add_conditional_edges(
        "debate",
        lambda x: "podcast" if x["debate_turns"] >= 2 else "debate"
    )
    
    # Add edge from podcast to end
    workflow.add_edge("podcast", END)

    # Compile the graph
    return workflow.compile()

async def run_workflow(
    graph: Graph, 
    query: str, 
    agent_type: str = "believer",
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run the workflow with a given query."""
    # Initialize the state
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "current_agent": "extractor",
        "debate_turns": 0,
        "extractor_data": {},
        "debate_history": [],
        "supervisor_notes": [],
        "supervisor_chunks": [],
        "final_podcast": {},
        "agent_type": agent_type,
        "context": context
    }
    
    # Run the graph
    result = await graph.ainvoke(initial_state)
    
    return {
        "debate_history": result["debate_history"],
        "supervisor_notes": result["supervisor_notes"],
        "supervisor_chunks": result["supervisor_chunks"],
        "extractor_data": result["extractor_data"],
        "final_podcast": result["final_podcast"]
    } 