import asyncio
from dotenv import load_dotenv
import os
from workflow import create_workflow, run_workflow, TRANSCRIPTS_FILE
import json
from datetime import datetime
import traceback

# Load environment variables
load_dotenv()

def log_step(step: str, details: str = None):
    """Print a formatted step log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] 🔄 {step}")
    if details:
        print(f"    {details}")

def log_agent(agent: str, message: str):
    """Print a formatted agent message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    agent_icons = {
        "extractor": "🔍",
        "skeptic": "🤔",
        "believer": "💡", 
        "supervisor": "👀",
        "storage": "📦",
        "podcast": "🎙️",
        "error": "❌",
        "step": "➡️"
    }
    icon = agent_icons.get(agent.lower(), "💬")
    print(f"\n[{timestamp}] {icon} {agent}:")
    print(f"    {message}")

def check_api_keys():
    """Check if all required API keys are present"""
    required_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ELEVEN_API_KEY": os.getenv("ELEVEN_API_KEY"),
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
    
    return required_keys["TAVILY_API_KEY"]

async def test_transcript_saving(workflow, query: str):
    """Test that transcripts are properly saved to podcasts.json"""
    try:
        # Get initial transcript count
        initial_transcripts = []
        if os.path.exists(TRANSCRIPTS_FILE):
            with open(TRANSCRIPTS_FILE, 'r') as f:
                initial_transcripts = json.load(f)
        initial_count = len(initial_transcripts)
        
        # Run workflow
        result = await run_workflow(workflow, query)
        
        # Verify transcript was saved
        if not os.path.exists(TRANSCRIPTS_FILE):
            return False, "Transcripts file was not created"
            
        with open(TRANSCRIPTS_FILE, 'r') as f:
            transcripts = json.load(f)
            
        if len(transcripts) <= initial_count:
            return False, "No new transcript was added"
            
        latest_transcript = transcripts[-1]
        if not all(key in latest_transcript for key in ["id", "podcastScript", "topic"]):
            return False, "Transcript is missing required fields"
            
        if latest_transcript["topic"] != query:
            return False, f"Topic mismatch. Expected: {query}, Got: {latest_transcript['topic']}"
            
        return True, "Transcript was saved successfully"
        
    except Exception as e:
        return False, f"Error in transcript test: {str(e)}\n{traceback.format_exc()}"

async def test_single_turn(workflow_graph, query: str):
    """Test a single turn of the workflow"""
    result = await run_workflow(workflow_graph, query)
    return len(result["debate_history"]) > 0

async def test_debate_length(workflow, query):
    """Test that debate history does not exceed 20 messages"""
    result = await run_workflow(workflow, query)
    return len(result["debate_history"]) <= 20

async def test_podcast_generation(workflow, query):
    """Test podcast generation functionality"""
    try:
        result = await run_workflow(workflow, query)
        
        # Check for podcast data
        if "final_podcast" not in result:
            return False, "No podcast data in result"
        
        podcast_data = result["final_podcast"]
        
        # Check for errors in podcast generation
        if "error" in podcast_data:
            return False, f"Podcast generation error: {podcast_data['error']}"
        
        # Verify script generation
        if not podcast_data.get("content"):
            return False, "No podcast script generated"
        
        # Verify audio file generation
        if not podcast_data.get("audio_file"):
            return False, "No audio file generated"
        
        # Check if audio file exists
        audio_path = os.path.join(os.path.dirname(__file__), "audio_storage", podcast_data["audio_file"])
        if not os.path.exists(audio_path):
            return False, f"Audio file not found at {audio_path}"
        
        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            return False, "Audio file is empty"
        
        # Check if transcript was saved
        transcript_success, transcript_message = await test_transcript_saving(workflow, query)
        if not transcript_success:
            return False, f"Transcript saving failed: {transcript_message}"
        
        return True, f"Podcast generated successfully (file size: {file_size} bytes)"
    
    except Exception as e:
        return False, f"Error in podcast test: {str(e)}\n{traceback.format_exc()}"

async def run_all_tests():
    log_step("Running all tests")
    
    try:
        # Check API keys
        tavily_api_key = check_api_keys()
        
        # Create workflow
        workflow = create_workflow(tavily_api_key)
        
        # Test queries
        queries = [
            "What are the environmental impacts of electric vehicles?",
            "How does artificial intelligence impact healthcare?",
            "What are the pros and cons of remote work?",
            "Discuss the future of space exploration"
        ]
        
        results = {}
        for query in queries:
            try:
                log_step(f"Testing query", query)
                
                # Test transcript saving
                transcript_success, transcript_message = await test_transcript_saving(workflow, query)
                log_agent("step", f"Transcript test: {transcript_message}")
                
                result = await run_workflow(workflow, query)
                
                podcast_success, podcast_message = await test_podcast_generation(workflow, query)
                
                results[query] = {
                    "success": True,
                    "debate_length": len(result["debate_history"]),
                    "supervisor_notes": len(result["supervisor_notes"]),
                    "transcript_saved": transcript_success,
                    "transcript_status": transcript_message,
                    "podcast_generated": podcast_success,
                    "podcast_status": podcast_message,
                    "timestamp": datetime.now().isoformat()
                }
                
                log_agent("step", f"Test completed for: {query}")
                
            except Exception as e:
                results[query] = {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                }
                log_agent("error", f"Test failed for: {query}\n{str(e)}")
            
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        log_step("Results saved", f"File: {filename}")
        print("\nTest Results:")
        print(json.dumps(results, indent=2))
        
        return results
    
    except Exception as e:
        log_agent("error", f"Critical error in tests: {str(e)}\n{traceback.format_exc()}")
        raise

async def test_workflow():
    log_step("Starting workflow test")
    
    try:
        # Check API keys
        tavily_api_key = check_api_keys()

        # Create the workflow
        log_step("Creating workflow graph")
        workflow_graph = create_workflow(tavily_api_key)
        
        # Test query
        test_query = "Should artificial intelligence be regulated?"
        log_step("Test Query", test_query)
        
        # Run the workflow
        log_step("Running workflow")
        result = await run_workflow(workflow_graph, test_query)
        
        # Test transcript saving
        log_step("Testing transcript saving")
        transcript_success, transcript_message = await test_transcript_saving(workflow_graph, test_query)
        log_agent("step", f"Transcript test: {transcript_message}")
        
        # Print extractor results
        log_step("Information Extraction Phase")
        if "extractor_data" in result:
            log_agent("Extractor", result["extractor_data"].get("content", "No content"))
        
        # Print debate history
        log_step("Debate Phase")
        print("\nDebate Timeline:")
        for i, entry in enumerate(result["debate_history"], 1):
            log_agent(entry["speaker"], entry["content"])
            if i < len(result["supervisor_notes"]):
                log_agent("Supervisor", f"Analysis of Turn {i}:\n    {result['supervisor_notes'][i]}")
        
        # Print final supervisor analysis
        log_step("Final Supervisor Analysis")
        if result["supervisor_notes"]:
            log_agent("Supervisor", result["supervisor_notes"][-1])
        
        # Print podcast results
        log_step("Podcast Production Phase")
        if "final_podcast" in result:
            podcast_data = result["final_podcast"]
            if "error" in podcast_data:
                log_agent("Podcast", f"Error: {podcast_data['error']}")
            else:
                log_agent("Podcast", "Script:\n" + podcast_data["content"])
                if podcast_data.get("audio_file"):
                    audio_path = os.path.join(os.path.dirname(__file__), "audio_storage", podcast_data["audio_file"])
                    file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
                    log_agent("Podcast", f"Audio file saved as: {podcast_data['audio_file']} (size: {file_size} bytes)")
        
        # Save results
        log_step("Saving Results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "query": test_query,
                "workflow_results": result,
                "transcript_saved": transcript_success,
                "transcript_status": transcript_message
            }, f, indent=2)
        log_step("Results Saved", f"File: {filename}")
            
    except Exception as e:
        log_step("ERROR", f"Workflow execution failed: {str(e)}")
        print("\nFull traceback:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 Starting AI Debate Workflow Test")
    print("="*50)
    
    try:
        asyncio.run(test_workflow())
        print("\n" + "="*50)
        print("✅ Test Complete")
        print("="*50)
        
        # Run comprehensive tests
        print("\nRunning comprehensive tests...")
        asyncio.run(run_all_tests())
        print("\n" + "="*50)
        print("✅ All Tests Complete")
        print("="*50)
    except Exception as e:
        print("\n" + "="*50)
        print(f"❌ Tests Failed: {str(e)}")
        print("="*50)
        raise 