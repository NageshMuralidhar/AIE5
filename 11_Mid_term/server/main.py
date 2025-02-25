from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
from workflow import create_workflow, run_workflow
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not configured")

# Initialize OpenAI components
chat_model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=openai_api_key
)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS for frontend development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
    expose_headers=["Content-Type", "Content-Length"],
    max_age=600,
)

# Configure audio storage
audio_dir = os.path.join(os.path.dirname(__file__), "audio_storage")
os.makedirs(audio_dir, exist_ok=True)

# Mount the audio directory as a static file directory
app.mount("/audio-files", StaticFiles(directory=audio_dir), name="audio")

# Configure context storage
context_dir = os.path.join(os.path.dirname(__file__), "context_storage")
os.makedirs(context_dir, exist_ok=True)

class ChatMessage(BaseModel):
    content: str
    context: Optional[Dict[str, Any]] = None
    agent_type: Optional[str] = "believer"

class WorkflowResponse(BaseModel):
    debate_history: List[Dict[str, str]]
    supervisor_notes: List[str]
    supervisor_chunks: List[Dict[str, List[str]]]
    extractor_data: Dict[str, Any]
    final_podcast: Dict[str, Any]

class PodcastChatRequest(BaseModel):
    message: str

class PodcastChatResponse(BaseModel):
    response: str

@app.get("/audio-list")
async def list_audio_files():
    """List all available audio files."""
    try:
        files = os.listdir(audio_dir)
        audio_files = []
        for file in files:
            if file.endswith(('.mp3', '.wav')):
                file_path = os.path.join(audio_dir, file)
                audio_files.append({
                    "filename": file,
                    "path": f"/audio-files/{file}",
                    "size": os.path.getsize(file_path)
                })
        return audio_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/audio/{filename}")
async def delete_audio_file(filename: str):
    """Delete an audio file and its corresponding transcript."""
    try:
        # Delete audio file
        file_path = os.path.join(audio_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get all audio files to determine the podcast ID
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.mp3', '.wav'))]
        try:
            # Find the index (0-based) of the file being deleted
            podcast_id = audio_files.index(filename) + 1  # Convert to 1-based ID
            logger.info(f"Deleting podcast with ID: {podcast_id}")
            
            # Path to transcripts file
            transcripts_file = os.path.join(os.path.dirname(__file__), "transcripts", "podcasts.json")
            
            # Update transcripts if file exists
            if os.path.exists(transcripts_file):
                with open(transcripts_file, 'r') as f:
                    transcripts = json.load(f)
                
                # Remove the transcript at the corresponding index
                if len(transcripts) >= podcast_id:
                    transcripts.pop(podcast_id - 1)  # Convert back to 0-based index
                    
                    # Save updated transcripts
                    with open(transcripts_file, 'w') as f:
                        json.dump(transcripts, f, indent=2)
                    logger.info(f"Removed transcript for podcast ID {podcast_id}")
            
            # Delete the audio file
            os.remove(file_path)
            logger.info(f"Deleted audio file: {filename}")
            
            return {"message": "File and transcript deleted successfully"}
            
        except ValueError:
            logger.error(f"Could not determine podcast ID for file: {filename}")
            # Still delete the audio file even if transcript removal fails
            os.remove(file_path)
            return {"message": "Audio file deleted, but transcript could not be removed"}
            
    except Exception as e:
        logger.error(f"Error in delete_audio_file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Get an audio file by filename."""
    try:
        file_path = os.path.join(audio_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/podcast/{podcast_id}/context")
async def get_podcast_context(podcast_id: str):
    """Get or generate context for a podcast."""
    try:
        logger.info(f"Getting context for podcast {podcast_id}")
        context_path = os.path.join(context_dir, f"{podcast_id}_context.json")
        
        # If context exists, return it
        if os.path.exists(context_path):
            logger.info(f"Found existing context file at {context_path}")
            with open(context_path, 'r') as f:
                return json.load(f)
        
        # If no context exists, we need to create it from the podcast content
        logger.info("No existing context found, creating new context")
        
        # Get the audio files to find the podcast filename
        files = os.listdir(audio_dir)
        logger.info(f"Found {len(files)} files in audio directory")
        podcast_files = [f for f in files if f.endswith('.mp3')]
        logger.info(f"Found {len(podcast_files)} podcast files: {podcast_files}")
        
        if not podcast_files:
            logger.error("No podcast files found")
            raise HTTPException(status_code=404, detail="No podcast files found")
            
        # Find the podcast file that matches this ID
        try:
            podcast_index = int(podcast_id) - 1  # Convert 1-based ID to 0-based index
            if podcast_index < 0 or podcast_index >= len(podcast_files):
                raise ValueError(f"Invalid podcast ID: {podcast_id}, total podcasts: {len(podcast_files)}")
            podcast_filename = podcast_files[podcast_index]
            logger.info(f"Selected podcast file: {podcast_filename}")
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid podcast ID: {podcast_id}, Error: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Invalid podcast ID: {podcast_id}")
        
        # Extract topic from filename
        try:
            topic = podcast_filename.split('-')[0].replace('_', ' ')
            logger.info(f"Extracted topic: {topic}")
        except Exception as e:
            logger.error(f"Error extracting topic from filename: {podcast_filename}, Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting topic from filename: {str(e)}")

        # Initialize OpenAI chat model for content analysis
        try:
            chat_model = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.3,
                openai_api_key=openai_api_key
            )
            logger.info("Successfully initialized ChatOpenAI")
        except Exception as e:
            logger.error(f"Error initializing ChatOpenAI: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error initializing chat model: {str(e)}")

        # Create prompt template for content analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content analyzer. Your task is to:
            1. Analyze the given topic and create balanced, factual content chunks about it
            2. Generate two types of chunks:
               - Believer chunks: Positive aspects, opportunities, and solutions related to the topic
               - Skeptic chunks: Challenges, risks, and critical questions about the topic
            3. Each chunk should be self-contained and focused on a single point
            4. Keep chunks concise (2-3 sentences each)
            5. Ensure all content is factual and balanced
            
            Format your response as a JSON object with two arrays:
            {{
                "believer_chunks": ["chunk1", "chunk2", ...],
                "skeptic_chunks": ["chunk1", "chunk2", ...]
            }}"""),
            ("human", "Create balanced content chunks about this topic: {topic}")
        ])

        # Generate content chunks
        chain = prompt | chat_model
        
        try:
            logger.info(f"Generating content chunks for topic: {topic}")
            response = await chain.ainvoke({
                "topic": topic
            })
            logger.info("Successfully received response from OpenAI")
            
            # Parse the response content as JSON
            try:
                content_chunks = json.loads(response.content)
                logger.info(f"Successfully parsed response JSON with {len(content_chunks.get('believer_chunks', []))} believer chunks and {len(content_chunks.get('skeptic_chunks', []))} skeptic chunks")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing response JSON: {str(e)}, Response content: {response.content}")
                raise HTTPException(status_code=500, detail=f"Error parsing content chunks: {str(e)}")
            
            # Create the context object
            context = {
                "topic": topic,
                "believer_chunks": content_chunks.get("believer_chunks", []),
                "skeptic_chunks": content_chunks.get("skeptic_chunks", [])
            }
            
            # Save the context
            try:
                with open(context_path, 'w') as f:
                    json.dump(context, f)
                    logger.info(f"Saved new context to {context_path}")
            except Exception as e:
                logger.error(f"Error saving context file: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error saving context file: {str(e)}")
            
            return context
            
        except Exception as e:
            logger.error(f"Error generating content chunks: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating content chunks: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_podcast_context: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(message: ChatMessage):
    """Process a chat message with context-awareness."""
    try:
        # Log incoming message
        logger.info(f"Received chat message: {message}")
        
        # Get API key
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            logger.error("Tavily API key not found")
            raise HTTPException(status_code=500, detail="Tavily API key not configured")

        # Initialize the workflow
        try:
            workflow = create_workflow(tavily_api_key)
            logger.info("Workflow created successfully")
        except Exception as e:
            logger.error(f"Error creating workflow: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating workflow: {str(e)}")
        
        # Run the workflow with context
        try:
            result = await run_workflow(
                workflow,
                message.content,
                agent_type=message.agent_type,
                context=message.context
            )
            logger.info("Workflow completed successfully")
        except Exception as e:
            logger.error(f"Error running workflow: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error running workflow: {str(e)}")
        
        return WorkflowResponse(**result)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/podcast-chat/{podcast_id}", response_model=PodcastChatResponse)
async def podcast_chat(podcast_id: str, request: PodcastChatRequest):
    """Handle chat messages for a specific podcast."""
    try:
        logger.info(f"Processing chat message for podcast {podcast_id}")
        
        # Path to transcripts file
        transcripts_file = os.path.join(os.path.dirname(__file__), "transcripts", "podcasts.json")
        
        # Check if transcripts file exists
        if not os.path.exists(transcripts_file):
            raise HTTPException(status_code=404, detail="Transcripts file not found")
            
        # Read transcripts
        with open(transcripts_file, 'r') as f:
            transcripts = json.load(f)
            
        # Convert podcast_id to zero-based index
        try:
            podcast_index = int(podcast_id) - 1
            if podcast_index < 0 or podcast_index >= len(transcripts):
                raise ValueError(f"Invalid podcast ID: {podcast_id}")
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
            
        # Get podcast transcript
        podcast_transcript = transcripts[podcast_index]["podcastScript"]

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        # Use split_text for strings instead of split_documents
        chunks = text_splitter.split_text(podcast_transcript)
        
        # Initialize embedding model
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

        # Create a unique collection name for this podcast
        collection_name = f"podcast_{podcast_id}"
        
        # Initialize Qdrant with local storage
        vectorstore = Qdrant.from_texts(
            texts=chunks,
            embedding=embedding_model,
            location=":memory:",  # Use in-memory storage
            collection_name=collection_name
        )
        
        # Configure the retriever with search parameters
        qdrant_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Get top 3 most relevant chunks
        )

        base_rag_prompt_template = """\
        You are a helpful podcast assistant. Answer the user's question based on the provided context from the podcast transcript.
        If you can't find the answer in the context, just say "I don't have enough information to answer that question."
        Keep your responses concise and focused on the question.

        Context:
        {context}

        Question:
        {question}
        """

        base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)
        base_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

        # Create the RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Add logging for the retrieved documents and final prompt
        def get_context_and_log(input_dict):
            context = format_docs(qdrant_retriever.get_relevant_documents(input_dict["question"]))
            logger.info("Retrieved context from podcast:")
            logger.info("-" * 50)
            logger.info(f"Context:\n{context}")
            logger.info("-" * 50)
            logger.info(f"Question: {input_dict['question']}")
            logger.info("-" * 50)
            return {"context": context, "question": input_dict["question"]}

        # Create the chain
        chain = (
            RunnablePassthrough()
            | get_context_and_log
            | base_rag_prompt
            | base_llm
        )

        # Get response
        response = chain.invoke({"question": request.message})
        
        return PodcastChatResponse(response=response.content)
        
    except Exception as e:
        logger.error(f"Error in podcast chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 