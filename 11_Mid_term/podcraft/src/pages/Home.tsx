import React, { useState, useEffect, useRef } from 'react';
import '../App.css';

interface Message {
  id: number;
  text: string;
  agent: string;
  audio_file?: string;
  title?: string;
  description?: string;
  category?: string;
}

interface DebateEntry {
  speaker: string;
  content: string;
}

interface ChatResponse {
  debate_history: DebateEntry[];
  supervisor_notes: string[];
  final_podcast?: {
    content: string;
    audio_file: string;
    title: string;
    description: string;
  };
}

const API_URL = 'http://localhost:8000';

const Home: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, text: "Welcome! I'll help you create your own podcast content by exploring topics through an AI debate. Enter any topic of choice.", agent: "system" },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessageToServer = async (message: string): Promise<ChatResponse> => {
    try {
      setMessages(prev => [...prev, {
        id: prev.length + 1,
        text: `Processing your request...`,
        agent: "system"
      }]);

      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: message,
          agent_type: "believer",
          context: {
            podcast_id: null,
            agent_chunks: [],
            current_agent: "believer"
          }
        })
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`Server error: ${response.status} - ${errorData}`);
      }

      const data: ChatResponse = await response.json();
      return data;
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const userMessage: Message = {
      id: messages.length + 1,
      text: inputMessage,
      agent: "user"
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await sendMessageToServer(inputMessage);
      
      if (response.debate_history) {
        response.debate_history.forEach((entry) => {
          if (entry.speaker && entry.content) {
            const message: Message = {
              id: messages.length + 1,
              text: entry.content,
              agent: entry.speaker
            };
            setMessages(prev => [...prev, message]);
          }
        });
      }

      if (response.supervisor_notes) {
        response.supervisor_notes.forEach((note) => {
          const supervisorMessage: Message = {
            id: messages.length + 1,
            text: note,
            agent: "supervisor"
          };
          setMessages(prev => [...prev, supervisorMessage]);
        });
      }

      if (response.final_podcast && response.final_podcast.audio_file) {
        const filename = response.final_podcast.audio_file;
        const [queryPart, descriptionPart, categoryWithExt] = filename.split('-');
        const category = categoryWithExt.replace('.mp3', '');
        
        const podcastMessage: Message = {
          id: messages.length + 1,
          text: response.final_podcast.content || "Podcast generated successfully!",
          agent: "system",
          audio_file: `/audio-files/${filename}`,
          title: descriptionPart.replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase()),
          description: `A debate exploring ${queryPart.replace(/_/g, ' ')}`,
          category: category.replace(/_/g, ' ')
        };
        setMessages(prev => [...prev, podcastMessage]);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        id: prev.length + 1,
        text: `Error: ${error.message}`,
        agent: "system"
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.agent}-message`}>
            <div className="message-content">
              <div className="agent-icon">
                {message.agent === "user" ? "👤" : 
                 message.agent === "system" ? "🤖" :
                 message.agent === "believer" ? "💡" :
                 message.agent === "skeptic" ? "🤔" :
                 message.agent === "supervisor" ? "👀" :
                 message.agent === "extractor" ? "🔍" : "💬"}
              </div>
              <div className="message-text-content">
                <div className="agent-name">{message.agent}</div>
                <div className="message-text">{message.text}</div>
                {message.audio_file && (
                  <div className="podcast-card">
                    <div className="podcast-content">
                      <h2 className="podcast-title">{message.title || "Generated Podcast"}</h2>
                      {message.category && (
                        <div className="category-pill">{message.category}</div>
                      )}
                      <p className="description">{message.description || "An AI-generated debate podcast exploring different perspectives"}</p>
                      <div className="audio-player">
                        <audio controls src={`${API_URL}${message.audio_file}`} ref={audioRef}>
                          Your browser does not support the audio element.
                        </audio>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message system-message">
            <div className="message-content">
              <div className="loading-dots">Debating</div>
              <div className="loading-dots">This might take a few moments since 2 agents are fighting over getting the best insights for you</div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="chat-input-form">
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="Type your message..."
          className="chat-input"
          disabled={isLoading}
        />
        <button type="submit" className="chat-send-button" disabled={isLoading}>
          Send
        </button>
      </form>
    </div>
  );
};

export default Home; 