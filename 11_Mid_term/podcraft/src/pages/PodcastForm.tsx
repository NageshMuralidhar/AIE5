import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import '../App.css';

const API_URL = 'http://localhost:8000';

interface Message {
  id: number;
  text: string;
  agent: string;
}

interface Podcast {
  id: number;
  title: string;
  description: string;
  audio_file: string;
}

interface PodcastContext {
  topic: string;
  believer_chunks: string[];
  skeptic_chunks: string[];
}

const PodcastForm: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [podcast, setPodcast] = useState<Podcast | null>(null);
  const [podcastContext, setPodcastContext] = useState<PodcastContext | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    const fetchPodcastAndContext = async () => {
      try {
        if (!id) return;

        // Fetch podcast details
        const response = await fetch(`${API_URL}/audio-list`);
        if (!response.ok) {
          throw new Error('Failed to fetch podcasts');
        }
        const files = await response.json();
        
        const podcastList = files.map((file: any, index: number) => ({
          id: index + 1,
          title: file.filename.split('-')[0].replace(/_/g, ' '),
          description: "An AI-generated debate podcast exploring different perspectives",
          audio_file: `${API_URL}/audio/${file.filename}`
        }));

        const selectedPodcast = podcastList.find(p => p.id === parseInt(id));
        if (selectedPodcast) {
          setPodcast(selectedPodcast);
          
          // Fetch podcast context
          const contextResponse = await fetch(`${API_URL}/podcast/${id}/context`);
          if (contextResponse.ok) {
            const contextData: PodcastContext = await contextResponse.json();
            setPodcastContext(contextData);
          }
        }
      } catch (err) {
        console.error('Error fetching podcast:', err);
      }
    };

    fetchPodcastAndContext();
  }, [id]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || !id) return;

    const userMessage: Message = {
      id: messages.length + 1,
      text: inputMessage,
      agent: "user"
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/podcast-chat/${id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage
        })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      const botMessage: Message = {
        id: messages.length + 2,
        text: data.response,
        agent: "assistant"
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, {
        id: prev.length + 1,
        text: `Error: ${error instanceof Error ? error.message : 'Failed to send message'}`,
        agent: "system"
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="podcast-form-container">
      <div className="chat-column">
        <div className="podcast-chat-container">
          {podcast && (
            <div className="podcast-player-header">
              <h2 className="podcast-title">{podcast.title}</h2>
              {podcastContext && (
                <p className="podcast-topic">Topic: {podcastContext.topic}</p>
              )}
              <div className="audio-player">
                <audio 
                  controls
                  src={podcast.audio_file}
                >
                  Your browser does not support the audio element.
                </audio>
              </div>
            </div>
          )}
          
          <div className="podcast-chat-messages">
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.agent}-message`}>
                <div className="message-content">
                  <div className="agent-icon">
                    {message.agent === "user" ? "👤" : 
                     message.agent === "system" ? "🤖" : 
                     message.agent === "assistant" ? "🤖" : "💬"}
                  </div>
                  <div className="message-text-content">
                    <div className="agent-name">{message.agent}</div>
                    <div className="message-text">{message.text}</div>
                  </div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message system-message">
                <div className="message-content">
                  <div className="loading-dots">Processing</div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="podcast-chat-input-form">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask a question about this podcast..."
              className="podcast-chat-input"
              disabled={isLoading}
            />
            <button type="submit" className="podcast-chat-send-button" disabled={isLoading}>
              Send
            </button>
          </form>
          
          {podcastContext && (
            <div className="relevant-chunks">
              <div className="chunk-section">
                <h4>Key Points</h4>
                <ul>
                  {podcastContext.believer_chunks.map((chunk, i) => (
                    <li key={`believer-${i}`}>{chunk}</li>
                  ))}
                  {podcastContext.skeptic_chunks.map((chunk, i) => (
                    <li key={`skeptic-${i}`}>{chunk}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PodcastForm; 