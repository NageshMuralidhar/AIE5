import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import '../App.css';

const API_URL = 'http://localhost:8000';

interface Podcast {
  id: number;
  title: string;
  description: string;
  audio_file: string;
  filename: string;
  category: string;
}

const Podcasts: React.FC = () => {
  const navigate = useNavigate();
  const [podcasts, setPodcasts] = useState<Podcast[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchPodcasts();
  }, []);

  const handleDelete = async (podcast: Podcast) => {
    try {
      const response = await fetch(`${API_URL}/audio/${podcast.filename}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(
          `Failed to delete podcast (${response.status}): ${errorData}`
        );
      }
      
      setPodcasts(prev => prev.filter(p => p.filename !== podcast.filename));
      
    } catch (err) {
      console.error('Delete error:', err);
      setError(err instanceof Error ? err.message : 'Failed to delete podcast');
      
      setTimeout(() => setError(""), 5000);
    }
  };

  const fetchPodcasts = async () => {
    try {
      const response = await fetch(`${API_URL}/audio-list`);
      if (!response.ok) {
        throw new Error('Failed to fetch podcasts');
      }
      
      const files = await response.json();
      
      const podcastList: Podcast[] = files.map((file: any, index: number) => {
        const filename = file.filename;
        const [queryPart, descriptionPart, categoryWithExt] = filename.split('-');
        const category = categoryWithExt.replace('.mp3', '');
        
        return {
          id: index + 1,
          title: `${descriptionPart.replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase())}`,
          description: `A debate exploring ${queryPart.replace(/_/g, ' ')}`,
          audio_file: `${API_URL}${file.path}`,
          filename: filename,
          category: category.replace(/_/g, ' ')
        };
      });

      setPodcasts(podcastList);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="podcasts-container">
        <div className="loading-message">Loading podcasts...</div>
      </div>
    );
  }

  if (error !== "") {
    return (
      <div className="podcasts-container">
        <div className="error-message">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="podcasts-container">
      <header className="podcasts-header">
        <h1>Your Generated Podcasts</h1>
        <p>Listen to AI-generated debate podcasts on various topics</p>
      </header>

      <div className="podcasts-grid">
        {podcasts.map(podcast => (
          <div 
            key={podcast.id} 
            className="podcast-card"
            onClick={() => navigate(`/podcast/${podcast.id}`)}
            style={{ cursor: 'pointer' }}
          >
            <div className="podcast-content">
              <div className="podcast-header">
                <h2 className="podcast-title">{podcast.title}</h2>
                <button 
                  className="delete-button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(podcast);
                  }}
                  aria-label="Delete podcast"
                >
                  ×
                </button>
              </div>
              <div className="category-pill">{podcast.category}</div>
              <p className="description">{podcast.description}</p>
              <div className="audio-player" onClick={e => e.stopPropagation()}>
                <audio 
                  controls
                  src={podcast.audio_file}
                >
                  Your browser does not support the audio element.
                </audio>
              </div>
            </div>
          </div>
        ))}
        {podcasts.length === 0 && (
          <div className="no-podcasts-message">
            No podcasts found. Generate your first podcast from the home page!
          </div>
        )}
      </div>
    </div>
  );
};

export default Podcasts; 