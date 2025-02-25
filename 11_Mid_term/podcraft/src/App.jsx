import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import './App.css';
import Home from './pages/Home';
import Podcasts from './pages/Podcasts';
import PodcastForm from './pages/PodcastForm';
import WaveCanvas from './components/WaveCanvas';

function App() {
  const [isDarkTheme, setIsDarkTheme] = useState(true);

  useEffect(() => {
    document.body.setAttribute('data-theme', isDarkTheme ? 'dark' : 'light');
  }, [isDarkTheme]);

  return (
    <Router>
      <div className="app">
        <WaveCanvas />
        <nav className="leftnav">
          <div className="nav-brand">PodCraft</div>
          <div className="nav-links">
            <NavLink to="/" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              <span className="nav-icon">🏠</span>
              Home
            </NavLink>
            <NavLink to="/podcasts" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              <span className="nav-icon">🎧</span>
              Podcasts
            </NavLink>
          </div>
          <div className="theme-toggle">
            <button 
              className="theme-button"
              onClick={() => setIsDarkTheme(!isDarkTheme)}
            >
              <span className="nav-icon">{isDarkTheme ? '☀️' : '🌙'}</span>
              {isDarkTheme ? 'Light Mode' : 'Dark Mode'}
            </button>
          </div>
        </nav>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/podcasts" element={<Podcasts />} />
            <Route path="/podcast/:id" element={<PodcastForm />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
