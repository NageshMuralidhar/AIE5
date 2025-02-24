export interface Message {
  id: number;
  text: string;
  agent: string;
  audio_file?: string;
  title?: string;
  description?: string;
}

export interface DebateEntry {
  speaker: string;
  content: string;
}

export interface PodcastData {
  content: string;
  audio_file: string;
  title: string;
  description: string;
  category: string;
}

export interface ChatResponse {
  debate_history: DebateEntry[];
  supervisor_notes: string[];
  supervisor_chunks: {
    [key: string]: string[];
  }[];
  extractor_data: {
    content: string;
    raw_results: any;
  };
  final_podcast: PodcastData;
} 