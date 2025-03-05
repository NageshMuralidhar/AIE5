# PodCraft: AI-Powered Debate-Driven Podcast Generator

## Task 0: Introduction

PodCraft represents a groundbreaking midterm submission that revolutionizes podcast content creation through AI-driven debate synthesis. This project showcases the power of combining multiple AI agents in a structured workflow to generate engaging, balanced, and thought-provoking podcast content.

## Task 1: Problem Definition and Audience

### Problem Statement

Content creators struggle to produce balanced, well-researched podcast content that presents multiple perspectives on complex topics while maintaining engagement and factual accuracy.

### Target Audience Impact

For podcast creators and content developers, the traditional process of creating balanced content is extremely time-consuming and resource-intensive. They must:

- Research multiple viewpoints extensively
- Structure coherent arguments and counter-arguments
- Maintain objectivity while ensuring engagement
- Fact-check and verify information from various sources
- Script and re-script content to maintain flow and coherence

This challenge often results in either biased content or superficial coverage of topics, limiting the educational and entertainment value for listeners.

## Task 2: Proposed Solution

### Solution Overview

PodCraft introduces an innovative AI-powered podcast generation system that orchestrates a structured debate between AI agents representing different viewpoints. The system:

- Automatically extracts key points and context from user queries
- Facilitates a balanced debate between "believer" and "skeptic" agents
- Synthesizes the debate into engaging podcast scripts
- Maintains factual accuracy while ensuring entertainment value
- Provides real-time supervision and quality control

### Technical Stack

#### LLM

- **Primary Model**: GPT-3.5 Turbo
- **Choice Rationale**: Optimal balance between performance and cost, with strong capabilities in natural conversation and debate synthesis

#### Embedding Model

- **Technology**: OpenAI Embeddings
- **Rationale**: Industry-leading semantic understanding for context matching and retrieval

#### Orchestration

- **Framework**: LangGraph
- **Rationale**: Powerful workflow management for complex agent interactions and state management

#### Vector Database

- **Technology**: Qdrant
- **Rationale**: High-performance vector storage with excellent scaling capabilities

#### Monitoring

- **Implementation**: Custom logging system with detailed agent interaction tracking
- **Rationale**: Comprehensive visibility into agent behavior and system performance

#### Evaluation

- **Approach**: Supervisor agent with real-time quality assessment
- **Rationale**: Ensures content quality and debate balance throughout the generation process

#### User Interface

- **Frontend**: React with Vite
- **Backend**: FastAPI
- **Rationale**: Modern, responsive design with efficient API handling

#### Serving & Inference

- **Architecture**: Containerized microservices with Docker
- **Rationale**: Scalable and portable deployment with isolated components

### Agentic Reasoning

The system employs multiple specialized agents:

1. **Extractor Agent**: Analyzes user queries and extracts key topics
2. **Believer Agent**: Presents positive perspectives and opportunities
3. **Skeptic Agent**: Challenges assumptions and identifies potential risks
4. **Supervisor Agent**: Monitors debate quality and ensures balance
5. **Podcast Producer Agent**: Synthesizes debates into cohesive scripts

## Task 3: Data Strategy

### Data Sources and APIs

1. **OpenAI API**: Core language model capabilities
2. **Tavily API**: Real-time fact-checking and research
3. **Custom Knowledge Base**: Stored in Qdrant for context retrieval
4. **Transcript Storage**: JSON-based storage for generated content

### Chunking Strategy

- **Implementation**: RecursiveCharacterTextSplitter
- **Chunk Size**: Optimized for podcast script segments (1000 characters)
- **Overlap**: 200 characters to maintain context continuity
- **Rationale**: Balances context preservation with processing efficiency

### Additional Data Requirements

1. **Debate History**: Stored for continuity and reference
2. **Supervisor Notes**: Maintained for quality control
3. **Context Chunks**: Organized by perspective (believer/skeptic)
4. **Audio Storage**: For potential future audio generation integration

## Task 4: Building a Quick End-to-End Prototype

Link: https://huggingface.co/spaces/dataera2013/midterm

## Task 5: Creating a Golden Test Data Set

| Metric                | Score  |
| --------------------- | ------ |
| Context Recall        | 0.7302 |
| Faithfulness          | 0.8521 |
| Factual Correctness   | 0.5267 |
| Answer Relevancy      | 0.7904 |
| Context Entity Recall | 0.2991 |
| Noise Sensitivity     | 0.2556 |

### Pipeline assessment

- Here are the key conclusions about the performance and effectiveness of the pipeline:

- Context Understanding and Relevancy (Moderate Performance)

- The Context Recall (0.7302) and Answer Relevancy (0.7904) scores indicate that the pipeline performs fairly well in understanding the provided context and generating answers relevant to the context. However, there is still room for improvement in making responses more comprehensive.
Faithfulness (High Performance)

- The Faithfulness (0.8521) score shows that the pipeline generates responses that are largely consistent with the information provided in the context, meaning the outputs align well with the context's content without hallucination.
Factual Correctness (Low Performance)

- The Factual Correctness (0.5267) score highlights a major area for improvement. While the model is faithful to the input, the actual factual accuracy of the generated content is relatively low, suggesting that either the RAG model is not retrieving the most up-to-date or correct information, or the generation process is misinterpreting the data.
Entity Coverage (Weak Performance)

- The Context Entity Recall (0.2991) score is quite low, indicating that the pipeline struggles to extract and reference key entities from the context. This suggests the model might miss out on critical details, which could impact the richness and informativeness of the podcast content.
Noise Sensitivity (Good Robustness)

- The Noise Sensitivity (0.2556) score indicates that the pipeline is fairly robust against noisy or irrelevant information. This means the model can filter out distracting information effectively, contributing to more coherent responses.

#### Final Assessment:
- The pipeline is faithful and contextually relevant but struggles with factual correctness and comprehensive entity extraction.
- Improving entity recall and fact-checking mechanisms through better RAG retrieval strategies or multi-agent validation layers could significantly boost the overall content quality.
- Adding a user review/edit interface for generated insights could further improve factual correctness and allow for human intervention where the model might fail.

## Task 6: Fine-Tuning Open-Source Embeddings

Link: https://huggingface.co/dataera2013/mt-1

# Task 7: Assessing Performance

| Metric                | Score  |
| --------------------- | ------ |
| Context Recall        | 0.7778 |
| Faithfulness          | 0.8532 |
| Factual Correctness   | 0.5658 |
| Answer Relevancy      | 0.7909 |
| Context Entity Recall | 0.3107 |
| Noise Sensitivity     | 0.2628 |

**Key Improvements:**

- Context Recall: +4.76%
- Factual Correctness: +3.91%
- Context Entity Recall: +1.16%

The fine-tuned model shows modest improvements across all metrics, with the most significant gains in context recall and factual correctness.

# Task 8: Loom video

https://www.loom.com/share/475f5524cd7f4bbd9d24b8a59ade3c0a

# Task 9: Improvements for next iteration

- Introduce advanced, purpose-driven AI agents to enhance podcast articulation and depth.
- Integrate two or more voice modules to deliver a more natural, human-like podcast experience.
- Develop a Spotify or YouTube agent to automate podcast uploads directly to streaming platforms.
- Implement an Insight Generator using Retrieval-Augmented Generation (RAG) to extract meaningful insights from real-time data, allowing users to edit or add their own insights to enrich the podcast content.
- Add hidden Easter Egg voice modules, featuring personalities like Elon Musk or other notable figures, to create surprising and engaging moments.
- Build a Workflow Automation Builder that enables users to design custom workflows and automate the entire podcast creation and publishing process with a single click.
