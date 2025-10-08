# Car Troubleshoot Europe - LangChain ChatBot

[![Backend CI](https://github.com/Razeling/Langchain_Chatbot/actions/workflows/backend-ci.yml/badge.svg)](https://github.com/Razeling/Langchain_Chatbot/actions/workflows/backend-ci.yml)
[![Frontend CI](https://github.com/Razeling/Langchain_Chatbot/actions/workflows/frontend-ci.yml/badge.svg)](https://github.com/Razeling/Langchain_Chatbot/actions/workflows/frontend-ci.yml)

AI-powered automotive diagnostic assistant specialized for European markets with RAG (Retrieval Augmented Generation) and function calling capabilities.

## 🚗 Features

- **European Market Focus**: Specialized knowledge for 27+ European countries
- **Multi-language Support**: Optimized for European languages and regional differences
- **RAG-Powered Diagnostics**: Retrieval Augmented Generation using automotive knowledge base
- **Function Calling**: Intelligent cost estimation and maintenance scheduling
- **Real-time Chat**: Modern React frontend with streaming responses
- **Country-Specific Pricing**: Localized repair cost estimates in local currencies
- **Intelligent Learning**: Automatic knowledge acquisition from web sources with deduplication
- **Clean Source Categorization**: Separate internal knowledge, learned content, and web sources
- **Vehicle-Specific Prompting**: Smart detection of vehicle-specific queries with information gathering

## 🏗️ Project Structure

```
LangChain_ChatBot/
├── app/                          # Next.js application pages
├── components/                   # React components
├── backend_app/                  # FastAPI backend
│   ├── api/                      # API routes
│   ├── core/                     # Settings and configuration
│   ├── knowledge_base/           # Car knowledge documents
│   ├── models/                   # Pydantic models
│   ├── services/                 # Core business logic
│   └── utils/                    # Utility functions
├── data/                         # Vector store and databases
├── logs/                         # Application logs
└── docker-compose.yml            # Docker deployment
```

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- Python 3.9+
- OpenAI API key

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd LangChain_ChatBot
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Node.js dependencies:
```bash
npm install
```

4. Create `.env` file (copy from `env.example`):
```bash
cp env.example .env   # Windows: copy env.example .env
```

### Running the Application

1. Start the backend (from project root):
```bash
python run_backend.py
```

2. Start the frontend:
```bash
npm run dev
```

3. Open your browser to `http://localhost:3000`

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

## 🧪 Quality

- Linting (Python): `ruff check .`
- Formatting (Python): `black .`
- Type-checking (Python): `mypy backend_app`
- Frontend lint: `npm run lint`
- Frontend build: `npm run build`

## 🛠️ Core Features

### ✅ Intelligent Learning System
- **Automatic Knowledge Acquisition**: Learns from web sources during conversations
- **Deduplication**: Prevents duplicate content across multiple languages
- **Vehicle-Specific Matching**: Matches learned content to specific vehicle models
- **Fallback Mechanisms**: Ensures learned content is always retrievable

### ✅ Clean Source Categorization
- **Internal Knowledge**: Direct access to european_car_knowledge.py content
- **Previously Learned**: Content from learned_documents.jsonl
- **Web Sources**: Real-time web search results
- **Newly Learned**: Fresh content acquired during current session

### ✅ Vehicle Context Intelligence
- **Smart Detection**: Identifies vehicle-specific queries requiring detailed information
- **Model-Year Validation**: Validates vehicle/year combinations (e.g., prevents 2017 W204 claims)
- **Progressive Information Gathering**: Guides users to provide necessary vehicle details

### ✅ Enhanced Web Search Logic
- **Generic Content Detection**: Distinguishes between generic advice and model-specific information
- **Intelligent Triggering**: Only searches web when specific information is needed
- **European Focus**: Prioritizes European automotive sources and regulations

### ✅ Robust Architecture
- **Dependency Injection**: Proper service management and resource cleanup
- **Error Handling**: Comprehensive error management with user-friendly messages
- **Rate Limiting**: 30 requests per 15 minutes to prevent abuse
- **CORS Configuration**: Supports both development and production environments

## 🌍 Supported Countries

**Primary Focus**: Lithuania (LT) and Baltic region
**Supported**: All 27 EU countries plus UK, Norway, Switzerland

## 📖 API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔧 Configuration

Key settings in `backend_app/core/settings.py`:

- **OpenAI Models**: GPT-4 for chat, Ada-002 for embeddings
- **RAG Settings**: Chunk size, overlap, retrieval parameters
- **Rate Limiting**: 30 requests per 15 minutes
- **CORS**: Configured for development and production
- **Learning Thresholds**: Intelligent content relevancy scoring

## 📝 License

This project is licensed under the MIT License. See `LICENSE`.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if needed
5. Submit a pull request

## 🐛 Issues & Support

Please report issues via GitHub Issues with:
- Environment details
- Steps to reproduce
- Expected vs actual behavior 