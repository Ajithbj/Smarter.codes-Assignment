# RAG-Style Website Content Search
  This is a Single-Page Application (SPA) designed to fetch content from a given URL, chunk it by token size, and perform a semantic search using vector embeddings to find the most relevant content chunks based on a user query.
  
✨ Key Features

  Frontend: Built with React for a responsive, single-page interface.

  Backend: Powered by FastAPI (Python) for high-performance and asynchronous processing.

  HTML Parsing & Cleaning: Uses BeautifulSoup to fetch HTML and strip unnecessary elements (scripts, styles, headers, footers).

  Tokenization & Chunking: Uses the Hugging Face transformers library (specifically the all-MiniLM-L6-v2 model tokenizer) to accurately chunk content into $\le 500$ token segments.

  Vector Database Simulation: Utilizes the Faiss library for efficient, in-memory indexing and semantic search, simulating the functionality of a production-grade vector database like Milvus or Weaviate.

🛠️ Setup and Running Locally
  This project consists of two independent services: the FastAPI backend and the React frontend. Both must be run concurrently.

Prerequisites
Python 3.8+ and pip

Node.js 16+ and npm or yarn

1. Backend Setup
Navigate to the backend/ directory:
Bash

cd backend
Install the required Python dependencies:

Bash

pip install -r requirements.txt
Note: The requirements file should include: fastapi, uvicorn, requests, pydantic, beautifulsoup4, transformers, torch, and faiss-cpu.

Run the FastAPI application using Uvicorn:

Bash

uvicorn main:app --reload
The backend should now be running at http://127.0.0.1:8000.

2. Frontend Setup
Navigate to the frontend/ directory:

Bash

cd ../frontend
(Assuming you use a standard React setup, where App.js is in src/)

Install the Node.js dependencies:

Bash

npm install or yarn install
Note: The only necessary dependencies are React, react-dom, and the lucide-react icons.

Start the React development server:

Bash

npm start or yarn start
The frontend should open automatically in your browser at http://localhost:3000.
 
Prerequisites and DependenciesBackend Dependencies (requirements.txt)Plaintextfastapi>=0.104.1
uvicorn[standard]>=0.23.2
pydantic>=2.4.2
requests>=2.31.0
beautifulsoup4>=4.12.2
transformers>=4.34.0
torch>=2.1.0
numpy>=1.26.1
faiss-cpu>=1.7.4
Frontend Dependencies (package.json)JSON{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "lucide-react": "^0.292.0"
  },
  "scripts": {
    "start": "react-scripts start"
    // ... other scripts
  }
}

⚙️ Vector Database/Faiss ConfigurationThe application uses Faiss for its vector search capability, simulating an external vector database.Indexing: On every /search request, the following steps are performed:The retrieved and chunked text content is converted into embeddings using the sentence-transformers/all-MiniLM-L6-v2 model.A new in-memory Faiss index (IndexFlatIP) is created from these embeddings.Search: The search query is also embedded, normalized, and used to query the in-memory Faiss index to retrieve the $K=10$ most relevant chunk indices based on Inner Product (a proxy for Cosine Similarity).
