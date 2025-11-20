import React, { useState } from 'react';
import { Search, Globe, ChevronRight, Loader2, Link, FileText } from 'lucide-react';


const API_BASE_URL = 'http://127.0.0.1:8000'; 

const ResultCard = ({ result, index }) => (
    <div className="result-card">
        <div className="result-card-header">
            <span className="match-badge">
                Match #{index + 1}
            </span>
            <div className="match-score">
                <FileText className="icon-small" />
                Score: {result.score.toFixed(4)}
            </div>
        </div>
        <p className="chunk-content">
            {result.chunk_content}
        </p>
        <div className="token-range">
            Source Token Range: {result.token_start} - {result.token_end}
        </div>
    </div>
);

const App = () => {
    const [url, setUrl] = useState('');
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isSearchPerformed, setIsSearchPerformed] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);
        setResults([]);
        setLoading(true);
        setIsSearchPerformed(true);

        if (!url || !query) {
            setError("Please enter both a valid URL and a search query.");
            setLoading(false);
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url, query }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            setResults(data.results || []);
        } catch (err) {
            console.error("Fetch Error:", err);
            setError(`Could not connect to the backend or process the request: ${err.message}. Ensure the FastAPI server is running.`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app-container">
            {/* Embedded Styles */}
            <style jsx="true">{`
                /* General Setup & Typography */
                .app-container {
                    min-height: 100vh;
                    background-color: #f3f4f6; /* bg-gray-100 */
                    padding: 1rem; /* p-4 */
                    font-family: 'Inter', sans-serif;
                }
                @media (min-width: 640px) { /* sm:p-8 */
                    .app-container {
                        padding: 2rem;
                    }
                }

                /* Header */
                .header {
                    margin-bottom: 2rem;
                    text-align: center;
                }
                .header-title {
                    font-size: 2.25rem; /* text-4xl */
                    font-weight: 800; /* font-extrabold */
                    color: #111827; /* text-gray-900 */
                    letter-spacing: -0.025em; /* tracking-tight */
                }
                .header-title svg {
                    width: 2rem;
                    height: 2rem;
                    margin-right: 0.5rem;
                    color: #4f46e5; /* text-indigo-600 */
                    display: inline-block;
                    vertical-align: middle;
                }
                @media (min-width: 640px) {
                    .header-title {
                        font-size: 3rem; /* sm:text-5xl */
                    }
                }
                .header-subtitle {
                    margin-top: 0.5rem;
                    font-size: 1.125rem; /* text-lg */
                    color: #4b5563; /* text-gray-600 */
                }

                /* Form Card */
                .form-card {
                    max-width: 48rem; /* max-w-4xl */
                    margin-left: auto;
                    margin-right: auto;
                    background-color: #ffffff;
                    padding: 1.5rem; /* p-6 */
                    border-radius: 0.75rem; /* rounded-xl */
                    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04); /* shadow-2xl */
                }
                @media (min-width: 640px) {
                    .form-card {
                        padding: 2rem; /* sm:p-8 */
                    }
                }
                .form-layout > div:not(:last-child) {
                    margin-bottom: 1.5rem; /* space-y-6 */
                }

                /* Inputs */
                .input-label {
                    display: flex;
                    align-items: center;
                    font-size: 0.875rem; /* text-sm */
                    font-weight: 500; /* font-medium */
                    color: #374151; /* text-gray-700 */
                    margin-bottom: 0.5rem;
                }
                .input-label svg {
                    width: 1rem;
                    height: 1rem;
                    margin-right: 0.5rem;
                    color: #6366f1; /* text-indigo-500 */
                }
                .input-field {
                    width: 100%;
                    padding: 0.75rem;
                    border: 1px solid #d1d5db; /* border-gray-300 */
                    border-radius: 0.5rem; /* rounded-lg */
                    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-sm */
                    font-size: 0.875rem; /* text-sm */
                    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
                }
                .input-field:focus {
                    outline: 2px solid transparent;
                    outline-offset: 2px;
                    border-color: #6366f1; /* focus:border-indigo-500 */
                    box-shadow: 0 0 0 1px #6366f1; /* focus:ring-indigo-500 (simulated) */
                }

                /* Button */
                .submit-button {
                    width: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 0.75rem 1.5rem; /* px-6 py-3 */
                    font-size: 1rem; /* text-base */
                    font-weight: 500; /* font-medium */
                    border-radius: 0.75rem; /* rounded-xl */
                    color: #ffffff;
                    background-color: #4f46e5; /* bg-indigo-600 */
                    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); /* shadow-lg */
                    transition: background-color 0.2s ease, opacity 0.2s ease;
                    border: none;
                    cursor: pointer;
                }
                .submit-button:hover {
                    background-color: #4338ca; /* hover:bg-indigo-700 */
                }
                .submit-button:focus {
                    outline: 2px solid transparent;
                    outline-offset: 2px;
                    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.5); /* focus:ring-indigo-500 (simulated) */
                }
                .submit-button:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
                .submit-button svg {
                    width: 1.25rem;
                    height: 1.25rem;
                    margin-right: 0.5rem;
                }
                .loader-icon {
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }

                /* Results Area */
                .results-container {
                    max-width: 48rem; /* max-w-4xl */
                    margin-left: auto;
                    margin-right: auto;
                    margin-top: 2.5rem; /* mt-10 */
                }
                .results-heading {
                    font-size: 1.5rem; /* text-2xl */
                    font-weight: 700; /* font-bold */
                    color: #111827; /* text-gray-900 */
                    margin-bottom: 1.5rem;
                    display: flex;
                    align-items: center;
                }
                .results-heading svg {
                    width: 1.5rem;
                    height: 1.5rem;
                    margin-right: 0.75rem;
                    color: #4f46e5; /* text-indigo-600 */
                }

                /* Messages */
                .alert-message {
                    padding: 1rem;
                    border-left: 4px solid;
                    border-radius: 0.5rem;
                }
                .alert-message p {
                    font-size: 1rem;
                }
                .alert-message .font-bold {
                    font-weight: 700;
                }
                .error-message {
                    background-color: #fee2e2; /* bg-red-100 */
                    border-left-color: #ef4444; /* border-red-500 */
                    color: #b91c1c; /* text-red-700 */
                }
                .no-matches {
                    background-color: #fffbeb; /* bg-yellow-100 */
                    border-left-color: #f59e0b; /* border-yellow-500 */
                    color: #b45309; /* text-yellow-700 */
                }

                /* Results List */
                .results-list > div:not(:last-child) {
                    margin-bottom: 1.5rem; /* space-y-6 */
                }

                /* Result Card */
                .result-card {
                    background-color: #ffffff;
                    padding: 1.25rem; /* p-5 */
                    border-radius: 0.5rem; /* rounded-lg */
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.06); /* shadow-md */
                    border-left: 4px solid #6366f1; /* border-l-4 border-indigo-500 */
                    transition: box-shadow 0.3s ease;
                }
                .result-card:hover {
                    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1); /* hover:shadow-lg */
                }

                .result-card-header {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin-bottom: 0.75rem;
                }
                .match-badge {
                    font-size: 0.875rem; /* text-sm */
                    font-weight: 600; /* font-semibold */
                    color: #4f46e5; /* text-indigo-600 */
                    background-color: #eef2ff; /* bg-indigo-50 */
                    padding: 0.25rem 0.75rem; /* px-3 py-1 */
                    border-radius: 9999px; /* rounded-full */
                }
                .match-score {
                    font-size: 0.75rem; /* text-xs */
                    color: #6b7280; /* text-gray-500 */
                    display: flex;
                    align-items: center;
                }
                .icon-small {
                    width: 0.75rem;
                    height: 0.75rem;
                    margin-right: 0.25rem;
                }
                
                .chunk-content {
                    color: #374151; /* text-gray-700 */
                    font-size: 0.875rem; /* text-sm */
                    font-style: italic;
                    font-family: monospace;
                    background-color: #f9fafb; /* bg-gray-50 */
                    padding: 0.75rem;
                    border-radius: 0.375rem; /* rounded-md */
                    overflow-x: auto;
                    white-space: pre-wrap;
                }
                .token-range {
                    margin-top: 0.75rem;
                    font-size: 0.75rem; /* text-xs */
                    color: #9ca3af; /* text-gray-400 */
                }

                /* Footer */
                .footer {
                    margin-top: 3rem; /* mt-12 */
                    text-align: center;
                    font-size: 0.875rem; /* text-sm */
                    color: #6b7280; /* text-gray-500 */
                }
            `}</style>

            <header className="header">
                <h1 className="header-title">
                    <Search />
                    Website Content Search
                </h1>
                <p className="header-subtitle">Search through website Content with precision</p>
            </header>

            {/* Input Form */}
            <div className="form-card">
                <form onSubmit={handleSubmit} className="form-layout">
                    {/* URL Input */}
                    <div>
                        <label htmlFor="url" className="input-label">
                            <Globe />
                            Website URL
                        </label>
                        <input
                            type="url"
                            id="url"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="e.g., https://en.wikipedia.org/wiki/React_(web_framework)"
                            required
                            className="input-field"
                        />
                    </div>

                    {/* Query Input */}
                    <div>
                        <label htmlFor="query" className="input-label">
                            <Search />
                            Search Query
                        </label>
                        <input
                            type="text"
                            id="query"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="e.g., benefits of component-based architecture"
                            required
                            className="input-field"
                        />
                    </div>

                    {/* Submit Button */}
                    <button
                        type="submit"
                        disabled={loading}
                        className="submit-button"
                    >
                        {loading ? (
                            <Loader2 className="loader-icon" />
                        ) : (
                            <ChevronRight />
                        )}
                        {loading ? 'Processing Content...' : 'Search'}
                    </button>
                </form>
            </div>

            {/* Results Area */}
            <div className="results-container">
                <h2 className="results-heading">
                    <Link />
                    Top Search Matches
                </h2>

                {error && (
                    <div className="alert-message error-message" role="alert">
                        <p className="font-bold">Error</p>
                        <p>{error}</p>
                    </div>
                )}

                {!loading && isSearchPerformed && results.length === 0 && !error && (
                    <div className="alert-message no-matches">
                        <p className="font-bold">No Matches Found</p>
                        <p>The search yielded no relevant results for your query in the provided URL.</p>
                    </div>
                )}

                {results.length > 0 && (
                    <div className="results-list">
                        {results.map((result, index) => (
                            <ResultCard key={index} result={result} index={index} />
                        ))}
                    </div>
                )}
            </div>

            <footer className="footer">
                <p>Simulated RAG Search using FastAPI and In-Memory Vector Index (Faiss).</p>
            </footer>
        </div>
    );
};

export default App;
