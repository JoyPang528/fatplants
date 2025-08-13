from flask import Flask, request, jsonify
from citation_retrieval_module_python import CitationRetrievalService # Assuming your service is in this file
import asyncio
import os

app = Flask(__name__)

# Initialize the CitationRetrievalService
# It's good practice to get the API key from environment variables in a real app

ncbi_api_key = "your API key"

citation_service = CitationRetrievalService(ncbi_api_key=ncbi_api_key)

@app.route('/retrieve-citations', methods=['POST'])
def retrieve_citations_endpoint():
    """
    API endpoint to retrieve and rank citations for a given query.
    Expected JSON body: {"query": "your biomedical question", "num_citations": 3}
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query = data['query']
    num_citations = data.get('num_citations', 3) # Default to 3 citations

    try:
        # Run the async method from the service
        # Flask is typically sync, so we use asyncio.run to call the async method
        citations = asyncio.run(citation_service.retrieve_and_rank_citations(query, num_citations))
        return jsonify({"citations": citations}), 200
    except Exception as e:
        app.logger.error(f"Error retrieving citations: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For development, you can run this directly: python citation_api.py
    # For production, use a WSGI server like Gunicorn (e.g., gunicorn -w 4 -b 0.0.0.0:5001 citation_api:app)
    app.run(host='0.0.0.0', port=5001, debug=True) # Run on a different port than NestJS
