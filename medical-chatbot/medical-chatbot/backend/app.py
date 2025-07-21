from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_engine import MedicalRAGEngine
import os
from dotenv import load_dotenv
import threading

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize RAG Engine - do this in a separate thread to not block app startup
rag_engine = None
initialization_complete = False
initialization_error = None

def initialize_engine():
    global rag_engine, initialization_complete, initialization_error
    try:
        rag_engine = MedicalRAGEngine()
        initialization_complete = True
    except Exception as e:
        initialization_error = str(e)
        initialization_complete = True
        print(f"Error initializing RAG engine: {e}")

# Start initialization in background
init_thread = threading.Thread(target=initialize_engine)
init_thread.daemon = True
init_thread.start()

@app.route('/status', methods=['GET'])
def status():
    """Check if the RAG engine is initialized"""
    if not initialization_complete:
        return jsonify({"status": "initializing"}), 202
    elif initialization_error:
        return jsonify({"status": "error", "message": initialization_error}), 500
    else:
        return jsonify({"status": "ready"}), 200

@app.route('/query', methods=['POST'])
def query():
    """Process a medical query"""
    if not initialization_complete:
        return jsonify({"error": "System is still initializing. Please try again in a moment."}), 503
    
    if initialization_error:
        return jsonify({"error": f"System initialization failed: {initialization_error}"}), 500
    
    # Get query from request
    data = request.json
    user_query = data.get('query')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Process query through RAG pipeline
        response, sources = rag_engine.query(user_query)
        
        return jsonify({
            "response": response,
            "sources": sources
        })
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": "An error occurred while processing your query"}), 500

@app.route('/process_pdfs', methods=['POST'])
def process_pdfs():
    """Endpoint to trigger PDF processing"""
    data = request.json
    directory = data.get('directory')
    
    if not directory:
        return jsonify({"error": "No directory provided"}), 400
    
    if not os.path.isdir(directory):
        return jsonify({"error": f"Directory '{directory}' does not exist"}), 400
    
    try:
        # Import here to avoid circular imports
        from pdf_processor import process_and_index_pdfs
        
        # Process PDFs in a separate thread
        def process_thread():
            process_and_index_pdfs(directory)
        
        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({"message": f"Processing PDFs from '{directory}' has started"}), 202
    
    except Exception as e:
        return jsonify({"error": f"Error starting PDF processing: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)