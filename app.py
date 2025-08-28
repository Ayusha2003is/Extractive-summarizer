# app.py
import sys
import os

# Path to SummarizationModel folder
summarization_dir = os.path.join(os.path.dirname(__file__), '..', 'SummarizationModel')
summarization_dir = os.path.abspath(summarization_dir)

# Add to Python path
if summarization_dir not in sys.path:
    sys.path.insert(0, summarization_dir)

# Now you can import
from model_classes import (
    ImprovedRNNEncoder,
    ImprovedExtractiveRNNSummarizer,
    ImprovedSentenceEncoder,
    ImprovedBinaryClassifier,
    TextPreprocessor
)


from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
import traceback
import os
import sys
import pickle

# Fix the path to SummarizationModel directory
current_file_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current file directory: {current_file_dir}")

# Go up one level to backend, then into SummarizationModel
backend_dir = os.path.dirname(current_file_dir) if 'Flask' in current_file_dir else current_file_dir
summarization_dir = os.path.join(backend_dir, 'SummarizationModel')

print(f"Looking for SummarizationModel at: {summarization_dir}")
print(f"Directory exists: {os.path.exists(summarization_dir)}")

if summarization_dir not in sys.path:
    sys.path.insert(0, summarization_dir)

# Check if model_classes.py exists
model_classes_file = os.path.join(summarization_dir, 'model_classes.py')
print(f"model_classes.py exists: {os.path.exists(model_classes_file)}")

# Import the classes BEFORE trying to load the model
try:
    from model_classes import (
        ImprovedRNNEncoder, 
        ImprovedExtractiveRNNSummarizer, 
        ImprovedSentenceEncoder,
        ImprovedBinaryClassifier,
        TextPreprocessor
    )
    print("✅ Successfully imported SummarizationModel classes")
except Exception as e:
    print(f"❌ Failed to import model classes: {e}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths

# ================================
# App Initialization
# ================================
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"], supports_credentials=True)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///mydb.sqlite3')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwtsecret')

bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# ================================
# Database & Auth Setup
# ================================
try:
    from models import db
    db.init_app(app)

    from auth import auth, set_bcrypt_instance
    set_bcrypt_instance(bcrypt)
    app.register_blueprint(auth, url_prefix="/auth")

    with app.app_context():
        db.create_all()
        print("Database tables created successfully")
    DB_AVAILABLE = True
except Exception as e:
    print(f"Warning: Database setup failed: {e}")
    DB_AVAILABLE = False

# ================================
# Model Loading Functions (Inline)
# ================================
def load_model_inline():
    """Load the model directly in app.py"""
    # Fix the model path - go up one level if we're in Flask directory
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_file_dir) if 'Flask' in current_file_dir else current_file_dir
    MODEL_PATH = os.path.join(backend_dir, 'SummarizationModel', 'fast_extractive_model.pkl')
    
    try:
        print(f"Loading model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print("❌ Model file not found!")
            return None, None
        
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        
        print(f"Loaded data type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print(f"Dictionary keys: {list(model_data.keys())}")
            
            if "model" in model_data and "preprocessor" in model_data:
                print("Found 'model' and 'preprocessor' keys")
                model = model_data["model"]
                preprocessor = model_data["preprocessor"]
                
                print(f"Model type: {type(model)}")
                print(f"Preprocessor type: {type(preprocessor)}")
                print(f"Model is None: {model is None}")
                print(f"Preprocessor is None: {preprocessor is None}")
                
                if model is None or preprocessor is None:
                    print("❌ Model or preprocessor is None after loading!")
                    return None, None
                
                print("✅ Model loaded successfully!")
                return model, preprocessor
            else:
                print(f"❌ Pickle file did not contain (model, preprocessor)")
                print(f"Available keys: {list(model_data.keys())}")
                return None, None
        else:
            print("❌ Unsupported pickle format")
            return None, None
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(traceback.format_exc())
        return None, None

def generate_summary_inline(model, preprocessor, text, max_sentences=3, threshold=0.5):
    """Generate summary using the loaded model"""
    try:
        # Import the sentence splitting function
        import re
        
        def split_into_sentences(text, max_length=30):
            sentences = re.split(r'[.!?]+', text)
            processed = []
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 0:
                    words = sent.split()
                    if len(words) > max_length:
                        for i in range(0, len(words), max_length):
                            chunk = ' '.join(words[i:i + max_length])
                            if chunk.strip():
                                processed.append(chunk.strip())
                    elif len(words) >= 3:
                        processed.append(sent)
            return processed[:10]
        
        # Split text into sentences
        sentences_text = split_into_sentences(text)
        if not sentences_text:
            return text
        
        print(f"Split sentences: {len(sentences_text)}")
        
        # Convert sentences to indices
        sentences_indices = [preprocessor.text_to_indices(sent) for sent in sentences_text]
        
        # Get model predictions
        probabilities, _ = model.forward(sentences_indices, training=False)
        print(f"Model probabilities: {probabilities}")
        
        # Select sentences based on probabilities and threshold
        selected_indices = []
        sorted_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)
        
        for idx in sorted_indices[:max_sentences]:
            if probabilities[idx] >= threshold:
                selected_indices.append(idx)
        
        # If no sentences meet threshold, select top sentences anyway
        if not selected_indices:
            selected_indices = sorted_indices[:min(max_sentences, len(sentences_text))]
        
        # Sort selected indices to maintain original order
        selected_indices.sort()
        print(f"Selected sentence indices: {selected_indices}")
        
        # Generate summary
        summary_sentences = [sentences_text[i] for i in selected_indices]
        summary = '. '.join(summary_sentences) + '.' if summary_sentences else text
        
        return summary
        
    except Exception as e:
        print(f"Error in generate_summary_inline: {e}")
        print(traceback.format_exc())
        # Fallback to simple extraction
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return '. '.join(sentences[:max_sentences]) + '.' if len(sentences) > max_sentences else text

# ================================
# Summarization Model Setup
# ================================
model, preprocessor = None, None
MODEL_AVAILABLE = False

print("Attempting to load summarization model...")
model, preprocessor = load_model_inline()

if model is not None and preprocessor is not None:
    MODEL_AVAILABLE = True
    print("✅ Summarization model loaded successfully")
else:
    print("⚠ Model loading returned None - using fallback mode")

print(f"Final model status - Available: {MODEL_AVAILABLE}, Loaded: {model is not None}")

# ================================
# Routes
# ================================

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        # Input validation
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Text too short to summarize'}), 400
            
        # Get optional parameters
        max_sentences = data.get('max_sentences', 3)
        threshold = data.get('threshold', 0.5)
        
        # Validate parameters
        if not isinstance(max_sentences, int) or max_sentences < 1 or max_sentences > 10:
            max_sentences = 3
            
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            threshold = 0.5
        
        print(f"Summarizing text (length: {len(text)}, max_sentences: {max_sentences})")
        
        # Generate summary
        if MODEL_AVAILABLE and model and preprocessor:
            try:
                summary = generate_summary_inline(model, preprocessor, text, max_sentences, threshold)
                model_used = 'trained'
                print("✓ Used trained model for summarization")
            except Exception as e:
                print(f"⚠ Trained model failed, using fallback: {e}")
                # Fallback if trained model fails
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                summary = '. '.join(sentences[:max_sentences]) + '.' if len(sentences) > max_sentences else text
                model_used = 'fallback_after_error'
        else:
            # Fallback extractive summary
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            summary = '. '.join(sentences[:max_sentences]) + '.' if len(sentences) > max_sentences else text
            model_used = 'fallback'
            print("Using fallback summarization")
        
        # Ensure summary is not empty
        if not summary or len(summary.strip()) == 0:
            summary = text[:200] + "..." if len(text) > 200 else text
            model_used += '_emergency_fallback'
        
        return jsonify({
            'summary': summary.strip(),
            'model_used': model_used,
            'status': 'success',
            'original_length': len(text),
            'summary_length': len(summary.strip())
        })
        
    except Exception as e:
        print("Error in /summarize:", traceback.format_exc())
        return jsonify({
            'error': str(e), 
            'status': 'error',
            'message': 'An unexpected error occurred during summarization'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'model_available': MODEL_AVAILABLE,
            'model_loaded': model is not None,
            'preprocessor_loaded': preprocessor is not None,
            'database_available': DB_AVAILABLE
        }
        
        # Test model if available
        if MODEL_AVAILABLE and model and preprocessor:
            try:
                test_summary = generate_summary_inline(model, preprocessor, "This is a test. It should work.")
                health_status['model_test'] = 'passed' if test_summary else 'failed'
            except Exception as e:
                health_status['model_test'] = f'failed: {str(e)}'
        else:
            health_status['model_test'] = 'skipped'
        
        return jsonify(health_status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get detailed model information"""
    try:
        info = {
            'model_available': MODEL_AVAILABLE,
            'model_loaded': model is not None,
            'preprocessor_loaded': preprocessor is not None,
            'model_type': type(model).__name__ if model else None,
            'preprocessor_type': type(preprocessor).__name__ if preprocessor else None,
        }
        
        if model:
            info['vocab_size'] = getattr(model, 'vocab_size', 'unknown')
            info['embed_dim'] = getattr(model, 'embed_dim', 'unknown')
            info['hidden_dim'] = getattr(model, 'hidden_dim', 'unknown')
                
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ================================
# Error Handlers
# ================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Request too large"}), 413

# ================================
# Run Server
# ================================
if __name__ == "__main__":
    print("=" * 50)
    print("FLASK SERVER STARTUP SUMMARY")
    print("=" * 50)
    print(f"Database Available: {DB_AVAILABLE}")
    print(f"Model Available: {MODEL_AVAILABLE}")
    print(f"Model Loaded: {model is not None}")
    print(f"Preprocessor Loaded: {preprocessor is not None}")
    print("=" * 50)
    print("Starting Flask server...")
    app.run(debug=True, host='127.0.0.1', port=5000)