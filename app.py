from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import uuid
from ChromaDBManager import ChromaDBManager;
import json
from dotenv import load_dotenv


load_dotenv()

os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)  # To handle CORS if frontend and backend are on different servers

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'csv', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set chroma_db cnfigurations
CHROMA_DB_FOLDER = os.path.join(os.getcwd(), 'chroma_db')
collection_name  = 'ca17be90-b943-46a6-82aa-c118fd9e14c4_embedding'
aiClient = ChromaDBManager(persist_directory=CHROMA_DB_FOLDER,collection_name=collection_name)
files=[]

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')


# Route for uploading files
@app.route('/upload', methods=['POST'])
def upload_files():
    print('hello')
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'})
    
    uploaded_files = request.files.getlist('files')

    # List to store the details of uploaded files
    new_files = []
    
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = file.filename
            file_id = str(uuid.uuid4())
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file to the upload folder
            file.save(file_path)
            
            # Store file details in the files array
            file_details = {
                'id': file_id,
                'name': filename,
                'path': file_path, 
                'deleted' : False
            }
            files.append(file_details)
            new_files.append(file_details)
            
            try:
                # Process the file with aiClient
                aiClient.genrate_text_chunks(os.path.join(os.getcwd(), file_path), file_id)
            except Exception as e:
                print(f'Error processing file {filename}: {e}')
    
    if new_files:
        return jsonify({'message': 'Files successfully uploaded', 'files': new_files})
    else:
        return jsonify({'error': 'No valid files uploaded'}), 400


# Route to list all uploaded files
@app.route('/files', methods=['GET'])
def list_files():
    print(f'Files :::: {files} ')
    return jsonify({'files': files})  


# Route to create a session ID
@app.route('/create_session', methods=['GET'])
def create_session():
    session_id = str(uuid.uuid4())  # Generate a unique session ID using uuid
    return jsonify({'session_id': session_id})


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)