from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import uuid

from ChromaDBManager import ChromaDBManager;
import json


app = Flask(__name__)
CORS(app)  # To handle CORS if frontend and backend are on different servers

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'csv', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


CHROMA_DB_FOLDER = os.path.join(os.getcwd(), 'chroma_db')
collection_name  = 'ca17be90-b943-46a6-82aa-c118fd9e14c4_embedding'
aiClient = ChromaDBManager(persist_directory=CHROMA_DB_FOLDER,collection_name=collection_name)
files = []


# Check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for uploading files
@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'})
    
    uploaded_files = request.files.getlist('files')

    # List to store the details of uploaded files
    new_files = []
    
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = file.filename
            file_id = str(uuid.uuid4())  # Generate a unique ID for the file
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
    return jsonify({'files': files})   

# Route for uploading files
@app.route('/query', methods=['POST'])
def query():
    
    # Check if the request has content
    if not request.data:
        return jsonify({'error': 'Request body is empty'}), 400
    
    try:
        # Parse the raw stringified JSON into a Python dictionary
        data = json.loads(request.data)
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON format'}), 400
    
    # Get the 'query' from the parsed data
    query = data.get('query')
    session_id = data.get('session_id')


    # List to store the paths of uploaded files
    response = aiClient.ask_question(
            query=query,
            session_id=session_id
        )


    if response:
        return jsonify({'answer' : response['answer']})
    else:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
# Route to create a session ID
@app.route('/create_session', methods=['GET'])
def create_session():
    session_id = str(uuid.uuid4())  # Generate a unique session ID using uuid
    # Optionally, you can store this session ID in a session store like Redis or in-memory dict for tracking
    # For this example, we'll just return it as part of the response
    return jsonify({'session_id': session_id})







# Route to soft delete a file (set 'deleted' metadata)
@app.route('/soft_delete/<file_id>', methods=['POST'])
def soft_delete(file_id):
    deleted = request.json.get('deleted', True)  # default to True for deletion
    # Set soft delete using ChromaDBManager
    try:
        aiClient.soft_delete_document(file_id, collection_name, deleted)
        # Update the file's 'deleted' status in memory (files list)
        for file in files:
            if file['id'] == file_id:
                file['deleted'] = True if deleted else False
                break
        return jsonify({'message': f'File {file_id} soft deleted' if deleted else f'File {file_id} restored'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Route to hard delete a file (permanently remove it)
@app.route('/hard_delete/<file_id>', methods=['DELETE'])
def hard_delete(file_id):
    try:
        # Delete the file physically from the file system
        file_to_delete = next((file for file in files if file['id'] == file_id), None)
        if file_to_delete:
            os.remove(file_to_delete['path'])
            files.remove(file_to_delete)
            # Hard delete from ChromaDB
            aiClient.hard_delete_document(file_id, collection_name)
            return jsonify({'message': f'File {file_id} hard deleted'}), 200
        else:
            return jsonify({'error': f'File {file_id} not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
