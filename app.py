import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

from services.pdf_processing import extract_text_from_pdf, chunk_text
from services.embedding_utils import get_embedding
from services.vector_store import VectorStore
from services.qna_pipeline import answer_query

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize a single in-memory vector store
# If you want a separate store per user or session, you can handle that differently.
vector_store = VectorStore(embedding_dim=1536)  # 1536 for text-embedding-ada-002

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle PDF uploads
        if "file" in request.files:
            files = request.files.getlist("file")
            for f in files:
                filename = secure_filename(f.filename)
                if filename.lower().endswith(".pdf"):
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    f.save(filepath)
                    # Process PDF
                    text = extract_text_from_pdf(filepath)
                    chunks = chunk_text(text, max_tokens=500)
                    # Create embeddings for each chunk and add to vector store
                    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
                    vector_store.add_embeddings(chunk_embeddings, chunks)

        return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():    
    data = request.get_json()
    query = data.get("question", "")
    if not query.strip():
        return {"answer": "No question provided."}, 400
    
    # Use QnA pipeline to get answer
    answer = answer_query(vector_store, query, top_k=3)
    return {"answer": answer}, 200


if __name__ == "__main__":
    # For local development
    app.run(debug=True)