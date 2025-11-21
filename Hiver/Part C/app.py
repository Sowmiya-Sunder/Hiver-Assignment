from flask import Flask, render_template, request
from rag_engine import retrieve, generate_answer

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    retrieved_docs = []
    answer = None

    if request.method == "POST":
        query = (request.form.get("query") or "").strip()
        if query:
            retrieved_docs = retrieve(query)
            answer = generate_answer(query, retrieved_docs)

    return render_template(
        "index.html",
        query=query,
        retrieved_docs=retrieved_docs,
        answer=answer,
    )


if __name__ == "__main__":
    app.run(debug=True)
