import os
from pathlib import Path
from typing import Any

import gradio as gr
import requests


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def _safe_json(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except Exception:
        return {
            "status_code": response.status_code,
            "text": response.text,
        }


def health_check():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        return _safe_json(response)
    except Exception as exc:
        return {"error": str(exc)}


def upload_pdf(user_id: str, project_id: str, pdf_file):
    if not user_id or not project_id:
        return {"error": "user_id and project_id are required."}

    if pdf_file is None:
        return {"error": "Please upload a PDF file."}

    file_path = Path(pdf_file.name)

    try:
        with file_path.open("rb") as f:
            files = {
                "file": (
                    file_path.name,
                    f,
                    "application/pdf",
                )
            }
            data = {
                "user_id": user_id,
                "project_id": project_id,
            }

            response = requests.post(
                f"{BACKEND_URL}/upload/raw",
                data=data,
                files=files,
                timeout=60,
            )

        return _safe_json(response)

    except Exception as exc:
        return {"error": str(exc)}


def index_pdf(user_id: str, project_id: str, filename: str):
    if not user_id or not project_id or not filename:
        return {"error": "user_id, project_id, and filename are required."}

    try:
        response = requests.post(
            f"{BACKEND_URL}/index/pdf",
            json={
                "user_id": user_id,
                "project_id": project_id,
                "filename": filename,
            },
            timeout=300,
        )
        return _safe_json(response)

    except Exception as exc:
        return {"error": str(exc)}


def search_docs(user_id: str, project_id: str, query: str, top_k: int):
    try:
        response = requests.post(
            f"{BACKEND_URL}/search",
            json={
                "user_id": user_id,
                "project_id": project_id,
                "query": query,
                "top_k": top_k,
            },
            timeout=120,
        )
        return _safe_json(response)

    except Exception as exc:
        return {"error": str(exc)}


def ask_docs(user_id: str, project_id: str, question: str, top_k: int):
    try:
        response = requests.post(
            f"{BACKEND_URL}/ask",
            json={
                "user_id": user_id,
                "project_id": project_id,
                "question": question,
                "top_k": top_k,
            },
            timeout=300,
        )
        data = _safe_json(response)

        if "answer" not in data:
            return data, {}, {}

        return data.get("answer", ""), data.get("citations", []), data.get("metrics", {})

    except Exception as exc:
        return {"error": str(exc)}, {}, {}


def compare_docs(user_id: str, project_id: str, question: str, top_k: int):
    try:
        response = requests.post(
            f"{BACKEND_URL}/compare",
            json={
                "user_id": user_id,
                "project_id": project_id,
                "question": question,
                "top_k": top_k,
            },
            timeout=300,
        )
        data = _safe_json(response)

        if "answer" not in data:
            return data, {}, {}

        return data.get("answer", ""), data.get("citations", []), data.get("metrics", {})

    except Exception as exc:
        return {"error": str(exc)}, {}, {}


def lit_review(user_id: str, project_id: str, topic: str, top_k: int):
    try:
        response = requests.post(
            f"{BACKEND_URL}/lit-review",
            json={
                "user_id": user_id,
                "project_id": project_id,
                "topic": topic,
                "top_k": top_k,
            },
            timeout=300,
        )
        data = _safe_json(response)

        if "answer" not in data:
            return data, {}, {}

        return data.get("answer", ""), data.get("citations", []), data.get("metrics", {})

    except Exception as exc:
        return {"error": str(exc)}, {}, {}


def extract_insights(user_id: str, project_id: str, focus: str, top_k: int):
    try:
        response = requests.post(
            f"{BACKEND_URL}/extract-insights",
            json={
                "user_id": user_id,
                "project_id": project_id,
                "focus": focus,
                "top_k": top_k,
            },
            timeout=300,
        )
        data = _safe_json(response)

        if "answer" not in data:
            return data, {}, {}

        return data.get("answer", ""), data.get("citations", []), data.get("metrics", {})

    except Exception as exc:
        return {"error": str(exc)}, {}, {}


def get_memory(user_id: str, project_id: str):
    try:
        response = requests.get(
            f"{BACKEND_URL}/memory/{user_id}/{project_id}",
            timeout=30,
        )
        return _safe_json(response)

    except Exception as exc:
        return {"error": str(exc)}


def get_vector_stats():
    try:
        response = requests.get(f"{BACKEND_URL}/vector/stats", timeout=30)
        return _safe_json(response)
    except Exception as exc:
        return {"error": str(exc)}


def get_latest_metrics():
    try:
        response = requests.get(f"{BACKEND_URL}/metrics/latest", timeout=30)
        return _safe_json(response)
    except Exception as exc:
        return {"error": str(exc)}


with gr.Blocks(title="Soundable Research Agent") as demo:
    gr.Markdown(
        """
        # Soundable Research Agent

        Local multi-user research automation MVP with:
        - PDF upload
        - Chroma vector search
        - cited RAG QA
        - LangGraph workflow
        - research automation tools
        - SQLite memory
        - local LLM serving abstraction
        """
    )

    with gr.Row():
        user_id = gr.Textbox(label="User ID", value="hojune")
        project_id = gr.Textbox(label="Project ID", value="test-project")
        top_k = gr.Slider(label="Top K", minimum=1, maximum=10, value=5, step=1)

    with gr.Tab("1. Upload & Index"):
        pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Upload PDF")
        upload_output = gr.JSON(label="Upload Result")

        filename = gr.Textbox(label="Filename to index", placeholder="Use filename returned by upload")
        index_btn = gr.Button("Index PDF into Chroma")
        index_output = gr.JSON(label="Index Result")

        upload_btn.click(
            upload_pdf,
            inputs=[user_id, project_id, pdf_file],
            outputs=upload_output,
        )

        index_btn.click(
            index_pdf,
            inputs=[user_id, project_id, filename],
            outputs=index_output,
        )

    with gr.Tab("2. Search"):
        search_query = gr.Textbox(label="Search Query", lines=2)
        search_btn = gr.Button("Search Vector DB")
        search_output = gr.JSON(label="Search Results")

        search_btn.click(
            search_docs,
            inputs=[user_id, project_id, search_query, top_k],
            outputs=search_output,
        )

    with gr.Tab("3. Ask"):
        question = gr.Textbox(label="Question", lines=3)
        ask_btn = gr.Button("Ask")
        answer_output = gr.Markdown(label="Answer")
        citations_output = gr.JSON(label="Citations")
        metrics_output = gr.JSON(label="Metrics")

        ask_btn.click(
            ask_docs,
            inputs=[user_id, project_id, question, top_k],
            outputs=[answer_output, citations_output, metrics_output],
        )

    with gr.Tab("4. Research Tools"):
        compare_question = gr.Textbox(
            label="Compare Request",
            value="Compare the methods, strengths, limitations, and future research directions of the uploaded papers.",
            lines=3,
        )
        compare_btn = gr.Button("Compare Papers")
        compare_answer = gr.Markdown(label="Comparison")
        compare_citations = gr.JSON(label="Citations")
        compare_metrics = gr.JSON(label="Metrics")

        compare_btn.click(
            compare_docs,
            inputs=[user_id, project_id, compare_question, top_k],
            outputs=[compare_answer, compare_citations, compare_metrics],
        )

        lit_topic = gr.Textbox(
            label="Literature Review Topic",
            value="memory mechanisms and retrieval-augmented planning in LLM agents",
            lines=2,
        )
        lit_btn = gr.Button("Generate Literature Review")
        lit_answer = gr.Markdown(label="Literature Review")
        lit_citations = gr.JSON(label="Citations")
        lit_metrics = gr.JSON(label="Metrics")

        lit_btn.click(
            lit_review,
            inputs=[user_id, project_id, lit_topic, top_k],
            outputs=[lit_answer, lit_citations, lit_metrics],
        )

        focus = gr.Textbox(
            label="Insight Extraction Focus",
            value="methods, datasets, experiments, limitations, and future work",
            lines=2,
        )
        extract_btn = gr.Button("Extract Insights")
        extract_answer = gr.Markdown(label="Insights")
        extract_citations = gr.JSON(label="Citations")
        extract_metrics = gr.JSON(label="Metrics")

        extract_btn.click(
            extract_insights,
            inputs=[user_id, project_id, focus, top_k],
            outputs=[extract_answer, extract_citations, extract_metrics],
        )

    with gr.Tab("5. Memory & Stats"):
        memory_btn = gr.Button("Load Memory")
        memory_output = gr.JSON(label="Project Memory")

        vector_btn = gr.Button("Vector DB Stats")
        vector_output = gr.JSON(label="Vector Stats")

        metrics_btn = gr.Button("Latest Metrics")
        latest_metrics_output = gr.JSON(label="Latest Metrics")

        health_btn = gr.Button("Backend Health")
        health_output = gr.JSON(label="Health")

        memory_btn.click(
            get_memory,
            inputs=[user_id, project_id],
            outputs=memory_output,
        )

        vector_btn.click(
            get_vector_stats,
            inputs=[],
            outputs=vector_output,
        )

        metrics_btn.click(
            get_latest_metrics,
            inputs=[],
            outputs=latest_metrics_output,
        )

        health_btn.click(
            health_check,
            inputs=[],
            outputs=health_output,
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
    )