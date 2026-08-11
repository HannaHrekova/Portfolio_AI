import datasets
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import YouTubeSearchTool
from pydantic import BaseModel, Field
import os

# Web Search Tool
class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(description="The search query string for the internet.")

web_search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description="Searches the web for general information, people, or topics. Always use a clear and concise search query.",
    args_schema=DuckDuckGoSearchInput
)

# Youtube Video Q&A Tool
youtube_qa_tool = YouTubeSearchTool(
    name="youtube_video_qa",
    description="Useful for answering specific questions about a YouTube video. Input should be a JSON with 'url' (the video URL) and 'question' (the question about the video)."
)

# Function to load a dataset and build a combined retriever
def load_guest_dataset():
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    docs = []
    doc_embeddings = []
    for guest in guest_dataset:
        text = "\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ])
        docs.append(Document(
            page_content=text,
            metadata={"name": guest["name"]}
        ))
        doc_embeddings.append(model.encode(text))

    bm25_retriever = BM25Retriever.from_documents(docs)

    def retrieve_guest_info(query: str) -> str:
        """Retrieves guest information from the dataset based on a query."""
        try:
            query_embedding = model.encode(query)
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            sem_index = similarities.argmax()
            sem_score = similarities[sem_index]
            sem_doc = docs[sem_index]

            bm25_results = bm25_retriever.invoke(query)
            bm25_doc = bm25_results[0] if bm25_results else None

            if sem_score > 0.75 and sem_doc:
                return sem_doc.page_content
            elif bm25_doc:
                return bm25_doc.page_content
            else:
                return "I don't know"
        except Exception:
            return "I don't know"

    guest_info_tool = Tool(
        name="guest_info_retriever",
        func=retrieve_guest_info,
        description="Retrieves specific guest information from a predefined dataset. Use this when the user asks about a specific person from the guest list, e.g., 'who is Dr. Nikola Tesla?' or 'tell me about Lady Ada Lovelace'."
    )

    return guest_info_tool

