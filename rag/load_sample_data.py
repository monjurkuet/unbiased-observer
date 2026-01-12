from urllib.parse import quote_plus

import google.generativeai as genai
import psycopg
from dotenv import dotenv_values
from google.generativeai import GenerativeModel

config = dotenv_values(".env")
# Configure API
genai.configure(api_key=config["GOOGLE_API_KEY"])

# Connection
conn_str = (
    f"postgresql://"
    f"{quote_plus(config['DB_USER'])}:"
    f"{quote_plus(config['DB_PASSWORD'])}@"
    f"{config['DB_HOST']}:{config['DB_PORT']}/"
    f"{config['DB_NAME']}"
)

# Sample data
entities = [
    {
        "name": "Dr. Sarah Chen",
        "type": "person",
        "content": "Chief AI Researcher with 15 years of experience in machine learning and neural networks. Published over 50 papers in top-tier conferences.",
        "summary": "Chief AI Researcher",
    },
    {
        "name": "Project Alpha",
        "type": "project",
        "content": "Advanced autonomous agent research project focused on developing AGI systems. Started in Q1 2024 with a $10M budget.",
        "summary": "Autonomous agent research project",
    },
    {
        "name": "AI Research Division",
        "type": "organization",
        "content": "Leading artificial intelligence research division with 50+ researchers. Focus areas include NLP, computer vision, and reinforcement learning.",
        "summary": "AI research organization",
    },
    {
        "name": "Machine Learning Fundamentals",
        "type": "document",
        "content": "Comprehensive 200-page guide covering supervised learning, unsupervised learning, deep learning architectures, and best practices for model deployment.",
        "summary": "ML educational document",
    },
]

relationships = [
    ("Dr. Sarah Chen", "Project Alpha", "LEADS"),
    ("Dr. Sarah Chen", "AI Research Division", "WORKS_FOR"),
    ("Project Alpha", "AI Research Division", "BELONGS_TO"),
    ("Dr. Sarah Chen", "Machine Learning Fundamentals", "AUTHORED"),
]


# Generate embeddings and insert
def get_embedding(text):
    result = genai.embed_content(
        model="models/text-embedding-004", content=text, task_type="retrieval_document"
    )
    return result["embedding"]


with psycopg.connect(conn_str) as conn:
    with conn.cursor() as cur:
        # Insert entities
        for entity in entities:
            embedding_text = f"{entity['name']} {entity['summary']} {entity['content']}"
            embedding = get_embedding(embedding_text)

            cur.execute(
                """
                INSERT INTO nodes (name, type, content, summary, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """,
                (
                    entity["name"],
                    entity["type"],
                    entity["content"],
                    entity["summary"],
                    embedding,
                ),
            )

        # Insert relationships
        for source, target, rel_type in relationships:
            cur.execute(
                """
                INSERT INTO edges (source_id, target_id, relationship_type)
                SELECT s.id, t.id, %s
                FROM nodes s, nodes t
                WHERE s.name = %s AND t.name = %s
                ON CONFLICT DO NOTHING
            """,
                (rel_type, source, target),
            )

        conn.commit()
        print("✓ Sample data loaded successfully!")
