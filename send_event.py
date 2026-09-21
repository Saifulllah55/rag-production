from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["Hello world"],
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=3072,
    ),
)

print(len(response.embeddings))
print(len(response.embeddings[0].values))