import fitz, chromadb
import os 
from google import genai
from dotenv import load_dotenv 

load_dotenv()

doc = fitz.open("syllabus.pdf")
client = chromadb.PersistentClient(path = "chroma_db")
col = client.get_or_create_collection("syllabus")

pages =  []
for i, page in enumerate(doc):
    text = page.get_text()
    if text.strip():
        pages.append(text)

col.add(documents = pages, 
        ids = [f"page{n+1}"
               for n in range(len(pages))])
print(f"stored {len(pages)} pages")

# res = col.query(query_texts = ["operating systems CS23501 semester V credits"], n_results = 6)
# for doc_text, meta in zip(res["documents"][0], res["ids"][0]):
#     print(meta,"->", doc_text[:150], "\n")

client_ai = genai.Client(api_key = os.environ["GEMINI_KEY"])

while True:
    q = input ("\nask: ")
    if q == "quit":
        break
    res = col.query(query_texts = [q], n_results = 6)
    context ="\n\n".join(f"[{pid}] {doc}" for pid, doc in zip(res["ids"][0], res["documents"][0]))
    prompt = (
        "You answer questions about an Anna University syllabus. "
        "Use only the context below and cite pages ids like [page4].\n\n"
        f"Context:\n{context}\n\nQuestion: {q}"
    )
    answer = client_ai.models.generate_content(
        model = "gemini-3.8-flash",
        contents = prompt 
    )
    print(answer.text)
    
