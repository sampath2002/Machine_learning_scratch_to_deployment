from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import string, random

app = FastAPI()
db = {}

class URLItem(BaseModel):
    url: str

def generate_short_id(num=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=num))

@app.post("/shorten")
def shorten_url(item: URLItem):
    short_id = generate_short_id()
    db[short_id] = item.url
    return {"short_url": f"http://localhost:8000/{short_id}"}

@app.get("/{short_id}")
def redirect_url(short_id: str):
    url = db.get(short_id)
    if url: 
        return {"url": url}
    raise HTTPException(status_code=404, detail="URL not found")