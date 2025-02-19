import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel
from ingest import Ingest
from chat import Chat

app = FastAPI()

chat = Chat()
ingest = Ingest()

@app.get("/ingest")
def ingest_file(filename: str):
    filepath = os.path.join('./data', filename)
    ingest.ingest_doc(filepath)
    return {
        "success": True,
        "message": "Ingest and save file successfully"
    }

@app.get("/remove")
def remove_file(filename: str):
    filepath = os.path.join('./data', filename)
    ingest.remove_doc(filepath)
    return {
        "success": True,
        "message": "Remove file successfully"
    }

@app.get("/chat")
def invoke_chat(query: str):
    result = chat.ask(query)
    return {
        "success": True,
        "message": result
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
