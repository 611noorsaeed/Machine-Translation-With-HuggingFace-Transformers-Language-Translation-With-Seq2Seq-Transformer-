# pip install -U transformers torch fastapi uvicorn jinja2
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Initialize FastAPI app
app = FastAPI()

# Load the saved model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("my_trans_model")
tokenizer = AutoTokenizer.from_pretrained("my_trans_model")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Pydantic model for input validation
class TranslationRequest(BaseModel):
    text: str

# Translation function
def translate_text(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Route for index page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API route for translation
@app.post("/translate")
async def translate(request: TranslationRequest):
    translation = translate_text(request.text)
    return {"translation": translation}
