"""
Gradio demo for the fine-tuned Text-to-SQL model.
Deploy this on Render.com (free tier, Docker) -- see the accompanying
Dockerfile and requirements.txt.

Runs on CPU (no GPU on Render's free tier, and bitsandbytes 4-bit quantization
is GPU-only), so the adapter is merged into the base model once at startup and
run in plain float32 -- a bit slower per query than the T4 we trained on, but
fine for a demo where someone tries a handful of questions.
"""

import os
import re
import random
import sqlite3
import pandas as pd
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_ID = "Harsh3567475586/qwen2.5-1.5b-text-to-sql-lora"

SYSTEM_PROMPT = (
    "You are a precise text-to-SQL assistant. Given a database schema and a "
    "natural language question, output ONLY the SQL query that answers it. "
    "No explanation, no markdown, just the query."
)

print("Loading model (first load takes ~30-60s)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model = model.merge_and_unload()
model.eval()
print("Model ready.")

def clean_sql(text):
    text = re.sub(r"```sql|```", "", text, flags=re.IGNORECASE).strip()
    if ";" in text:
        text = text.split(";")[0]
    elif text.splitlines():
        text = text.splitlines()[0]
    return text.strip()

def generate_sql(schema, question, max_new_tokens=128):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return clean_sql(tokenizer.decode(gen, skip_special_tokens=True))

DESTRUCTIVE = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|ATTACH|PRAGMA|REPLACE)\b", re.IGNORECASE
)

def populate_synthetic_data(conn, create_sql, num_rows=10):
    cur = conn.cursor()
    tables = []
    for stmt in [s.strip() for s in create_sql.split(";") if s.strip()]:
        cur.execute(stmt)
        m = re.search(r"CREATE TABLE\s+(\w+)", stmt, re.IGNORECASE)
        if m:
            tables.append(m.group(1))
    text_pool = ["alpha", "beta", "gamma", "delta", "epsilon"]
    for table in tables:
        cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
        for _ in range(num_rows):
            row = []
            for col in cols:
                t = (col[2] or "").upper()
                if "INT" in t:
                    row.append(random.randint(1, 100))
                elif any(k in t for k in ("REAL", "FLOA", "DOUB", "DEC")):
                    row.append(round(random.uniform(1, 1000), 2))
                else:
                    row.append(random.choice(text_pool) + str(random.randint(1, 20)))
            cur.execute(f"INSERT INTO {table} VALUES ({','.join(['?'] * len(row))})", row)
    conn.commit()

def run_demo(schema, question):
    if not schema.strip() or not question.strip():
        return "", pd.DataFrame({"message": ["Enter a schema and a question above."]})

    sql = generate_sql(schema, question)

    if DESTRUCTIVE.search(sql):
        return sql, pd.DataFrame({"message": ["Blocked -- looks destructive, not executed on the sample data."]})

    conn = sqlite3.connect(":memory:")
    try:
        populate_synthetic_data(conn, schema)
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else ["result"]
        rows = cur.fetchall()
        conn.close()
        return sql, pd.DataFrame(rows, columns=cols)
    except Exception as e:
        conn.close()
        return sql, pd.DataFrame({"error": [str(e)]})

EXAMPLE_SCHEMA = "CREATE TABLE employees (name VARCHAR, department VARCHAR, salary INTEGER)"
EXAMPLE_QUESTION = "What is the average salary in the Engineering department?"

demo = gr.Interface(
    fn=run_demo,
    inputs=[
        gr.Textbox(label="Table schema (CREATE TABLE ...)", lines=3, value=EXAMPLE_SCHEMA),
        gr.Textbox(label="Question in plain English", value=EXAMPLE_QUESTION),
    ],
    outputs=[
        gr.Code(label="Generated SQL", language="sql"),
        gr.Dataframe(label="Result (on random sample data generated to match your schema)"),
    ],
    title="Text-to-SQL — Qwen2.5-1.5B, QLoRA fine-tuned",
    description=(
        "Fine-tuned on 9K schema + question + SQL examples "
        "(85.4% -> 90.9% execution accuracy vs. the zero-shot base model). "
        "There's no real database behind this demo, so sample rows matching "
        "your schema are generated on the fly so the query actually runs."
    ),
    examples=[
        [EXAMPLE_SCHEMA, EXAMPLE_QUESTION],
        ["CREATE TABLE orders (customer VARCHAR, amount REAL, status VARCHAR)",
         "How many orders have a status of 'delivered'?"],
    ],
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
