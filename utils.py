# ============================================================
# utils.py
# Shared utility functions for Phase 2 and Phase 3 experiments
# Author  : [Your Name]
# Date    : 2026-05-14
# ============================================================

import os
import re
import time
import sqlite3
import yaml
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================
# Configuration
# ============================================================
DB_PATH   = "./dev/dev_20240627/dev_databases/dev_databases/financial/financial.sqlite"
YAML_PATH = "./financial_semantic_layer.yaml"
MODEL     = "gpt-4o-mini"

client = OpenAI()

FINANCIAL_TABLES = {
    "account" : ["account_id", "district_id", "frequency", "date"],
    "client"  : ["client_id", "gender", "birth_date", "district_id"],
    "disp"    : ["disp_id", "client_id", "account_id", "type"],
    "loan"    : ["loan_id", "account_id", "date", "amount", "duration", "payments", "status"],
    "trans"   : ["trans_id", "account_id", "date", "type", "operation",
                 "amount", "balance", "k_symbol", "bank", "account"],
    "card"    : ["card_id", "disp_id", "type", "issued"],
    "order"   : ["order_id", "account_id", "bank_to", "account_to", "amount", "k_symbol"],
    "district": ["district_id", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
                 "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"],
}

FEW_SHOT_EXAMPLES = """
## Few-shot Examples

Q: How many accounts in Prague are eligible for loans?
Evidence: A3 contains region data
A: SELECT COUNT(account.account_id)
   FROM account
   JOIN district ON account.district_id = district.district_id
   JOIN loan ON account.account_id = loan.account_id
   WHERE district.A3 = 'Prague'
-- NOTE: "eligible for loans" = has a record in the loan table. Do NOT filter by loan.status.

Q: How many male customers have average salary greater than 8000?
Evidence: A11 refers to average salary; Male means gender = 'M'
A: SELECT COUNT(DISTINCT client.client_id)
   FROM client
   JOIN district ON client.district_id = district.district_id
   WHERE client.gender = 'M' AND district.A11 > 8000
-- NOTE: avg_salary concept → A11 in district table.
"""


# ============================================================
# Schema Builders
# ============================================================

def get_ddl_schema() -> str:
    """
    Extract raw DDL from SQLite.
    LLM receives original column names (A2, A3, A11, Czech labels).
    Represents full sensitive schema exposure in Text-to-SQL method.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    parts  = []
    for tbl in tables:
        cols     = pd.read_sql(f"PRAGMA table_info('{tbl}')", conn)
        col_defs = ", ".join(f"{r['name']} {r['type']}" for _, r in cols.iterrows())
        parts.append(f"CREATE TABLE {tbl} ({col_defs});")
    conn.close()
    return "\n".join(parts)


def get_semantic_schema() -> str:
    """
    Build abstracted schema from Semantic Layer YAML.
    LLM receives only cube/dimension/measure names and descriptions.
    Raw column names (A2~A16), Czech codes, and status codes are hidden.
    Implements Information Hiding principle (Parnas, 1972).
    """
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        sl = yaml.safe_load(f)
    lines = ["# Semantic Layer Definition\n"]
    for cube in sl['cubes']:
        lines.append(f"## Cube: {cube['name']}")
        lines.append(f"Description: {cube.get('description','')}\n")
        if cube.get('joins'):
            lines.append("Joins:")
            for j in cube['joins']:
                lines.append(f"  - {cube['name']} -> {j['name']} ({j['relationship']})")
            lines.append("")
        if cube.get('dimensions'):
            lines.append("Dimensions:")
            for d in cube['dimensions']:
                lines.append(f"  - {d['name']} ({d['type']}): {d.get('description','')}")
            lines.append("")
        if cube.get('measures'):
            lines.append("Measures:")
            for m in cube['measures']:
                lines.append(f"  - {m['name']} ({m['type']}): {m.get('description','')}")
            lines.append("")
    return "\n".join(lines)


def build_sl_prompt_schema() -> str:
    """Build explicit concept-to-SQL mapping for Semantic Layer prompt."""
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        sl = yaml.safe_load(f)
    lines = [
        "# Semantic Layer: Concept → Actual SQL Mapping",
        "# Concept names are NOT column names. Use 'Actual SQL expression' in queries.\n"
    ]
    for cube in sl['cubes']:
        tbl = cube['sql_table']
        lines.append(f"## Cube: {cube['name']}  (SQL table: `{tbl}`)")
        lines.append(f"   {cube.get('description','')}")
        if cube.get('joins'):
            lines.append("   Joins:")
            for j in cube['joins']:
                lines.append(
                    f"     {tbl} JOIN {j['name']} ON "
                    f"{j['sql'].replace('{'+cube['name']+'}', tbl).replace('{'+j['name']+'}', j['name'])}"
                )
        lines.append(f"   {'Concept':<30} {'Actual SQL expression'}")
        lines.append("   " + "-" * 65)
        for d in cube.get('dimensions', []):
            lines.append(f"   {d['name']:<30} {d['sql'].strip().replace(chr(10), ' ')}")
        for m in cube.get('measures', []):
            lines.append(f"   {m['name']:<30} {m['type'].upper()}({m['sql'].strip().replace(chr(10), ' ')})")
        lines.append("")
    return "\n".join(lines)


# ============================================================
# LLM Query Functions (Prompt v5)
# ============================================================

def query_text_to_sql(question: str, evidence: str = "") -> dict:
    """
    Text-to-SQL baseline method.
    LLM receives raw DDL schema — full sensitive schema exposure.
    """
    schema = get_ddl_schema()
    prompt = f"""You are an expert SQLite query generator for a financial database.

## Database Schema (DDL)
{schema}

## Rules
- Use ONLY column names that exist in the DDL above.
- Always prefix columns with their table name (e.g., account.account_id, district.A3).
- SQL clause order: SELECT → FROM → JOIN → WHERE → GROUP BY → HAVING → ORDER BY → LIMIT.
- "eligible for loans" means the account has a record in the loan table. Do NOT filter by loan.status.
- Do NOT invent column values.
- Return ONLY the SQL query. No explanation, no markdown, no backticks.
{FEW_SHOT_EXAMPLES}

## Question
{question}

## Additional Evidence
{evidence if evidence else 'None'}
"""
    t0  = time.time()
    res = client.chat.completions.create(
        model=MODEL, max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "method"      : "text_to_sql",
        "question"    : question,
        "sql"         : res.choices[0].message.content.strip(),
        "latency_ms"  : round((time.time() - t0) * 1000),
        "prompt_chars": len(prompt),
    }


def query_semantic_layer(question: str, evidence: str = "") -> dict:
    """
    Semantic Layer method.
    LLM receives abstracted concept definitions — sensitive schema hidden.
    Implements Information Hiding principle (Parnas, 1972).
    """
    sl_schema = build_sl_prompt_schema()
    table_ref = "\n".join(
        f"  {t}: {', '.join(c)}" for t, c in FINANCIAL_TABLES.items()
    )
    prompt = f"""You are an expert SQLite query generator using a Semantic Layer.

## Semantic Layer (Concept → Actual SQL Mapping)
{sl_schema}

## Exact Table and Column Names (use ONLY these)
{table_ref}

## Critical Rules
1. Concept names are NOT column names. Always translate:
   - 'region'          → A3   (district table)
   - 'district_name'   → A2   (district table)
   - 'avg_salary'      → A11  (district table)
   - 'frequency_label' → use CASE frequency WHEN 'POPLATEK MESICNE' THEN 'monthly' ... END
   - 'status_label'    → use CASE status WHEN 'A' THEN 'completed_ok' ... END
2. Always prefix every column with its table name.
3. Exact table names: account, client, disp, loan, trans, card, order, district
   NOTE: use 'order' not 'orders'.
4. SQL clause order: SELECT → FROM → JOIN → WHERE → GROUP BY → HAVING → ORDER BY → LIMIT.
5. "eligible for loans" = has a record in the loan table. Do NOT filter by loan.status.
6. Use standard SQLite functions only (IFNULL not ISNULL, no NUMBER()).
7. Return ONLY valid SQLite SQL. No explanation, no markdown, no backticks.
{FEW_SHOT_EXAMPLES}

## Question
{question}

## Additional Evidence
{evidence if evidence else 'None'}
"""
    t0  = time.time()
    res = client.chat.completions.create(
        model=MODEL, max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "method"      : "semantic_layer",
        "question"    : question,
        "sql"         : res.choices[0].message.content.strip(),
        "latency_ms"  : round((time.time() - t0) * 1000),
        "prompt_chars": len(prompt),
    }


# ============================================================
# Evaluation Functions
# ============================================================

def execute_sql(sql: str) -> tuple:
    """Execute SQL against the financial DB. Returns (DataFrame, error)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)


def normalize_result(df: pd.DataFrame) -> set:
    """
    Normalize query result for value-level exact match comparison.
    Ignores column names and row order. Rounds floats to 4 decimal places.
    """
    if df is None or df.empty:
        return set()
    df = df.copy()
    df.columns = range(len(df.columns))
    df = df.map(lambda x: str(round(float(x), 4)) if isinstance(x, float) else str(x))
    return set(df.apply(tuple, axis=1))


def evaluate(generated_sql: str, gold_sql: str) -> dict:
    """
    Compare generated SQL result against ground truth.
    Metric: Execution Accuracy (EX) — standard BIRD benchmark metric.
    """
    df_gold, err_gold = execute_sql(gold_sql)
    df_gen,  err_gen  = execute_sql(generated_sql)
    if err_gold:
        return {"is_correct": None, "error": f"Gold SQL error: {err_gold}"}
    if err_gen:
        return {"is_correct": False, "error": f"Generated SQL error: {err_gen}",
                "gold_result": df_gold, "generated_result": None}
    return {
        "is_correct"      : (normalize_result(df_gold) == normalize_result(df_gen)),
        "gold_result"     : df_gold,
        "generated_result": df_gen,
        "error"           : None,
    }


# ============================================================
# RQ2 — Sensitive Information Exposure Measurement
# ============================================================

SENSITIVE_COLUMNS = [
    'A2','A3','A4','A5','A6','A7','A8','A9','A10',
    'A11','A12','A13','A14','A15','A16',
]
SENSITIVE_CODES = [
    'POPLATEK MESICNE','POPLATEK TYDNE','POPLATEK PO OBRATU',
    'VYBER KARTOU','VYBER','VKLAD','PREVOD Z UCTU','PRIJEM','VYDAJ',
]

def get_schema_context_sl() -> str:
    """Concept-only schema context sent to LLM in Semantic Layer method."""
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        sl = yaml.safe_load(f)
    lines = ["# Semantic Layer: Cube Definitions\n"]
    for cube in sl['cubes']:
        lines.append(f"## Cube: {cube['name']}")
        lines.append(f"Description: {cube.get('description','')}\n")
        if cube.get('dimensions'):
            lines.append("Dimensions:")
            for d in cube['dimensions']:
                lines.append(f"  - {d['name']} ({d['type']}): {d.get('description','')}")
            lines.append("")
        if cube.get('measures'):
            lines.append("Measures:")
            for m in cube['measures']:
                lines.append(f"  - {m['name']} ({m['type']}): {m.get('description','')}")
            lines.append("")
    return "\n".join(lines)


def measure_exposure(text: str) -> dict:
    """Measure sensitive token exposure in a schema context string."""
    col_counts  = {c: len(re.findall(r'\b' + re.escape(c) + r'\b', text))
                   for c in SENSITIVE_COLUMNS}
    code_counts = {c: len(re.findall(re.escape(c), text, re.IGNORECASE))
                   for c in SENSITIVE_CODES}
    return {
        'col_exposed' : sum(1 for v in col_counts.values()  if v > 0),
        'col_total'   : sum(col_counts.values()),
        'code_exposed': sum(1 for v in code_counts.values() if v > 0),
        'code_total'  : sum(code_counts.values()),
        'col_detail'  : col_counts,
        'code_detail' : code_counts,
    }


if __name__ == "__main__":
    print("utils.py loaded successfully.")
    print(f"  Model     : {MODEL}")
    print(f"  DB path   : {DB_PATH}")
    print(f"  YAML path : {YAML_PATH}")
    print(f"  API Key   : {'OK' if os.environ.get('OPENAI_API_KEY') else 'MISSING'}")