# ============================================================
# utils.py
# Shared utility functions for Phase 2 and Phase 3 experiments
#
# v1: original Text-to-SQL and Semantic Layer baseline methods
# v2: true information-hiding Semantic Layer architecture, added
#     during revision after exposure recheck found v1's Semantic
#     Layer prompt actually contained all 15 raw sensitive columns
#     (via its concept-to-SQL mapping table, table reference block,
#     and few-shot examples). v2's LLM-facing prompt contains ONLY
#     concept names, types, and descriptions; a local SemanticCompiler
#     (semantic_compiler.py) compiles LLM-generated concept-SQL into
#     real SQLite entirely outside the LLM's context, after the API
#     call returns. Verified via exposure recheck: 0/15 sensitive
#     columns across 6+ different sample questions.
#
# Author  : [Your Name]
# ============================================================

import os
import re
import time
import sqlite3
import threading
import yaml
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================
# Configuration
# ============================================================
# DB_PATH is read from the environment (.env file) rather than
# hardcoded, so this codebase is portable across machines/folders
# without silent failures. Set DB_PATH in your .env file, e.g.:
#   DB_PATH=./dev/dev_20240627/dev_databases/dev_databases/financial/financial.sqlite
# Falls back to the original relative path if not set, but a
# missing/incorrect path is checked explicitly below and raises a
# clear error immediately at import time rather than failing later
# inside individual query calls with an opaque sqlite3 error.
DB_PATH   = os.environ.get(
    "DB_PATH",
    "./dev/dev_20240627/dev_databases/dev_databases/financial/financial.sqlite"
)
YAML_PATH = os.environ.get("YAML_PATH", "./financial_semantic_layer.yaml")
MODEL     = os.environ.get("MODEL", "gpt-4o-mini")

client = OpenAI()

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(
        f"DB_PATH does not resolve to an existing file: '{DB_PATH}' "
        f"(resolved absolute path: '{os.path.abspath(DB_PATH)}'). "
        f"Set DB_PATH in your .env file to the correct location of "
        f"financial.sqlite on this machine, or place the BIRD "
        f"benchmark's 'dev/' folder at this relative path. This check "
        f"runs at import time so the error is caught immediately "
        f"rather than surfacing later as a confusing 'unable to open "
        f"database file' error during query evaluation."
    )

if not os.path.exists(YAML_PATH):
    raise FileNotFoundError(
        f"YAML_PATH does not resolve to an existing file: '{YAML_PATH}'. "
        f"Make sure financial_semantic_layer.yaml is in the working "
        f"directory, or set YAML_PATH in your .env file."
    )

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


# ============================================================
# v1: Text-to-SQL baseline and Semantic Layer (original)
# ============================================================

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


def build_sl_prompt_schema() -> str:
    """
    Build explicit concept-to-SQL mapping for the v1 Semantic Layer
    prompt. Kept for v1 reproducibility. NOTE: this is the function
    that, combined with FINANCIAL_TABLES table_ref and
    FEW_SHOT_EXAMPLES below, was found during exposure recheck to
    expose all 15 raw sensitive columns in v1's actual API payload.
    """
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


def query_text_to_sql(question: str, evidence: str = "") -> dict:
    """Text-to-SQL baseline method. LLM receives raw DDL schema."""
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
    Semantic Layer v1 method. Kept unmodified for v1 result
    reproducibility. See query_semantic_layer_v2() below for the
    corrected true-information-hiding architecture.
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
# v2: true information-hiding Semantic Layer
# ============================================================

from semantic_compiler import SemanticCompiler

# Built once at module load; the compiler object is stateless with
# respect to individual queries, so this reuse is safe.
_v2_compiler = SemanticCompiler(YAML_PATH)

FEW_SHOT_EXAMPLES_V2 = """
## Few-shot Examples (Semantic Layer v2 -- concept names only)

Q: How many accounts in the Prague region are eligible for loans?
Evidence: region is a district-level concept.
A: SELECT COUNT(account.account_id)
   FROM account
   JOIN district ON account.district_link
   JOIN loan ON loan.account_link
   WHERE district.region = 'Prague'
-- NOTE: "eligible for loans" = has a record in the loan table. Do NOT filter by loan.status.
-- NOTE: 'region' is a concept name, not a physical column name. Always use the concept
--       name exactly as listed above -- the compiler resolves it to the real column
--       after you respond; you never need or see the physical name yourself.

Q: How many male clients have a district avg_salary greater than 8000?
Evidence: avg_salary is a district-level concept.
A: SELECT COUNT(DISTINCT client.client_id)
   FROM client
   JOIN district ON client.district_link
   WHERE client.gender = 'M' AND district.avg_salary > 8000
-- NOTE: avg_salary is used directly as a dimension concept name here, not wrapped in
--       an aggregate function, because it filters a single row's value (not a group).

Q: What is the default rate for loans longer than 36 months?
Evidence: default_rate is a pre-defined measure.
A: SELECT loan.default_rate FROM loan WHERE loan.duration > 36
-- NOTE: loan.default_rate is a MEASURE. Use it as-is; do NOT wrap it in
--       SUM(), AVG(), or CAST() yourself -- it already IS a computed rate.
--       Do NOT write your own CASE WHEN status IN ('B','D') logic; that
--       information is not available to you and default_rate already
--       encapsulates it.

Q: How many loan contracts are currently past due?
Evidence: "past due" means the loan is running with an outstanding problem.
A: SELECT COUNT(loan.loan_id) FROM loan WHERE loan.status = 'D'
-- NOTE: This is a SIMPLE row-level count with NO grouping -- use the
--       loan.status DIMENSION directly (a plain equality/IN condition
--       in WHERE), NOT a measure like default_count or running_count.
--       Measures (default_count, default_rate, running_count, etc.)
--       are PRE-AGGREGATED expressions and are ONLY valid in SELECT
--       or HAVING, never in WHERE -- WHERE filters individual rows
--       BEFORE any aggregation happens, so an aggregate measure
--       cannot be evaluated there. If your query has no GROUP BY and
--       you are tempted to write "WHERE loan.default_count > 0" or
--       similar, STOP: rewrite it as a direct status/dimension
--       condition instead, exactly as shown above.

Q: What is the count of distinct clients, by gender, who hold a credit card?
Evidence: cards are linked to clients through an intermediate disposition table.
A: SELECT client.gender, COUNT(DISTINCT client.client_id)
   FROM client
   JOIN disp ON disp.client_link
   JOIN card ON disp.card_link
   GROUP BY client.gender
-- NOTE: client and card are NOT directly joinable -- there is no
--       client-to-card or client-to-loan relationship in the schema.
--       client only joins directly to district. To reach loan, card,
--       or trans from client, you MUST go through disp as an
--       intermediate step: client -> disp -> account -> (loan / card
--       / trans). Skipping disp and joining client directly to loan
--       or card (e.g. "JOIN loan ON client.client_id = loan.account_id")
--       will compile and run without error, but silently produces the
--       WRONG result, because client_id and account_id/card-linked
--       IDs are different identifier spaces that don't correspond
--       row-for-row. Always check the "Can be joined to" list for
--       each cube in the schema above before writing a join -- if two
--       cubes you need aren't directly listed as joinable to each
--       other, find the intermediate cube that connects them instead
--       of joining on same-looking column names as a shortcut.

Q: How many accounts have a primary (owner-type) account holder?
Evidence: primary account holder means disp.type = 'OWNER'.
A: SELECT COUNT(DISTINCT account.account_id)
   FROM account
   JOIN disp ON disp.account_link
   WHERE disp.type = 'OWNER'
-- NOTE: This question needs a ROW-LEVEL filter (disp.type = 'OWNER'),
--       so it uses the DIMENSION disp.type directly -- NOT the
--       disp.owner_count MEASURE, even though "owner_count" sounds
--       like it might answer this question.
-- NOTE: owner_count, gold_count, female_count, and similar measures
--       whose names END IN "_count" are DECEPTIVE: despite the name,
--       they are NOT a per-row flag or boolean you can filter on --
--       they are a MEASURE that is ALREADY a complete, pre-aggregated
--       COUNT with its filter baked in (e.g. owner_count already
--       means "COUNT of rows WHERE type = 'OWNER'", computed over
--       however the query is grouped). Writing
--         SELECT COUNT(*) FROM disp WHERE disp.owner_count = 1
--       or
--         SELECT COUNT(*) FROM disp WHERE disp.owner_count > 0
--       is a category error: owner_count is not a 0/1 value that
--       exists per row, so comparing it in a per-row WHERE clause is
--       invalid, regardless of what value you compare it against.
--       If a question describes a condition that one of these
--       "_count" measures' NAME suggests (e.g. "who are owners" for
--       owner_count, "who hold gold cards" for gold_count, "who are
--       female" for female_count), first check whether the
--       UNDERLYING dimension the measure filters on (disp.type,
--       card.type, client.gender) is available as its own concept --
--       it almost always is -- and use THAT dimension directly in
--       WHERE, exactly as shown above. Reserve the "_count" measures
--       themselves for when the question actually wants a count or
--       rate computed across a GROUP (e.g. "how many owners per
--       district", which needs GROUP BY and the measure in SELECT),
--       not as a substitute for a row-level condition.

Q: What percentage of accounts have a district avg_salary above 9000?
Evidence: percentage = matching accounts / total accounts, times 100.
A: SELECT
     CAST(SUM(CASE WHEN district.avg_salary > 9000 THEN 1 ELSE 0 END) AS REAL)
       / COUNT(account.account_id) * 100
   FROM account
   JOIN district ON account.district_link
-- NOTE: This is the CORRECT pattern when a DIMENSION value (unlike
--       the measures above, avg_salary is a plain dimension with no
--       filter baked in) is needed BOTH as a per-row condition AND
--       to compute an aggregate over the whole set: express the
--       condition inside a CASE WHEN inside the aggregate itself, NOT
--       as a bare WHERE clause filter using the same name you also
--       reference elsewhere in the query. Whenever a question asks
--       for a percentage, rate, or proportion, default to this CASE
--       WHEN ... aggregate pattern rather than a WHERE filter.
"""


# ============================================================
# Evidence sanitizer (added during revision)
#
# The 60-question benchmark's `evidence` field was written to support
# v1 (Text-to-SQL), which needs the LLM to know raw column
# identifiers AND raw internal codes directly (e.g. "Region
# information is in district table column A3." / "Credit card
# withdrawal means operation = 'VYBER KARTOU'."). When passed
# unmodified to build_v2_prompt(), this leaks exactly the raw schema
# identifiers and internal codes v2's architecture exists to hide.
# Confirmed via exhaustive scan across all 60 questions: 18 leak a
# raw column code (A2-A16) or an internal Czech-language operation
# code (VYBER, PRIJEM, etc.) through evidence text alone, independent
# of anything build_v2_prompt() or the compiler itself does.
#
# Two independent fixes are needed, since these are two different
# leak paths through the SAME evidence field:
#   1. Column-reference sentences (e.g. "... column A3.") are pure
#      schema-location trivia that v2's concept names already convey
#      -- REMOVED entirely, no information lost.
#   2. Internal code literals (e.g. 'VYBER KARTOU', 'PRIJEM') carry
#      information the LLM actually needs (which value to filter on)
#      -- these are REWRITTEN to their human-readable label
#      equivalent (e.g. 'card_withdrawal') rather than deleted, so
#      the LLM still knows what to filter on, but only ever sees the
#      label, matching the value_mapped_dimension concepts added to
#      financial_semantic_layer.yaml (trans.type_label,
#      trans.operation_label) that the compiler translates back to
#      the real code AFTER the LLM responds.
#
# Verified via exhaustive test: 0/60 questions leak either a column
# code or an internal operation code after sanitize_evidence() is
# applied, with 0 false-positive strips/rewrites on clean evidence.
# ============================================================

_COLUMN_REFERENCE_PATTERN = re.compile(
    r'[^.]*?\bA(?:1[0-6]|[2-9])\b[^.]*\.',
    re.IGNORECASE
)

# Reverse lookup (real code -> human label) built from the same
# value_mapped_dimensions the compiler uses, so this stays in sync
# with financial_semantic_layer.yaml automatically rather than
# duplicating the mapping as a second hardcoded copy. Sorted by code
# length descending so a longer code (e.g. 'VYBER KARTOU') is matched
# and replaced before a shorter code that is its own prefix (e.g.
# 'VYBER') -- otherwise replacing 'VYBER' first would corrupt
# 'VYBER KARTOU' before it could be matched as a whole.
def _build_code_to_label_map():
    reverse_map = {}
    for vkey, vinfo in _v2_compiler.value_mapped_dimensions.items():
        for label, code in vinfo['value_map'].items():
            reverse_map[code] = label
    return dict(sorted(reverse_map.items(), key=lambda kv: len(kv[0]), reverse=True))

_CODE_TO_LABEL_MAP = _build_code_to_label_map()


def sanitize_evidence(evidence: str) -> str:
    """
    Sanitize evidence text before it reaches the v2 prompt:
      1. Remove sentences referencing a raw schema column identifier
         (A2-A16) entirely.
      2. Rewrite any internal code literal (e.g. 'VYBER KARTOU') to
         its human-readable label equivalent (e.g. 'card_withdrawal'),
         preserving the filtering information the LLM needs without
         ever exposing the real code.
    Safe on evidence with no leak (returned unchanged/only whitespace-
    normalized) or multiple leaked sentences/codes (all handled).
    """
    if not evidence:
        return evidence

    result = _COLUMN_REFERENCE_PATTERN.sub('', evidence)

    for code, label in _CODE_TO_LABEL_MAP.items():
        # Match the code as a quoted literal (single or double quotes,
        # the only forms observed in the benchmark's evidence text) so
        # we don't accidentally rewrite the code if it ever appeared
        # as a substring of an unrelated English word.
        pattern = re.compile(r"(['\"])" + re.escape(code) + r"\1")
        result = pattern.sub(f"'{label}'", result)

    result = re.sub(r'\s+', ' ', result).strip()
    return result


def build_v2_prompt(question: str, evidence: str = "") -> str:
    """
    Build the v2 Semantic Layer prompt. Contains ONLY concept names,
    types, descriptions, abstract join instructions, the excluded-
    concepts note, and few-shot examples using concept names only.
    Deliberately contains NO 'sql:' field and NO literal A2..A16 or
    status-code values anywhere in the SCHEMA-BUILDING portions.

    The `evidence` parameter is run through sanitize_evidence() before
    being embedded, since the benchmark's evidence field was authored
    for v1 and can itself contain raw column references -- see the
    sanitizer's docstring above for the full explanation and the
    exhaustive verification behind this.
    """
    evidence = sanitize_evidence(evidence)

    schema_block = _v2_compiler.build_concept_only_prompt_schema()
    join_block = _v2_compiler.build_join_hint_block()
    excluded_block = _v2_compiler.excluded_concepts_note()

    prompt = f"""You are an expert SQL query generator using a Semantic Layer.
You do NOT have access to the physical database schema. You may ONLY
reference the business concepts listed below. A separate compiler
(which you cannot see) will translate your concept-SQL into real SQL
after you respond.

{schema_block}

{join_block}

{excluded_block}

## Critical Rules
1. Reference every column as `<cube>.<concept_name>` using ONLY the
   concept names listed above. NEVER invent or guess a physical
   column name. Physical column names are not part of your available
   vocabulary and do not appear anywhere in what you can see -- if a
   concept you need isn't listed above, it is not available; do not
   attempt to approximate it with a guessed identifier.
2. Write joins as: JOIN <cube_name> ON <cube_a>.<cube_b>_link
   (do not write your own ON clause with column names).
3. Table/cube names: account, client, disp, loan, trans, card, district
   (note: the 'order' table exists physically but has no defined cube;
   do not reference it).
4. Measures (concepts under "Measures" in each cube above) are already
   full aggregate expressions. Use them directly; do NOT wrap them in
   your own SUM(), COUNT(), AVG(), or CAST(). Do NOT reconstruct the
   business logic they encapsulate (e.g. do not write your own default
   logic -- use loan.default_rate / default_count / running_count).
   Measures are only valid in SELECT or HAVING -- never in WHERE.
5. SQL clause order: SELECT → FROM → JOIN → WHERE → GROUP BY → HAVING
   → ORDER BY → LIMIT.
6. "eligible for loans" = has a record in the loan table. Do NOT filter
   by loan.status.
7. Use standard SQLite functions only (IFNULL not ISNULL, no NUMBER()).
8. Return ONLY your concept-SQL query. No explanation, no markdown, no
   backticks.
{FEW_SHOT_EXAMPLES_V2}

## Question
{question}

## Additional Evidence
{evidence if evidence else 'None'}
"""
    return prompt


def query_semantic_layer_v2(question: str, evidence: str = "", model: str = None) -> dict:
    """
    Semantic Layer v2 method (true information-hiding architecture).

    The LLM sees ONLY concept names + descriptions, never a raw
    column and never a concept-to-column mapping table. It generates
    "concept-SQL" using concept names in place of column names. This
    function compiles that concept-SQL into real, executable SQLite
    LOCALLY, entirely outside the LLM's context -- implementing
    Parnas (1972) Information Hiding for real.

    Returns a dict with BOTH the LLM's raw concept-SQL output and the
    compiled real SQL, plus a compile_error field distinct from any
    later execution error -- giving a genuine two-stage failure
    taxonomy (LLM generation/compilation failure vs. SQL execution
    failure).

    `model` (added for multi-model comparison, Reviewer 2 Concern #2):
    if not specified, uses the module-level MODEL constant
    (gpt-4o-mini by default) exactly as before, so all existing
    single-model notebooks (00-06) are unaffected. Pass an explicit
    model string (e.g. "gpt-4o") to run this same v2 pipeline -- same
    prompt construction, same compiler, same evaluation -- against a
    different model, which is what 07_multi_model_experiment.py does.
    """
    resolved_model = model if model is not None else MODEL
    prompt = build_v2_prompt(question, evidence)
    t0 = time.time()
    res = client.chat.completions.create(
        model=resolved_model, max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    latency_ms = round((time.time() - t0) * 1000)
    concept_sql = res.choices[0].message.content.strip()

    compile_result = _v2_compiler.compile(concept_sql)

    return {
        "method"        : "semantic_layer_v2",
        "model"         : resolved_model,
        "question"      : question,
        "concept_sql"   : concept_sql,
        "sql"           : compile_result['sql'],
        "compile_error" : compile_result['error'],
        "substitutions" : compile_result['substitutions'],
        "latency_ms"    : latency_ms,
        "prompt_chars"  : len(prompt),
    }


def get_schema_context_sl_v2() -> str:
    """
    Returns the EXACT prompt build_v2_prompt() sends (schema portion
    is fixed regardless of question content -- confirmed during
    exposure recheck). Use this as the measurement target for any
    exposure analysis, since it's guaranteed to match the real payload
    (unlike v1's get_schema_context_sl(), which measured a different,
    lighter string than what was actually sent to the API).
    """
    return build_v2_prompt(question="PLACEHOLDER", evidence="")


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


def evaluate(generated_sql: str, gold_sql: str, timeout_seconds: float = 15.0) -> dict:
    """
    Compare generated SQL result against ground truth.
    Metric: Execution Accuracy (EX) — standard BIRD benchmark metric.

    TIMEOUT: SQL execution against this financial DB can be very slow
    for some queries (the trans table alone has ~1.06M rows, and
    certain gold queries involve per-account correlated subqueries
    over the full transaction history -- e.g. CR23's gold SQL took
    over 200 seconds during initial benchmark validation). Without a
    timeout, a single slow query can silently stall an entire batch
    run for minutes with no indication of whether it's still working
    or has hung. This wraps the actual execution in a background
    thread with a hard time limit, matching the pattern
    credit_rating_benchmark_experiment.ipynb used for v1
    (evaluate_with_timeout there); built directly into evaluate() here
    so every notebook that imports utils.py gets this protection
    automatically, without needing to redefine a wrapper each time.
    """
    if generated_sql is None:
        return {"is_correct": False, "error": "No SQL to evaluate (generated_sql is None)"}

    result_holder = [None]

    def _run():
        df_gold, err_gold = execute_sql(gold_sql)
        df_gen,  err_gen  = execute_sql(generated_sql)
        if err_gold:
            result_holder[0] = {"is_correct": None, "error": f"Gold SQL error: {err_gold}"}
            return
        if err_gen:
            result_holder[0] = {"is_correct": False, "error": f"Generated SQL error: {err_gen}",
                                 "gold_result": df_gold, "generated_result": None}
            return
        result_holder[0] = {
            "is_correct"      : (normalize_result(df_gold) == normalize_result(df_gen)),
            "gold_result"     : df_gold,
            "generated_result": df_gen,
            "error"           : None,
        }

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        # Thread is still running after the timeout. We can't forcibly
        # kill it (Python threads can't be terminated externally), so
        # it will keep running in the background and eventually finish
        # or the DB connection will time out on its own -- but we
        # return immediately so the calling batch loop isn't blocked.
        # The daemon=True flag ensures this thread won't prevent the
        # Python process itself from exiting later.
        return {
            "is_correct": False,
            "error": f"SQL evaluation timeout after {timeout_seconds}s (query still running in background)",
        }

    return result_holder[0]


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
    """
    v1's concept-only schema context. NOTE: this builds a LIGHTER
    string than what query_semantic_layer() (v1) actually sends to
    the API (which also includes build_sl_prompt_schema()'s mapping
    table, FINANCIAL_TABLES' table_ref, and FEW_SHOT_EXAMPLES). Kept
    for v1 reproducibility; use get_schema_context_sl_v2() for a
    measurement target that matches its actual prompt exactly.
    """
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
    print(f"  DB path   : {DB_PATH} (exists: {os.path.exists(DB_PATH)})")
    print(f"  YAML path : {YAML_PATH} (exists: {os.path.exists(YAML_PATH)})")
    print(f"  API Key   : {'OK' if os.environ.get('OPENAI_API_KEY') else 'MISSING'}")
    print(f"  v2 architecture: query_semantic_layer_v2 available")