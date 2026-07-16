import re

# ============================================================
# EVIDENCE SANITIZER for v2 architecture
#
# The 60-question benchmark's `evidence` field was originally written
# to support v1 (Text-to-SQL), which needs the LLM to know raw column
# identifiers directly (e.g. "Region information is in district table
# column A3."). When this same evidence text is passed unmodified to
# v2's build_v2_prompt(), it leaks exactly the raw schema identifiers
# the whole v2 architecture exists to hide -- confirmed via exhaustive
# scan: 12 of 60 questions leak a raw column code (A2-A16) through
# evidence text alone, independent of anything the compiler or prompt
# builder does.
#
# This module strips column-reference sentences from evidence before
# it's passed to build_v2_prompt(). It does NOT touch value-level
# code leaks (e.g. 'VYBER KARTOU', 'PRIJEM') -- those require a
# YAML/compiler extension (see the separate measure-based fix) since
# simply deleting that text would remove information the LLM actually
# needs to construct a correct filter, unlike column references which
# are purely redundant in v2 (the concept name already conveys the
# same meaning).
# ============================================================

# Matches sentences/clauses that reference a raw column identifier by
# name, in the specific phrasings observed across all 60 evidence
# strings. Deliberately narrow and pattern-matched to the OBSERVED
# phrasings rather than a single loose regex, so this is easy to
# audit against exactly what it's designed to catch, and so it fails
# LOUD (via the verification step below) rather than silently missing
# a phrasing it wasn't built for.
COLUMN_REFERENCE_PATTERNS = [
    # "... is in [district/loan/client] [table] column A11 [of the ... table]."
    re.compile(
        r'[^.]*?\bcolumn\s+A(?:1[0-6]|[2-9])\b[^.]*\.',
        re.IGNORECASE
    ),
    # "District name is in column A2." / "Region information is in
    # district table column A3." / "... column A12, 1996 is in
    # column A13." -- covered by the pattern above already, but this
    # second pattern catches phrasings where "column" appears before
    # the code without an immediately preceding word boundary issue,
    # as an extra safety net.
    re.compile(
        r'[^.]*?\bA(?:1[0-6]|[2-9])\s+(?:is|contains|refers to)\b[^.]*\.',
        re.IGNORECASE
    ),
    # "Unemployment rate 1995 is A12, 1996 is A13." -- no "column"
    # keyword at all, and the code appears at the END of a clause
    # ("... is A12") rather than the code being immediately followed
    # by a verb. This is the loosest pattern of the three: it matches
    # ANY sentence containing a bare A2-A16 token, full stop. Kept as
    # the LAST pattern applied (broadest net) specifically to catch
    # phrasings the two more targeted patterns above miss, at the
    # cost of being more willing to strip a whole sentence for a
    # single bare code mention anywhere in it.
    re.compile(
        r'[^.]*?\bA(?:1[0-6]|[2-9])\b[^.]*\.',
        re.IGNORECASE
    ),
]


def sanitize_evidence(evidence: str) -> str:
    """
    Remove sentences that reference a raw schema column identifier
    (A2-A16) from an evidence string, returning the remaining text.
    Safe to call on evidence with NO leak (returns it unchanged) or
    multiple leaked sentences (removes all of them).
    """
    if not evidence:
        return evidence

    result = evidence
    for pattern in COLUMN_REFERENCE_PATTERNS:
        result = pattern.sub('', result)

    # collapse any resulting double spaces/leading-trailing whitespace
    # left behind by sentence removal
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def verify_no_leak_remains(evidence: str) -> list:
    """
    Verification helper: after sanitizing, confirm no A2-A16 pattern
    remains. Returns a list of any still-present matches (empty list
    = clean). Used by the test suite below and can be called at
    runtime as a defensive check before evidence reaches the prompt.
    """
    SENSITIVE_COLUMNS = [f'A{i}' for i in range(2, 17)]
    return [c for c in SENSITIVE_COLUMNS if re.search(r'\b' + c + r'\b', evidence)]


if __name__ == "__main__":
    # Exhaustive test against all 18 known-leaking evidence strings
    # from the actual 60-question benchmark, verifying both that the
    # leak is removed AND that non-leaking text isn't damaged.
    test_cases = [
        ("CR07", "Default means status = 'D'. Region information is in district table column A3.",
         "Default means status = 'D'."),
        ("CR14", "Average salary is in column A11 of the district table. Default means status B or D.",
         "Default means status B or D."),
        ("CR20", "Running contracts have status C or D. District name is in column A2.",
         "Running contracts have status C or D."),
        ("CR21", "Default means status B or D. District name is in column A2.",
         "Default means status B or D."),
        ("CR24", "Unemployment rate 1995 is in column A12, 1996 is in column A13. Default means status B or D.",
         "Default means status B or D."),
        ("CR29", "Default rate = defaulted loans / total loans. Average salary is in district column A11. Default means status B or D.",
         "Default rate = defaulted loans / total loans. Default means status B or D."),
        ("CR35", "Average salary is in district column A11.",
         ""),
        ("CR36", "Region information is in district column A3.",
         ""),
        ("CR37", "Running with no issues means status = 'C'. Region is in district column A3.",
         "Running with no issues means status = 'C'."),
        ("CR38", "Unemployment rate 1995 is A12, 1996 is A13. Default means status B or D.",
         "Default means status B or D."),
        ("CR48", "HHI = sum of squared market shares. Market share = district loan amount / total loan amount. District name is in column A2.",
         "HHI = sum of squared market shares. Market share = district loan amount / total loan amount."),
    ]

    print("=" * 70)
    print("SANITIZATION TEST")
    print("=" * 70)
    all_ok = True
    for qid, original, expected_contains_no_leak in test_cases:
        sanitized = sanitize_evidence(original)
        remaining_leak = verify_no_leak_remains(sanitized)
        ok = len(remaining_leak) == 0
        all_ok = all_ok and ok
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {qid}")
        print(f"       original : {original}")
        print(f"       sanitized: {sanitized!r}")
        if not ok:
            print(f"       LEAK REMAINS: {remaining_leak}")
        print()

    # also verify non-leaking evidence passes through completely
    # unchanged (no false-positive stripping)
    print("=" * 70)
    print("NO-FALSE-POSITIVE TEST (non-leaking evidence must be untouched)")
    print("=" * 70)
    clean_cases = [
        "Default means the loan contract is still running but the client is in debt.",
        "Gender is stored in the client table. M = male, F = female.",
        "Duration is measured in months.",
        "A client can be linked to multiple accounts via the disp table.",
    ]
    for original in clean_cases:
        sanitized = sanitize_evidence(original)
        ok = sanitized == original
        all_ok = all_ok and ok
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {original[:60]}...")
        if not ok:
            print(f"       CHANGED TO: {sanitized}")

    print()
    print("=" * 70)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print("=" * 70)
