import re
import yaml


class SemanticCompiler:
    def __init__(self, yaml_path: str):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            self.sl = yaml.safe_load(f)

        self.EXCLUDED_CONCEPTS = {
            'account.frequency_label': None,
            'client.gender_label': None,
            'loan.status_label': (
                "status_label is not available. For default-related "
                "questions, use the measures loan.default_rate, "
                "loan.default_count, or loan.running_count instead."
            ),
            'loan.is_default': (
                "is_default is not available as a dimension. For "
                "default-related questions, use the measures "
                "loan.default_rate, loan.default_count, or "
                "loan.running_count instead."
            ),
        }

        self._build_lookup_tables()

    def _build_lookup_tables(self):
        self.cubes = {}
        self.dimensions = {}
        self.measures = {}
        self.joins = {}
        self.cube_descriptions = {}
        self.dimension_meta = {}
        self.measure_meta = {}
        # NEW: dimensions with a value_map (label -> real code
        # translation), e.g. trans.type_label mapping 'deposit' ->
        # 'PRIJEM'. Kept separate from self.dimensions because these
        # need BOTH a column-name substitution (like any dimension)
        # AND a value-level substitution wherever the concept is
        # compared against one of its label values.
        self.value_mapped_dimensions = {}

        for cube in self.sl['cubes']:
            cname = cube['name']
            self.cubes[cname] = cube['sql_table']
            self.cube_descriptions[cname] = cube.get('description', '')
            self.joins[cname] = cube.get('joins', [])

            for d in cube.get('dimensions', []):
                key = f"{cname}.{d['name']}"
                sql_val = d['sql'].strip() if isinstance(d['sql'], str) else str(d['sql'])
                is_bare_column = (
                    sql_val.replace('_', '').isalnum()
                    and ' ' not in sql_val and '(' not in sql_val
                )
                self.dimension_meta[key] = {
                    'type': d.get('type'),
                    'description': d.get('description', ''),
                    'primary_key': d.get('primary_key', False),
                }
                if is_bare_column and key not in self.EXCLUDED_CONCEPTS:
                    self.dimensions[key] = sql_val
                if 'value_map' in d and key not in self.EXCLUDED_CONCEPTS:
                    self.value_mapped_dimensions[key] = {
                        'raw_column': sql_val,
                        'value_map': d['value_map'],
                    }

            for m in cube.get('measures', []):
                key = f"{cname}.{m['name']}"
                filters = [f['sql'] for f in m.get('filters', [])]
                self.measures[key] = {
                    'sql': m['sql'].strip() if isinstance(m['sql'], str) else str(m['sql']),
                    'type': m['type'],
                    'filters': filters,
                }
                self.measure_meta[key] = {
                    'type': m['type'],
                    'description': m.get('description', ''),
                }

    def build_concept_only_prompt_schema(self) -> str:
        lines = ["# Semantic Layer: Available Concepts\n"]
        lines.append(
            "Below are business concepts you may use in your query. "
            "Concept names are NOT physical column names -- do not "
            "guess or invent a physical column name. Reference each "
            "concept as `<cube>.<concept_name>`.\n"
        )
        for cube in self.sl['cubes']:
            cname = cube['name']
            lines.append(f"## {cname}")
            lines.append(f"{self.cube_descriptions[cname]}\n")

            if self.joins[cname]:
                lines.append("Can be joined to:")
                for j in self.joins[cname]:
                    lines.append(f"  - {j['name']} ({j['relationship']})")
                lines.append("")

            dim_keys = [k for k in self.dimensions if k.startswith(f"{cname}.")]
            if dim_keys:
                lines.append("Dimensions:")
                for k in dim_keys:
                    concept = k.split('.', 1)[1]
                    meta = self.dimension_meta[k]
                    pk_note = " [primary key]" if meta['primary_key'] else ""
                    lines.append(f"  - {concept} ({meta['type']}){pk_note}: {meta['description']}")
                    if k in self.value_mapped_dimensions:
                        valid_labels = sorted(self.value_mapped_dimensions[k]['value_map'].keys())
                        lines.append(f"      Valid values for filtering: {', '.join(repr(v) for v in valid_labels)}")
                lines.append("")

            measure_keys = [k for k in self.measures if k.startswith(f"{cname}.")]
            if measure_keys:
                lines.append("Measures (pre-defined aggregates -- use as-is, do not wrap in your own aggregate function):")
                for k in measure_keys:
                    concept = k.split('.', 1)[1]
                    meta = self.measure_meta[k]
                    lines.append(f"  - {concept} ({meta['type']}): {meta['description']}")
                lines.append("")

        return "\n".join(lines)

    def build_join_hint_block(self) -> str:
        lines = ["# Join Instructions",
                 "Write joins as: JOIN <cube_name> ON <cube_a>.<cube_b>_link",
                 "The compiler will resolve the actual join condition. Example:",
                 "  FROM account JOIN district ON account.district_link",
                 "  (do NOT write 'ON account.district_id = district.district_id' yourself)\n"]
        return "\n".join(lines)

    def excluded_concepts_note(self) -> str:
        lines = ["# Not Available",
                 "The following are NOT valid concepts. Do not use them:"]
        for concept, hint in self.EXCLUDED_CONCEPTS.items():
            if hint:
                lines.append(f"  - {concept}: {hint}")
            else:
                cube, name = concept.split('.', 1)
                lines.append(f"  - {concept} (not available; use {cube}.gender instead)" if 'gender' in name
                              else f"  - {concept} (not available)")
        return "\n".join(lines)

    def _prefix_raw_columns(self, expr: str, cube_name: str, table: str) -> str:
        """
        BUG FIX (found in pilot run): originally only collected raw
        columns from the cube's DIMENSIONS, so a bare column that
        exists ONLY inside a measure's own base_expr (e.g. trans.amount
        and trans.balance, which have no corresponding dimension in
        this YAML -- only avg_amount/total_amount/avg_balance measures
        reference them) was never added to raw_cols_for_cube, and so
        never got its table prefix applied. This surfaced as
        "AVG(amount)" instead of "AVG(trans.amount)" in compiled SQL,
        which works by accident in a single-table query but raises
        "ambiguous column name" once a JOIN brings in another table
        that also has a column named "amount" (e.g. the loan table).
        Fixed by ALSO scanning every measure belonging to this cube
        (base_expr and any filters) for bare single-token raw column
        references and adding those to the candidate set too.
        """
        raw_cols_for_cube = set()
        for key, raw_col in self.dimensions.items():
            if key.startswith(f"{cube_name}."):
                raw_cols_for_cube.add(raw_col)

        for cube in self.sl['cubes']:
            if cube['name'] != cube_name:
                continue
            for d in cube.get('dimensions', []):
                sql_val = d['sql'].strip() if isinstance(d['sql'], str) else str(d['sql'])
                is_bare_column = (
                    sql_val.replace('_', '').isalnum()
                    and ' ' not in sql_val and '(' not in sql_val
                )
                if is_bare_column:
                    raw_cols_for_cube.add(sql_val)
            # NEW: also scan this cube's own measures for bare raw
            # column tokens that aren't already covered by a dimension
            # (e.g. 'amount', 'balance', 'payments' used directly as a
            # measure's base_expr with no dimension wrapping them).
            for m in cube.get('measures', []):
                candidates = []
                m_sql = m['sql'].strip() if isinstance(m['sql'], str) else str(m['sql'])
                if m_sql.replace('_', '').isalnum() and ' ' not in m_sql and '(' not in m_sql:
                    candidates.append(m_sql)
                for filt in m.get('filters', []):
                    # filters are expressions like "status IN ('B','D')" or
                    # "gender = 'F'" -- extract bare identifier tokens
                    # (skip SQL keywords, string literals, numbers).
                    #
                    # BUG FIX: the original version ran the identifier
                    # regex over the RAW filter string, which also
                    # matched letters INSIDE single-quoted string
                    # literals (e.g. the 'B' and 'D' in "status IN
                    # ('B','D')", or the 'F' in "gender = 'F'"). Those
                    # got added to raw_cols_for_cube and were then
                    # incorrectly prefixed wherever they appeared as a
                    # bare token elsewhere -- e.g. turning the literal
                    # 'F' into 'client.F' inside an unrelated WHERE
                    # clause. Fixed by stripping all single-quoted
                    # string literal contents out of the filter text
                    # BEFORE extracting identifier tokens, so values
                    # inside quotes are never treated as column names.
                    filt_sql = filt['sql']
                    filt_sql_no_literals = re.sub(r"'[^']*'", "''", filt_sql)
                    tokens = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', filt_sql_no_literals)
                    sql_keywords = {'IN', 'AND', 'OR', 'NOT', 'NULL', 'IS', 'LIKE'}
                    candidates.extend(t for t in tokens if t.upper() not in sql_keywords)
                raw_cols_for_cube.update(candidates)

        result = expr
        for raw_col in sorted(raw_cols_for_cube, key=len, reverse=True):
            pattern = r'\b' + re.escape(raw_col) + r'\b'
            result = re.sub(pattern, f"{table}.{raw_col}", result)
        return result

    def compile(self, concept_sql: str) -> dict:
        sql = concept_sql
        substitutions = []

        for concept, hint in self.EXCLUDED_CONCEPTS.items():
            pattern = r'\b' + re.escape(concept) + r'\b'
            if re.search(pattern, sql):
                msg = hint if hint else f"{concept} is not an available concept."
                return {'sql': None, 'error': f"Excluded concept used: {concept}. {msg}",
                        'substitutions': substitutions}

        # BUG FIX (round 2, found in second pilot run): the LLM does
        # not consistently write the join hint in the "ON
        # cube_a.cube_b_link" order the few-shot examples demonstrate.
        # It sometimes reverses this to "ON <real-looking-column> =
        # cube_a.cube_b_link" -- e.g.
        #   "JOIN disp ON client.client_id = disp.client_link"
        # instead of
        #   "JOIN disp ON client.client_link"
        # The round-1 fix only added an OPTIONAL trailing "= x.y" after
        # the link token, which handles link-then-column but not
        # column-then-link. Rather than continuing to special-case
        # every ordering the LLM might invent, this version matches
        # the ENTIRE "ON ... " clause up to the next JOIN/WHERE/GROUP
        # BY/HAVING/ORDER BY/semicolon/end-of-string, then searches
        # WITHIN that captured clause for a "cube_x.cube_y_link" token
        # wherever it appears (start, middle, or end) and discards
        # everything else in the clause -- whatever the LLM wrote
        # around the link token is never authoritative; only the link
        # token itself determines the real join condition, which
        # resolve_join() looks up from the YAML's own join definitions.
        join_pattern = re.compile(
            r'JOIN\s+(\w+)\s+ON\s+(.*?)'
            r'(?=\bJOIN\b|\bWHERE\b|\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|;|$)',
            re.IGNORECASE | re.DOTALL
        )
        link_token_pattern = re.compile(r'(\w+)\.(\w+)_link', re.IGNORECASE)

        def resolve_join(m):
            join_cube_b, on_clause_text = m.group(1), m.group(2)
            link_match = link_token_pattern.search(on_clause_text)
            if not link_match:
                substitutions.append(
                    f"JOIN link UNRESOLVED for '{join_cube_b}': no _link token found in "
                    f"'{on_clause_text.strip()}' -- left unchanged"
                )
                return m.group(0)

            # BUG FIX (round 2): a link token "X.Y_link" means "X has a
            # defined join to Y" -- X is the OWNER of the link, Y is
            # its TARGET. The JOIN's actual target (join_cube_b, from
            # "JOIN <join_cube_b> ON ...") can match EITHER the link
            # owner (X) or the link target (Y), depending on which
            # cube the LLM happened to write the "_link" suffix on:
            #   "JOIN account ON disp.account_link"
            #     -> link owner=disp, target=account; join_cube_b=account
            #        matches the link's TARGET (Y) -- disp is joining
            #        TO account, exactly as its own join list says.
            #   "JOIN disp ON client.client_id = disp.client_link"
            #     -> link owner=disp, target=client; join_cube_b=disp
            #        matches the link's OWNER (X) -- so paradoxically
            #        the same relationship ("disp joins to client") is
            #        being expressed, just with the _link suffix
            #        attached to the cube being brought in (disp)
            #        rather than the cube already in scope (client).
            # Either way, the SAME underlying relationship is "disp
            # joins to client" (disp.joins list has an entry named
            # 'client'). So: whichever of (link_owner, link_target)
            # equals join_cube_b, look up the OTHER one in ITS OWN
            # joins list under the name equal to join_cube_b -- i.e.
            # always search self.joins[the_one_that_is_NOT_join_cube_b]
            # for an entry named join_cube_b, since that owner cube is
            # the one whose YAML join definition actually contains the
            # real ON clause for this relationship.
            link_owner, link_target = link_match.group(1), link_match.group(2)

            if link_target == join_cube_b:
                owner_cube = link_owner  # e.g. disp.account_link, JOIN account
            elif link_owner == join_cube_b:
                owner_cube = link_target  # e.g. disp.client_link, JOIN disp -> owner is 'client' side? see below
            else:
                # The link token references a REAL relationship in the
                # YAML (e.g. disp.account_link IS a valid disp->account
                # join), but neither side of it matches join_cube_b --
                # i.e. the LLM is trying to JOIN a DIFFERENT cube
                # (join_cube_b) using a link token that actually
                # describes a relationship between two OTHER cubes.
                # This is a distinct mistake from CR34-style cases
                # (where the link token itself corresponds to no
                # relationship anywhere in the YAML) -- here the
                # confusion is about WHICH link belongs on THIS JOIN
                # line, likely because the LLM is trying to compress a
                # multi-hop path (e.g. client->disp->account->loan)
                # into fewer JOIN lines than the hops require. Give a
                # more specific, actionable hint: point out that the
                # link token used belongs to a DIFFERENT relationship,
                # and that reaching join_cube_b requires its own
                # dedicated JOIN line with a link token that names
                # join_cube_b specifically.
                is_real_relationship_elsewhere = bool(
                    self.joins.get(link_owner) and
                    any(j['name'] == link_target for j in self.joins.get(link_owner, []))
                ) or bool(
                    self.joins.get(link_target) and
                    any(j['name'] == link_owner for j in self.joins.get(link_target, []))
                )
                if is_real_relationship_elsewhere:
                    substitutions.append(
                        f"JOIN link MISMATCHED for '{join_cube_b}': link token "
                        f"'{link_owner}.{link_target}_link' is a valid relationship, "
                        f"but it connects '{link_owner}' and '{link_target}' -- "
                        f"neither of which is '{join_cube_b}'. This JOIN line's "
                        f"link token must name '{join_cube_b}' itself. Left unchanged."
                    )
                else:
                    substitutions.append(
                        f"JOIN link UNRESOLVED for '{join_cube_b}': link token "
                        f"'{link_owner}.{link_target}_link' does not reference "
                        f"'{join_cube_b}' on either side -- left unchanged"
                    )
                return m.group(0)

            # owner_cube is the cube whose YAML join list we search for
            # an entry pointing at join_cube_b OR at the other link name
            # -- try both plausible lookups since the relationship is
            # symmetric in meaning even though only one side's YAML
            # entry actually holds the SQL.
            for lookup_cube, target_name in [(link_owner, link_target), (link_target, link_owner)]:
                for j in self.joins.get(lookup_cube, []):
                    if j['name'] == target_name and (lookup_cube == join_cube_b or target_name == join_cube_b):
                        real_on = (
                            j['sql']
                            .replace('{' + lookup_cube + '}', self.cubes[lookup_cube])
                            .replace('{' + target_name + '}', self.cubes[target_name])
                        )
                        substitutions.append(
                            f"JOIN link {lookup_cube}.{target_name} -> ON {real_on}"
                        )
                        return f"JOIN {self.cubes[join_cube_b]} ON {real_on} "

            substitutions.append(
                f"JOIN link UNRESOLVED: '{link_owner}.{link_target}_link' found but no "
                f"matching join definition connects to '{join_cube_b}' in the YAML"
            )
            return m.group(0)

        sql = join_pattern.sub(resolve_join, sql)

        # --- Step 2b: fail fast on any unresolved join ---
        # BUG FIX: previously, a join hint the compiler couldn't
        # resolve (e.g. "district.district_link", which references a
        # relationship that doesn't exist in the YAML -- district has
        # no joins: entry at all, only account -> district does) was
        # left completely unchanged in the SQL string (see the
        # "UNRESOLVED" logging in resolve_join above), and execution
        # then failed with an opaque "no such column: district.
        # district_link" error. That's a genuine LLM mistake (writing
        # a join hint that doesn't correspond to any real relationship
        # in the schema), but it's much more useful to catch it HERE,
        # at compile time, with a message that explains WHY the link
        # didn't resolve, than to let it surface only as a cryptic
        # SQLite error after execution. A leftover "\w+_link" token
        # anywhere in the compiled SQL means resolve_join() gave up on
        # at least one JOIN clause.
        leftover_link_match = re.search(r'\b(\w+)\.(\w+)_link\b', sql, re.IGNORECASE)
        if leftover_link_match:
            # Pull the specific diagnosis already logged in
            # substitutions (MISMATCHED vs UNRESOLVED) so the returned
            # error reflects the ACTUAL failure mode rather than one
            # generic message for both cases.
            specific_diagnosis = next(
                (s for s in substitutions if 'JOIN link MISMATCHED' in s or 'JOIN link UNRESOLVED' in s),
                None
            )
            if specific_diagnosis and 'MISMATCHED' in specific_diagnosis:
                error_msg = (
                    f"Invalid SQL: {specific_diagnosis} "
                    f"You likely need an ADDITIONAL, SEPARATE JOIN line for "
                    f"'{leftover_link_match.group(1)}' or "
                    f"'{leftover_link_match.group(2)}' -- whichever cube isn't "
                    f"yet in your query -- using a link token that names your "
                    f"actual JOIN target directly, rather than reusing a link "
                    f"token from a different relationship."
                )
            else:
                error_msg = (
                    f"Invalid SQL: the join hint "
                    f"'{leftover_link_match.group(0)}' does not correspond to "
                    f"any defined relationship between the cubes involved. "
                    f"Check that you're joining cubes that actually have a "
                    f"relationship defined (see 'Can be joined to' for each "
                    f"cube in the schema above), and that the join hint is "
                    f"written as JOIN <target_cube> ON <cube_already_in_your_"
                    f"query>.<target_cube>_link -- the cube name immediately "
                    f"before '_link' must be the cube you are newly joining "
                    f"TO (matching the name right after JOIN), not the cube "
                    f"already in scope."
                )
            return {
                'sql': None,
                'error': error_msg,
                'substitutions': substitutions,
            }

        measure_keys_sorted = sorted(self.measures.keys(), key=len, reverse=True)
        for key in measure_keys_sorted:
            pattern = r'\b' + re.escape(key) + r'\b'
            if not re.search(pattern, sql):
                continue
            m = self.measures[key]
            cube_name = key.split('.', 1)[0]
            table = self.cubes[cube_name]
            agg_type = m['type'].upper()
            base_expr = self._prefix_raw_columns(m['sql'], cube_name, table)

            if m['filters']:
                filter_clause = ' AND '.join(
                    self._prefix_raw_columns(f, cube_name, table) for f in m['filters']
                )
                if agg_type == 'COUNT':
                    replacement = f"COUNT(CASE WHEN {filter_clause} THEN {base_expr} END)"
                else:
                    replacement = f"{agg_type}(CASE WHEN {filter_clause} THEN {base_expr} END)"
            else:
                if agg_type in ('COUNT', 'SUM', 'AVG', 'MIN', 'MAX'):
                    replacement = f"{agg_type}({base_expr})"
                else:
                    replacement = base_expr

            sql = re.sub(pattern, replacement, sql)
            substitutions.append(f"MEASURE {key} -> {replacement}")

        # --- Step 3b: translate value-mapped dimension labels ---
        # For dimensions like trans.type_label (value_map: deposit ->
        # PRIJEM, withdrawal -> VYDAJ), find every place the concept
        # is compared against one of its label values -- via "=" or
        # "IN (...)" -- and replace BOTH the concept name (with its
        # table.column form) AND the label value (with its real code)
        # in the same pass. This must happen BEFORE the generic
        # dimension-name substitution below, because once
        # trans.type_label becomes trans.type, we'd lose the
        # information needed to know a value-translation should also
        # apply here (a bare trans.type filter, e.g. one the LLM wrote
        # directly rather than via the label dimension, should NOT
        # have its value silently reinterpreted).
        for vkey, vinfo in self.value_mapped_dimensions.items():
            cube_name = vkey.split('.', 1)[0]
            table = self.cubes[cube_name]
            raw_col = vinfo['raw_column']
            value_map = vinfo['value_map']

            # "<concept> = 'label'" or "<concept> != 'label'" etc.
            eq_pattern = re.compile(
                r'\b' + re.escape(vkey) + r'\b(\s*(?:=|!=|<>)\s*)\'(\w+)\'',
                re.IGNORECASE
            )
            def eq_replace(m, table=table, raw_col=raw_col, value_map=value_map, vkey=vkey):
                op, label = m.group(1), m.group(2)
                real_code = value_map.get(label)
                if real_code is None:
                    # LLM used a label that isn't in the value_map --
                    # leave unresolved; this will surface downstream
                    # as an execution error comparing against a
                    # nonexistent value, which is preferable to
                    # silently guessing.
                    return m.group(0)
                substitutions.append(
                    f"VALUE-MAP {vkey} = '{label}' -> {table}.{raw_col}{op}'{real_code}'"
                )
                return f"{table}.{raw_col}{op}'{real_code}'"
            sql = eq_pattern.sub(eq_replace, sql)

            # "<concept> IN ('label1', 'label2', ...)"
            in_pattern = re.compile(
                r'\b' + re.escape(vkey) + r'\b(\s*IN\s*)\(([^)]*)\)',
                re.IGNORECASE
            )
            def in_replace(m, table=table, raw_col=raw_col, value_map=value_map, vkey=vkey):
                op, inner = m.group(1), m.group(2)
                labels = re.findall(r"'(\w+)'", inner)
                if not labels:
                    return m.group(0)
                real_codes = []
                for label in labels:
                    real_code = value_map.get(label)
                    if real_code is None:
                        return m.group(0)  # unresolved label -- leave whole clause unchanged
                    real_codes.append(real_code)
                new_inner = ", ".join(f"'{c}'" for c in real_codes)
                substitutions.append(
                    f"VALUE-MAP {vkey} IN ({inner}) -> {table}.{raw_col}{op}({new_inner})"
                )
                return f"{table}.{raw_col}{op}({new_inner})"
            sql = in_pattern.sub(in_replace, sql)

        dim_keys_sorted = sorted(self.dimensions.keys(), key=len, reverse=True)
        for key in dim_keys_sorted:
            pattern = r'\b' + re.escape(key) + r'\b'
            if not re.search(pattern, sql):
                continue
            raw_col = self.dimensions[key]
            cube_name, concept_name = key.split('.', 1)
            table = self.cubes[cube_name]
            replacement = f"{table}.{raw_col}"
            sql = re.sub(pattern, replacement, sql)
            if key == replacement:
                substitutions.append(f"DIM {key} -> {replacement} (no-op: concept name equals raw column)")
            else:
                substitutions.append(f"DIM {key} -> {replacement}")

        for cube_name, table in self.cubes.items():
            if cube_name != table:
                sql = re.sub(r'\b' + re.escape(cube_name) + r'\b', table, sql)

        where_match = re.search(r'\bWHERE\b(.*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|$)',
                                  sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            agg_pattern = re.compile(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', re.IGNORECASE)
            if agg_pattern.search(where_clause):
                return {
                    'sql': None,
                    'error': (
                        "Invalid SQL: an aggregate function (from a measure "
                        "concept) was used directly inside a WHERE clause. "
                        "Measures are pre-aggregated and can only be used in "
                        "SELECT, HAVING, or ORDER BY -- not WHERE, since WHERE "
                        "filters rows before aggregation. If you need a "
                        "row-level existence check (e.g. 'has at least one "
                        "defaulted loan'), use a dimension-based condition "
                        "or a subquery instead of a measure."
                    ),
                    'substitutions': substitutions,
                }

        # --- Step 7: collapse redundant double-aggregation ---
        # BUG FIX (found in pilot run): measures are already fully
        # aggregated expressions (e.g. avg_balance compiles to
        # "AVG(balance)"), but the LLM sometimes wraps a measure
        # reference in its own matching aggregate call anyway --
        # e.g. writing "AVG(trans.avg_balance)" in concept-SQL, which
        # compiles to "AVG(AVG(balance))" and SQLite rejects with
        # "misuse of aggregate function AVG()". This also occurs
        # nested inside IFNULL, e.g. "AVG(IFNULL(AVG(balance), 0))".
        #
        # Rather than trying to prevent this at substitution time
        # (which would require tracking, for every individual measure
        # occurrence, whether the text immediately surrounding it in
        # the ORIGINAL concept-SQL was already an aggregate wrapper --
        # awkward to do cleanly mid-substitution when several measures
        # may be substituted in the same pass), we detect and collapse
        # the redundant pattern AFTER all substitutions are complete,
        # by finding "AGG(...AGG(...)...)" where both AGG are the SAME
        # function name and the inner one has no arguments of its own
        # beyond what the outer call would also see. Concretely: for
        # each aggregate function name, repeatedly replace
        # "FUNC(<balanced text with no top-level comma>FUNC(<inner>)<more text>)"
        # is unsafe to do with a single regex in general (nested
        # parens), so instead we use a small balanced-paren scan: find
        # "FUNC(" then FUNC( again immediately or after only
        # non-paren/non-comma "wrapper" text (IFNULL(, whitespace),
        # and if so, drop the OUTER "FUNC(" and its matching ")",
        # leaving the inner aggregate call intact as the sole
        # aggregation. This is safe because two aggregate calls of the
        # SAME function nested with nothing but a null-handling wrapper
        # between them is never valid, deliberate SQL -- it can only
        # arise from this substitution artifact.
        def collapse_double_aggregation(text: str) -> str:
            """
            BUG FIX (round 2): the round-1 version only detected
            IMMEDIATELY adjacent same-named aggregates, e.g.
            "AVG(AVG(x))" or "AVG(IFNULL(AVG(x), 0))". It missed cases
            where arbitrary wrapper text sits between the outer and
            inner aggregate calls, most commonly a CASE expression:
                AVG(CASE WHEN cond THEN AVG(x) END)
            which arises when the LLM applies its own aggregate to a
            CASE branch that itself evaluates to an already-aggregated
            measure. This version instead: for each occurrence of
            "AGG(", extracts the FULL span up to its matching close
            paren (real paren-depth matching, so nested parens inside
            are handled correctly), then checks whether that full span
            contains EXACTLY ONE same-named inner "AGG(...)" call and
            nothing else that looks like a real column/table reference
            (only SQL wrapper syntax: CASE, WHEN, THEN, ELSE, END,
            IFNULL, whitespace, commas, numbers). If so, the outer
            wrapper is redundant and is stripped, keeping the wrapper
            syntax (e.g. the CASE structure) intact around the
            now-unwrapped inner aggregate.
            """
            agg_names = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
            wrapper_only_tokens = {
                'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'IFNULL', 'NULL',
            }

            def find_matching_close(s: str, open_idx: int) -> int:
                depth = 0
                i = open_idx
                while i < len(s):
                    if s[i] == '(':
                        depth += 1
                    elif s[i] == ')':
                        depth -= 1
                        if depth == 0:
                            return i
                    i += 1
                return -1

            changed = True
            while changed:
                changed = False
                for agg in agg_names:
                    for m in re.finditer(re.escape(agg) + r'\s*\(', text, re.IGNORECASE):
                        outer_open_start = m.start()
                        paren_pos = m.end() - 1
                        matching_close = find_matching_close(text, paren_pos)
                        if matching_close == -1:
                            continue
                        inner_span = text[paren_pos + 1:matching_close]

                        # find same-named inner aggregate calls within this span
                        inner_matches = list(re.finditer(
                            re.escape(agg) + r'\s*\(', inner_span, re.IGNORECASE
                        ))
                        if len(inner_matches) != 1:
                            continue  # need exactly one inner same-named call to safely collapse

                        inner_m = inner_matches[0]
                        inner_paren_pos = inner_m.end() - 1
                        inner_close = find_matching_close(inner_span, inner_paren_pos)
                        if inner_close == -1:
                            continue

                        # text in inner_span OUTSIDE the inner AGG(...) call
                        # itself -- must contain nothing but wrapper syntax
                        # (CASE/WHEN/THEN/END/IFNULL/whitespace/punctuation)
                        # for this to be safely collapsible. If real column
                        # references or other function calls are present,
                        # this is NOT a simple redundant double-wrap and we
                        # leave it alone (e.g. AVG(x + AVG(y)) is NOT the
                        # same situation and should not be collapsed).
                        outside_text = (
                            inner_span[:inner_m.start()]
                            + inner_span[inner_close + 1:]
                        )
                        outside_tokens = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', outside_text)
                        non_wrapper_tokens = [t for t in outside_tokens if t.upper() not in wrapper_only_tokens]
                        if non_wrapper_tokens:
                            continue  # real content outside the inner call -- not safely collapsible

                        # Safe to collapse: remove the OUTER "AGG(" and its
                        # matching final ")", keeping everything between
                        # them (which includes the inner AGG(...) call and
                        # any CASE/WHEN/THEN/END wrapper text) intact.
                        text = (
                            text[:outer_open_start]
                            + text[paren_pos + 1:matching_close]
                            + text[matching_close + 1:]
                        )
                        changed = True
                        break
                    if changed:
                        break
            return text

        sql_before_collapse = sql
        sql = collapse_double_aggregation(sql)
        if sql != sql_before_collapse:
            substitutions.append(
                f"COLLAPSED redundant double-aggregation: "
                f"'{sql_before_collapse.strip()}' -> '{sql.strip()}'"
            )

        return {'sql': sql, 'error': None, 'substitutions': substitutions}
