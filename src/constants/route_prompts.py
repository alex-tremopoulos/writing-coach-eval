"""Route prompt definitions for the evaluation pipeline.

Each entry contains the orchestrator prompt section that describes when and how
a given route is used. These are passed verbatim into the rubrics generator as
context so it can build route-aware evaluation criteria.

Keys must match the route names used in all_results.csv:
    RESEARCH, RESPOND, REVISE_SIMPLE, REVISE_RESEARCH
"""

ROUTE_PROMPTS: dict[str, str] = {
    "RESEARCH": """\
## The RESEARCH route searches for academic papers and provides literature findings.
RESEARCH is used when:
- The user wants to find papers, explore literature, or discover what's known.
- The user asks to verify or validate claims against published evidence.
- The user asks what's missing or where the gaps are in their argument.
- The user asks for "more papers", "additional evidence", or "latest research".
- The query requires searching academic databases for new information.
- The user asks evidence-based questions that can only be answered by searching the literature: "is this idea unique?", "has this been done before?", "is this novel?", "is there prior work on this?", "who else has studied this?". These look like questions but require a literature search to answer properly.
- The user introduces a research topic and wants to understand the landscape, scope the field, or plan how to study it. Questions like "I want to look at X", "how would I approach X", "what are the pros/cons of studying X", "what do we know about X" require a literature search to answer meaningfully. Without seeing what actually exists, the response is generic methodology advice — grounding it in real papers is what makes this tool valuable.
**Trigger verbs**: find, search, explore, verify, validate, check, discover, review, look up, what's known, is this supported, evidence for, is this unique, is this novel, has this been done, who else, prior work, I want to look at, how would I approach, what do we know about, scope, landscape, what are the approaches
**Intent mapping**:
- validate_claims: checking if claims are supported by evidence
- explore_literature: finding papers on a topic
- identify_gaps: finding what's missing in the argumen
""",

    "RESPOND": """\
## The RESPOND route answers questions, discusses, or synthesizes
RESPOND is used when:
- The user wants to summarise, compare, or analyse information already provided in the conversation.
- The user asks to reformat or reorganise findings from previous turns.
- The user references specific earlier turns or findings without requesting new research or text edits.
- The user asks general questions about their document structure, writing process, or the assistant itself.
- The user makes meta-commentary ("What do you think?", "Is this approach reasonable?").
- The user asks about the assistant's capabilities or how to use it.
- The message is a follow-up question about previously found papers that doesn't require new search.
- Examples: "Summarise what you found", "Compare those papers", "What do you think?", "Is this approach reasonable?"
**Key distinction — subject matter vs. writing process**: RESPOND is for questions about writing process, document organisation, or existing information ("How should I structure this section?", "Summarise what you found", "What order should my arguments go in?"). Questions about the document's **subject matter and claims** — their strength, validity, completeness, or defensibility — benefit from evidence and should go to RESEARCH, even when phrased conversationally ("What would a reviewer say?", "What's wrong with this argument?", "Challenge this", "What am I missing?", "Is this convincing?"). Similarly, questions about how to **approach a research topic** ("how would you study X?", "what are the approaches to X?", "I want to look at X") are research scoping questions — they need the literature landscape to answer well and belong in RESEARCH, not RESPOND. The test: could the response be meaningfully improved by citing actual papers? If yes → RESEARCH.
**Evidence-based questions go to RESEARCH, not RESPOND**: Questions like "Is this idea unique?", "Has anyone done this before?", or "Is this novel?" require a literature search to answer — they cannot be answered from the document or conversation alone. Route these to RESEARCH. RESPOND is for questions answerable from existing context: "What do you think of my structure?", "Summarise what you found", "How should I organise this section?"
**Trigger verbs**: summarise, compare, explain, what did you mean, organise, recap, what do you think, how should I structure, how should I organise

**NOT RESPOND**: imperative creation commands ("write a paragraph on…", "make a list of…", "create a table comparing…", "draft a section about…") should go to REVISE, not RESPOND — the user wants content produced in their document, not discussed in chat.
""",

    "REVISE_SIMPLE": """\
## The REVISE_SIMPLE route edits text without new research.
REVISE_SIMPLE is used when:
- The user gives **clear, specific instructions** that don't need external evidence, like:
- Mechanical text changes: grammar, spelling, punctuation, formatting.
- Rephrase, simplify, shorten, change tone, or restructure without new evidence.
- Convert to bullets, make concise, use active voice, remove jargon.
- Apply findings from a PREVIOUS conversation turn to the text (e.g., "revise based on what you found", "apply that feedback", "use those papers to improve this").
- Add or insert new sections with **explicit, detailed instructions** about what to write (e.g., "add a paragraph explaining that X causes Y through Z mechanism").

**Trigger verbs**: fix grammar, simplify, shorten, rephrase, reformat, make concise, change tone, convert to bullets, apply, use those findings

**Key rule**: If the user references prior conversation research and asks to revise, use REVISE_SIMPLE — the evidence already exists, no new search needed.

**Critical distinction**: "prior research" means actual paper search results exist in earlier turns (with citation IDs like [C1-1-1]). General discussion, advice, or expansion suggestions from the assistant (without papers) do NOT count as prior research. If the user agrees to draft new factual content (mechanisms, evidence, claims) and no papers have been found yet → use REVISE_RESEARCH so the text is grounded in evidence.
""",

    "REVISE_RESEARCH": """\
## The REVISE_RESEARCH edits text with new research
REVISE_RESEARCH is used when:
- The user asks to improve, strengthen, or enhance text and no prior research is available.
- The user asks to add citations, evidence, or sources to text.
- The user gives vague revision commands ("improve this", "make it better", "strengthen this") without prior relevant research in the conversation.
- The user asks to add or insert new sections **without specifying what to write** (e.g., "add a conclusion", "write an introduction", "add a limitations section", "insert an abstract"). These need evidence to produce substantive content rather than generic boilerplate.
- The revision would benefit from new external evidence.

**Trigger verbs**: strengthen with evidence, add citations, add sources, make more convincing, improve argument, add a conclusion, write an introduction, add a section

**Default for ambiguous revisions**: When a revision command is vague AND no relevant prior research exists in the conversation, use REVISE_RESEARCH. This includes requests to create new sections without detailed instructions about their content.
""",
}
