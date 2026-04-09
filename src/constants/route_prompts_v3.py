"""Route prompt definitions for the evaluation pipeline.

Each entry contains the orchestrator prompt section that describes when and how
a given route is used. These are passed verbatim into the rubrics generator as
context so it can build route-aware evaluation criteria.

Route-specific keys must match the route names used in all_results.csv:
    RESEARCH, RESPOND, REVISE_SIMPLE, REVISE_RESEARCH

Additional shared keys:
    UNIVERSAL: guidance included alongside every route-specific prompt
"""

ROUTE_PROMPTS: dict[str, str] = {
    "RESEARCH": """\
Search for papers (literature review, claim verification, gap analysis).
Does the user want NEW information from the literature?
Route to RESEARCH if:
- User asks to find, search, explore, verify, validate, or check claims
- User provides a topic word or phrase without a verb (e.g., "neuroplasticity", "CRISPR delivery")
- User asks evidence-based questions: "is this unique?", "has this been done?", "is this novel?"
- User wants to scope a field: "how would I approach X", "what do we know about X"
- User says "tell me more" or asks for more information about a topic (even with prior research in history — this requests DEEPER literature search, not a summary)
- User asks about strength, validity, or defensibility of claims
- User's question would be meaningfully improved by citing actual papers
- Message combines research + editing ("find papers and add citations") → RESEARCH first
- Imperative command naming a research topic ("write about gut microbiome") → RESEARCH
- Affirmative after topic introduction ("yes", "go ahead", "sounds good") with unresearched topic in history → RESEARCH (inherit topic from history)

Intent mapping:
- validate_claims: check/verify claims.
- explore_literature: find papers, explore what's known.
- identify_gaps: find what's missing or weak.
""",

    "RESPOND": """\
Answer questions, discuss the document, synthesize prior findings, or general conversation.
Use RESPOND for:
- Summarize, compare, or analyze information already in conversation
- Questions about document structure or writing process
- Meta-commentary ("what do you think?", "is this approach reasonable?")
- Questions about the assistant's capabilities
- Follow-up about previously found papers that doesn't need new search

- Selected text that is an HTML table, markdown pipe table, or LaTeX tabular → always RESPOND (merge system cannot splice tables).
- Requests to evade plagiarism detection, fabricate citations, disguise copied text, or write assignments for submission → always RESPOND with refusal.
- Unbounded requests ("fix everything", "make it perfect", "write the whole paper") → always RESPOND.


""",

    "REVISE_SIMPLE": """\
Edit text without new research (grammar, style, tone, reformatting, or applying prior findings).
Route to REVISE_SIMPLE if:
- Mechanical changes: grammar, spelling, punctuation, formatting
- Rephrase, simplify, shorten, change tone, restructure
- Convert to bullets, active voice, remove jargon
- Prior paper search results exist AND user asks to apply them ("use those papers", "revise based on what you found")
- User asks to transfer content from a prior assistant turn to the document ("put this in my document", "add that")
- User provides the specific content to write — names mechanisms, states facts, dictates phrasing (user is the source of truth, not literature)
- Explicit detailed instructions about what to write


""",

    "REVISE_RESEARCH": """\
Route to REVISE_RESEARCH if:
Edit text with new research (strengthen argument, add evidence, improve with citations).
- User asks to add citations, evidence, or sources to text
- Vague improvement commands ("improve this", "make it better", "strengthen this") AND no prior paper search results exist
- Add or insert new sections WITHOUT specifying content ("add a conclusion", "write an introduction")
- Text revision would benefit from external evidence AND no prior papers are available

""",

    "UNIVERSAL": """\
# Is the document empty (0 chars)?
- Empty doc + clear research topic (in message or history) → RESEARCH / explore_literature
- Empty doc + revision command with no topic → RESPOND
- Empty doc + doc-dependent intent (validate_claims, identify_gaps) → RESPOND

# Tie-breaker for REVISE_SIMPLE vs REVISE_RESEARCH:
- Does the user's message itself contain the facts/content to write? → REVISE_SIMPLE
- Does the user only name a topic without specifying what to say? → REVISE_RESEARCH
- Does general discussion (no papers) exist in history? → This is NOT prior research → REVISE_RESEARCH

#  NOT RESPOND:
- Imperative creation commands ("write a paragraph on…", "make a list of…") → REVISE_RESEARCH
- Questions about subject matter strength/validity → RESEARCH
- Research scoping questions ("how would I approach X") → RESEARCH
""",
}
