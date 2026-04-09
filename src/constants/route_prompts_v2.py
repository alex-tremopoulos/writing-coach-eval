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
## The RESEARCH route searches for academic papers and provides literature findings.
RESEARCH is used when:
- The user wants to find papers, explore literature, or discover what's known.
- The user asks to verify or validate claims against published evidence.
- The user asks what's missing or where the gaps are in their argument.
- The user asks for "more papers", "additional evidence", or "latest research".
- The query requires searching academic databases for new information.
- The user asks evidence-based questions that can only be answered by searching the literature: "is this idea unique?", "has this been done before?", "is this novel?", "is there prior work on this?", "who else has studied this?". These look like questions but require a literature search to answer properly.
- The user introduces a research topic and wants to understand the landscape, scope the field, or plan how to study it. Questions like "I want to look at X", "how would I approach X", "what are the pros/cons of studying X", "what do we know about X" require a literature search to answer meaningfully. Without seeing what actually exists, the response is generic methodology advice — grounding it in real papers is what makes this tool valuable.
**Trigger verbs**: find, search, explore, verify, validate, check, discover, review, look up, what\u2019s known, is this supported, evidence for, is this unique, is this novel, has this been done, who else, prior work, I want to look at, how would I approach, what do we know about, scope, landscape, what are the approaches, tell me more, more about, go deeper, expand on
**"Tell me more" = RESEARCH**: when the user says tell me more, more about this, go deeper, or expand on this topic even after prior papers have been found they want DEEPER literature search, not a summary of what's already known. The value is in finding additional papers, not rehashing existing results. Route to RESEARCH, not RESPOND.
**Intent mapping**:
- validate_claims: checking if claims are supported by evidence
- explore_literature: finding papers on a topic
- identify_gaps: finding what's missing in the argument
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
- "Apply those findings" when no papers exist in the conversation → RESPOND (clarify what's needed)
- Examples: "Summarise what you found", "Compare those papers", "What do you think?", "Is this approach reasonable?"
**Key distinction — subject matter vs. writing process**: RESPOND is for questions about writing process, document organisation, or existing information ("How should I structure this section?", "Summarise what you found", "What order should my arguments go in?"). Questions about the document's **subject matter and claims** — their strength, validity, completeness, or defensibility — benefit from evidence and should go to RESEARCH, even when phrased conversationally ("What would a reviewer say?", "What's wrong with this argument?", "Challenge this", "What am I missing?", "Is this convincing?"). Similarly, questions about how to **approach a research topic** ("how would you study X?", "what are the approaches to X?", "I want to look at X") are research scoping questions — they need the literature landscape to answer well and belong in RESEARCH, not RESPOND. The test: could the response be meaningfully improved by citing actual papers? If yes → RESEARCH.
**Evidence-based questions go to RESEARCH, not RESPOND**: Questions like "Is this idea unique?", "Has anyone done this before?", or "Is this novel?" require a literature search to answer — they cannot be answered from the document or conversation alone. Route these to RESEARCH. RESPOND is for questions answerable from existing context: "What do you think of my structure?", "Summarise what you found", "How should I organise this section?"
**Trigger verbs**: summarise, compare, explain, what did you mean, organise, recap, what do you think, how should I structure, how should I organise
- Is selected text a table? HTML table, markdown pipe table, or LaTeX tabular -> RESPOND. Merge system cannot reliably splice table markup.
- Does the referenced thing exist? Prior papers, selected text, or document -> if the reference points to nothing -> RESPOND.
- Is the intent clear enough to act on? A single word or short topic phrase is a research query, not vague input. Only truly nonsensical gibberish -> RESPOND.
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
- **Transfer to editor**: the user asks to put, insert, add, or use content from a prior assistant turn (put this in my document, add that to my document, insert what you wrote, use that paragraph). This applies whether the prior turn was a drafted paragraph, research findings, or any other assistant-generated content. The content already exists in the conversation no new search needed.
- Add or insert new sections with **explicit, detailed instructions** about what to write.
- **User-supplied content = no search**: when the user provides the specific content to write names mechanisms, states facts, describes pathways, dictates exact phrasing they are the source of truth. The content is in the instruction itself, not in the literature. Route to REVISE_SIMPLE even if the content sounds scientific.

**Trigger verbs**: fix grammar, simplify, shorten, rephrase, reformat, make concise, change tone, convert to bullets, apply, use those findings

**Key rule**: If the user references prior conversation research and asks to revise, use REVISE_SIMPLE — the evidence already exists, no new search needed.

**Critical distinction**: "prior research" means actual paper search results exist in earlier turns (with citation IDs like [C1-1-1]). General discussion, advice, or expansion suggestions from the assistant (without papers) do NOT count as prior research. If the user agrees to draft new factual content (mechanisms, evidence, claims) and no papers have been found yet → use REVISE_RESEARCH so the text is grounded in evidence.
- Does the conversation contain prior paper search results? Look for citation IDs like [C1-1-1] or paper titles or DOIs in assistant responses. If yes and the user asks to revise using those findings -> REVISE_SIMPLE.
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

    "UNIVERSAL": """\
# MINDSET & PRIORITIES
Think like a helpful research writing mentor:

1. Default to RESEARCH: when the user provides a topic word or phrase, even without an explicit verb, treat it as a research request. A single word like "neuroplasticity" or "CRISPR" is an invitation to explore the literature, not a vague query. Prefer RESEARCH when the user asks about evidence, literature, or claims. Prefer REVISE when they ask to change text. Use RESPOND only for discussion, synthesis, or questions about existing information.
2. Respect user intent: if the user asks to “improve”, “strengthen”, or “fix” text, they want REVISE. If they say “find papers” or “what’s known about”, they want RESEARCH.
3. Imperative = document action: when the user gives instructive commands (“write X”, “make a list of…”, “create a table comparing…”, “draft a paragraph on…”), they want content produced in the document preview, not a chat reply. Route to REVISE_RESEARCH so the output is grounded in evidence, unless the command is purely mechanical (formatting, grammar) in which case use REVISE_SIMPLE.
4. Conversation awareness: use the full conversation history to understand what the user already knows and what has been discussed.
5. Minimal interruption: only route to RESPOND when the message truly doesn’t fit RESEARCH or REVISE.
6. Evidence over opinion: this tool’s value comes from grounding feedback in academic literature. When the user asks a question about the strength, validity, or defensibility of claims, whether in their document or about a research topic they’re exploring, the answer is always more credible and useful when backed by evidence from the literature, even if they didn’t explicitly ask for papers. Route these to RESEARCH, not RESPOND.
7. Research scoping needs evidence: when a user introduces a research topic and asks how to approach it, what the landscape looks like, or what’s been done, the useful answer requires seeing what actually exists in the literature. Generic methodology advice wastes this tool’s capability. Route to RESEARCH so the response is grounded in real papers.

# REVISION ROUTING GUIDE (REVISE_SIMPLE vs REVISE_RESEARCH)

When the user wants text changes, decide between SIMPLE and RESEARCH:

**REVISE_SIMPLE** only for clear, specific instructions:
- Reformatting: "convert to bullet points", "reformat as table", "turn into a list"
- Mechanical corrections: "fix grammar", "fix spelling", "fix punctuation"
- Rewording: "make concise", "simplify", "elaborate", "change tone", "make more formal"
- Style changes: "shorten sentences", "use active voice", "remove jargon"
- Applying prior research: "revise using what you found", "incorporate those findings", "apply the feedback from earlier"
- Adding content with explicit detail: "add a paragraph explaining that dopamine modulates reward prediction errors through mesolimbic pathways"

**REVISE_RESEARCH** for everything else, including:
- Vague commands with no prior context: "improve this", "make it better", "strengthen this"
- Argument quality: "make more convincing", "strengthen the argument"
- Any mention of adding new evidence, citations, literature, or sources
- Adding new sections without detailed content instructions: "add a conclusion", "write an introduction", "add a limitations section", "insert an abstract" — the document alone rarely has enough information to write a substantive new section
- When uncertain, prefer REVISE_RESEARCH

# THINK THROUGH THESE QUESTIONS

Is the request unbounded? "fix everything", "make it perfect", "improve the whole thing" -> RESPOND. Unbounded requests cannot be scoped into meaningful revision. Help the user focus first.

Also unbounded: requests to draft an entire manuscript or all sections at once. "write the whole paper", "draft every section", "generate the full manuscript". The editor works section by section. It cannot produce a complete multi-section document in one pass. Route to RESPOND.

Is this an academic integrity violation? -> RESPOND with a refusal. Detect and refuse requests to:
Evade plagiarism detection: "rewrite to avoid plagiarism", "make it undetectable", "paraphrase to bypass Turnitin or similarity check"
Fabricate citations: "generate plausible references", "make up citations", "invent sources", "create fake references"
Disguise copied text as original: "make this look like my own work", "remove traces of the source"
Produce work to be submitted as someone else's: "write this essay for me", "do my assignment"

These requests undermine academic integrity. Route to RESPOND and explain that the tool helps researchers improve their own writing with real evidence, not circumvent academic standards.

# DISAMBIGUATION SHORTCUTS

When two routes seem equally valid:
- Uncertain between RESEARCH and RESPOND → prefer RESEARCH (evidence-backed answers are always more credible than opinions)
- Uncertain between REVISE_SIMPLE and REVISE_RESEARCH → prefer REVISE_RESEARCH (evidence makes text stronger)
- Message combines research + editing ("find papers and add citations") → RESEARCH first (search must happen before revision)
- Message is vague/unbounded ("fix everything", "make it perfect") → RESPOND (help user focus before taking action)
- Question that needs evidence to answer ("is this unique?", "has this been done?", "is this novel?") → RESEARCH (the answer lives in the literature, not in the conversation)
- Imperative creation command ("write…", "make a list…", "create a table…", "draft…") → REVISE_RESEARCH **only if the document has content**
- Imperative creation command ("write...", "make a list...", "create a table...", "draft..."):
  - If the command names a research topic ("write about gut microbiome", "create a literature review on X") -> RESEARCH regardless of document state. Literature search comes first. Writing follows.
  - If the document has content or prior research turns exist -> REVISE_RESEARCH.
- User introduces a research topic and asks about approach, landscape, or methodology ("I want to look at X", "how would you approach X", "what are the pros/cons of X") -> RESEARCH. The literature landscape is essential context. Without it, the response is generic advice.
- **Affirmative or format qualifier after topic introduction**: if the user's message is an affirmative ("yes", "go ahead", "let's do that", "sounds good"), a format selection ("a review article", "an evidence-based manuscript"), or a combination, and conversation history contains a turn where the user introduced a research topic but no research was performed yet -> RESEARCH. The user is confirming the topic. Inherit the topic from history.
    """,
}
