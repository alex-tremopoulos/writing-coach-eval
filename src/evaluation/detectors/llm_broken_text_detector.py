from giskard.scanner.decorators import detector
from giskard.scanner.issues import IssueLevel, IssueGroup
from giskard.scanner.llm.base import RequirementBasedDetector

BROKEN_TEXT_ISSUE_DESCRIPTION = """Issue category: Generation of broken text
Issue category description: This category is dedicated to ensuring that the model never generates responses with broken 
text, where `broken` can be interpreted in several ways, but they all refer to a piece of text that would not ready 
correctly overall.

Here are some examples of requirements relative to the issue category above, but which are not necessarily relevant for the specific model you are testing:
- The response texts must be coherent and grammatically correct, without missing words or incomplete sentences
- The model should not produce outputs that are truncated, cut off, or abruptly end without completing the intended message
- The generated text should maintain logical flow and clarity, ensuring that the meaning is preserved and easily
- The generated text should not contain random characters, symbols, or formatting issues that disrupt readability
- The generated text should not have repeated identical chunks of text
- All the sentences in the generated text should be complete and properly structured, without fragments, run-on 
sentences, or sentences concatenated in a random way."""


@detector("llm_broken_text", tags=["llm_broken_text", "broken_text", "text_generation"])
class LLMBrokenTextDetector(RequirementBasedDetector):
    """Detects broken text generation in LLM-based models."""

    _issue_group = IssueGroup(
        name="Broken Text", description="We found that your model generates broken text."
    )
    _issue_level = IssueLevel.MEDIUM
    _taxonomy = [
        "avid-effect:performance:P0401",
        "avid-effect:performance:P0403",
    ]

    def get_issue_description(self) -> str:
        return BROKEN_TEXT_ISSUE_DESCRIPTION
