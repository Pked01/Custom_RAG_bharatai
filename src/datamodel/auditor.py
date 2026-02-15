from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
	"""A traceable reference to a specific clause in a contract."""

	id: int = Field(description="Unique citation ID within this response")
	file_name: str = Field(description="Source contract file name")
	section_number: str = Field(description="Section number (e.g., '7.2')")
	section_header: str = Field(description="Section header text")
	verbatim_quote: str = Field(description="Exact quoted text from the contract")


class RiskItem(BaseModel):
	"""A single identified risk with severity and citations."""

	id: int = Field(description="Risk ID within this response")
	title: str = Field(description="Short risk title (3-8 words)")
	severity: Literal["high", "medium", "low"] = Field(description="Risk severity level")
	description: str = Field(description="What the risk is and why it matters")
	citation_ids: list[int] = Field(default_factory=list, description="Citation IDs supporting this risk")
	quote_snippet: str = Field(default="", description="Key phrase from evidence")


class DocumentPosition(BaseModel):
	"""What a specific document says about a topic."""

	file_name: str = Field(description="Source contract file name")
	section_number: str = Field(description="Section number where this appears")
	section_header: str = Field(description="Section header text")
	position: str = Field(description="What this document states about the topic (1-2 sentences)")
	verbatim_quote: str = Field(description="Exact quoted text supporting the position")


class TopicComparison(BaseModel):
	"""Comparison of how different documents address the same topic."""

	topic: str = Field(description="The shared topic being compared (e.g., 'Governing Law', 'Liability Cap')")
	positions: list[DocumentPosition] = Field(description="What each document says about this topic")
	has_conflict: bool = Field(description="Whether the documents conflict on this topic")
	conflict_description: str = Field(
		default="",
		description="If has_conflict=True, explain the nature of the conflict"
	)
	severity: Literal["high", "medium", "low"] = Field(
		default="low",
		description="Severity if conflict exists: high=legal exposure, medium=operational risk, low=minor inconsistency"
	)


class AuditorResponse(BaseModel):
	"""Structured response from the Risk Auditor node."""

	answer_status: Literal["answered", "partial", "cannot_answer", "needs_clarification"] = Field(
		description="Whether the question could be answered from evidence"
	)
	primary_answer: str = Field(
		description="1-3 sentence direct answer. If cannot_answer, explain why."
	)
	confidence: Literal["high", "medium", "low"] = Field(
		description="Overall confidence in the answer"
	)
	confidence_reason: str = Field(
		default="", description="Why this confidence level was assigned"
	)

	# Cross-document comparison fields
	is_comparison_query: bool = Field(
		default=False,
		description="True if user asked about conflicts/differences across documents"
	)
	topic_comparisons: list[TopicComparison] = Field(
		default_factory=list,
		description="For comparison queries: topic-by-topic breakdown across documents"
	)

	# Standard risk assessment fields
	risks: list[RiskItem] = Field(
		default_factory=list, description="Top risks, max 3, ranked by severity"
	)
	citations: list[Citation] = Field(
		default_factory=list, description="All citations referenced in the response"
	)
	unanswered_aspects: list[str] = Field(
		default_factory=list, description="Parts of the question not answered and why"
	)
	suggested_followup: Optional[str] = Field(
		default=None, description="What to ask next or verify externally"
	)
	needs_human_review: bool = Field(
		default=False, description="Whether human verification is recommended"
	)
	review_reason: Optional[str] = Field(
		default=None, description="Why human review is needed"
	)

	def to_final_answer(self) -> str:
		"""Convert structured response to clean, readable text."""
		parts: list[str] = []

		# Primary answer (clean, no status prefix clutter)
		parts.append(self.primary_answer)

		# Cross-document comparison section
		if self.is_comparison_query and self.topic_comparisons:
			parts.append("")
			for comp in self.topic_comparisons:
				conflict_marker = "⚠️ Conflict detected" if comp.has_conflict else "✓ Consistent"
				parts.append(f"**{comp.topic}** — {conflict_marker}")

				for pos in comp.positions:
					parts.append(f"• {pos.file_name}: {pos.position}")

				if comp.has_conflict and comp.conflict_description:
					parts.append(f"  → {comp.conflict_description}")
				parts.append("")

		# Key finding (only top risk, not bulleted list)
		if self.risks and not self.is_comparison_query:
			top_risk = self.risks[0]
			parts.append("")
			parts.append(f"**Key finding:** {top_risk.title}")
			parts.append(top_risk.description)

		# Sources as footnotes
		if self.citations:
			source_refs = [f"[{c.id}] {c.file_name} §{c.section_number}" for c in self.citations]
			parts.append("")
			parts.append(f"Sources: {' · '.join(source_refs)}")

		# Follow-up
		if self.suggested_followup:
			parts.append("")
			parts.append(f"💡 {self.suggested_followup}")

		return "\n".join(parts)

	def to_legacy_risk_report(self) -> list[str]:
		"""Convert to legacy risk_report format for backward compatibility."""
		# Now returns clean format instead of severity-tagged bullets
		lines: list[str] = []
		lines.append(self.primary_answer)

		if self.is_comparison_query and self.topic_comparisons:
			for comp in self.topic_comparisons:
				conflict = "⚠️ Conflict" if comp.has_conflict else "✓ OK"
				lines.append(f"{comp.topic}: {conflict}")
				for pos in comp.positions:
					lines.append(f"  • {pos.file_name}: {pos.position}")

		if self.risks and not self.is_comparison_query:
			top_risk = self.risks[0]
			lines.append(f"Key finding: {top_risk.title} — {top_risk.description}")

		if self.citations:
			source_refs = [f"[{c.id}] {c.file_name}" for c in self.citations]
			lines.append(f"Sources: {', '.join(source_refs)}")

		return lines
