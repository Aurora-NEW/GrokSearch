from pydantic import BaseModel, Field
from typing import Optional
import uuid


class IntentOutput(BaseModel):
    core_question: Optional[str] = Field(default=None, description="Distilled core question in one sentence")
    query_type: Optional[str] = Field(
        default=None,
        description="factual=single answer, comparative=A vs B, exploratory=broad understanding, analytical=deep reasoning"
    )
    time_sensitivity: Optional[str] = Field(
        default=None,
        description="realtime=today, recent=days/weeks, historical=months+, irrelevant=timeless"
    )
    domain: Optional[str] = Field(default=None, description="Specific domain if identifiable")
    premise_valid: Optional[bool] = Field(default=None, description="False if the question contains a flawed assumption")
    ambiguities: Optional[list[str]] = Field(default=None, description="Unresolved ambiguities that may affect search direction")
    unverified_terms: Optional[list[str]] = Field(
        default=None,
        description="External classifications, rankings, or taxonomies that may be incomplete or outdated "
        "in training data (e.g., 'CCF-A', 'Fortune 500', 'OWASP Top 10'). "
        "Each should become a prerequisite sub-query in Phase 3."
    )

    model_config = {"extra": "allow"}


class ComplexityOutput(BaseModel):
    level: Optional[int] = Field(
        default=None,
        description="1=simple (1-2 searches), 2=moderate (3-5 searches), 3=complex (6+ searches)"
    )
    estimated_sub_queries: Optional[int] = Field(default=None)
    estimated_tool_calls: Optional[int] = Field(default=None)
    justification: Optional[str] = Field(default=None)

    model_config = {"extra": "allow"}


class SubQuery(BaseModel):
    id: Optional[str] = Field(default=None, description="Unique identifier (e.g., 'sq1')")
    goal: Optional[str] = Field(default=None)
    expected_output: Optional[str] = Field(default=None, description="What a successful result looks like")
    tool_hint: Optional[str] = Field(default=None, description="Suggested tool: web_search | web_fetch | web_map")
    boundary: Optional[str] = Field(default=None, description="What this sub-query explicitly excludes")
    depends_on: Optional[list[str]] = Field(default=None, description="IDs of prerequisite sub-queries")

    model_config = {"extra": "allow"}


class SearchTerm(BaseModel):
    term: Optional[str] = Field(default=None, description="Search query string.")
    purpose: Optional[str] = Field(default=None, description="Single sub-query ID this term serves (e.g., 'sq2').")
    round: Optional[int] = Field(default=None, description="Execution round: 1=broad discovery, 2+=targeted follow-up")

    model_config = {"extra": "allow"}


class StrategyOutput(BaseModel):
    approach: Optional[str] = Field(
        default=None,
        description="broad_first=wide then narrow, narrow_first=precise then expand, targeted=known-item"
    )
    search_terms: Optional[list[SearchTerm]] = Field(default=None)
    fallback_plan: Optional[str] = Field(default=None, description="Fallback if primary searches fail")

    model_config = {"extra": "allow"}


class ToolPlanItem(BaseModel):
    sub_query_id: Optional[str] = Field(default=None)
    tool: Optional[str] = Field(default=None)
    reason: Optional[str] = Field(default=None)
    params: Optional[dict] = Field(default=None, description="Tool-specific parameters")

    model_config = {"extra": "allow"}


class ExecutionOrderOutput(BaseModel):
    parallel: Optional[list[list[str]]] = Field(default=None, description="Groups of sub-query IDs runnable in parallel")
    sequential: Optional[list[str]] = Field(default=None, description="Sub-query IDs that must run in order")
    estimated_rounds: Optional[int] = Field(default=None)

    model_config = {"extra": "allow"}


PHASE_NAMES = [
    "intent_analysis",
    "complexity_assessment",
    "query_decomposition",
    "search_strategy",
    "tool_selection",
    "execution_order",
]

REQUIRED_PHASES: dict[int, set[str]] = {
    1: {"intent_analysis", "complexity_assessment", "query_decomposition"},
    2: {"intent_analysis", "complexity_assessment", "query_decomposition", "search_strategy", "tool_selection"},
    3: set(PHASE_NAMES),
}


class PhaseRecord(BaseModel):
    phase: str
    thought: str
    data: dict | list | None = None
    confidence: float = 1.0


class PlanningSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.phases: dict[str, PhaseRecord] = {}
        self.complexity_level: int | None = None

    @property
    def completed_phases(self) -> list[str]:
        return [p for p in PHASE_NAMES if p in self.phases]

    def required_phases(self) -> set[str]:
        return REQUIRED_PHASES.get(self.complexity_level or 3, REQUIRED_PHASES[3])

    def is_complete(self) -> bool:
        if self.complexity_level is None:
            return False
        return self.required_phases().issubset(self.phases.keys())

    def build_executable_plan(self) -> dict:
        return {name: record.data for name, record in self.phases.items()}


class PlanningEngine:
    def __init__(self):
        self._sessions: dict[str, PlanningSession] = {}

    def process_phase(
        self,
        phase: str,
        thought: str,
        session_id: str = "",
        is_revision: bool = False,
        revises_phase: str = "",
        confidence: float = 1.0,
        phase_data: dict | list | None = None,
    ) -> dict:
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
        else:
            sid = session_id if session_id else uuid.uuid4().hex[:12]
            session = PlanningSession(sid)
            self._sessions[sid] = session

        target = revises_phase if is_revision and revises_phase else phase
        if target not in PHASE_NAMES:
            return {"error": f"Unknown phase: {target}. Valid: {', '.join(PHASE_NAMES)}"}

        session.phases[target] = PhaseRecord(
            phase=target, thought=thought, data=phase_data, confidence=confidence
        )

        if target == "complexity_assessment" and isinstance(phase_data, dict):
            level = phase_data.get("level")
            if level in (1, 2, 3):
                session.complexity_level = level

        complete = session.is_complete()
        result: dict = {
            "session_id": session.session_id,
            "completed_phases": session.completed_phases,
            "complexity_level": session.complexity_level,
            "plan_complete": complete,
        }

        remaining = [p for p in PHASE_NAMES if p in session.required_phases() and p not in session.phases]
        if remaining:
            result["phases_remaining"] = remaining

        if complete:
            result["executable_plan"] = session.build_executable_plan()

        return result


engine = PlanningEngine()
