#!/usr/bin/env python3
"""
Contract Analysis CLI — Interactive conversation with the Multi-Agent RAG system.

Usage:
    python main_cli.py [options]

Examples:
    python main_cli.py                      # Basic conversation
    python main_cli.py -v                   # Verbose mode (show all debug info)
    python main_cli.py -c -r                # Show chunks and reflection scores
    python main_cli.py -t my-session        # Use custom thread ID
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from src.graph.builder import build_graph
from src.utils.config import load_guardian_config


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CLIConfig:
    """Runtime configuration for CLI session."""
    thread_id: str = "cli-session-1"
    show_chunks: bool = False
    show_reflection: bool = False
    verbose: bool = False
    use_color: bool = True
    enable_reflection: bool = False

    # Mutable state
    conversation_count: int = field(default=0, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal Formatting
# ─────────────────────────────────────────────────────────────────────────────

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


def c(text: str, color: str, config: CLIConfig) -> str:
    """Apply color to text if colors are enabled."""
    if not config.use_color:
        return text
    return f"{color}{text}{Colors.RESET}"


def print_divider(config: CLIConfig, char: str = "─", width: int = 60) -> None:
    """Print a horizontal divider line."""
    print(c(char * width, Colors.DIM, config))


def print_header(text: str, config: CLIConfig) -> None:
    """Print a section header."""
    print(c(f"\n{text}", Colors.BOLD + Colors.CYAN, config))


def print_debug(label: str, value: Any, config: CLIConfig) -> None:
    """Print debug information."""
    print(c(f"[DEBUG] {label}: ", Colors.DIM, config) + str(value))


def print_error(text: str, config: CLIConfig) -> None:
    """Print error message."""
    print(c(f"[ERROR] {text}", Colors.RED, config))


def print_warning(text: str, config: CLIConfig) -> None:
    """Print warning message."""
    print(c(f"[WARN] {text}", Colors.YELLOW, config))


def print_info(text: str, config: CLIConfig) -> None:
    """Print info message."""
    print(c(f"[INFO] {text}", Colors.BLUE, config))


# ─────────────────────────────────────────────────────────────────────────────
# Response Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_inline_citations(text: str, config: CLIConfig) -> str:
    """Convert [N] citation refs to colored superscript-style markers."""
    def replace_cite(match):
        cite_num = match.group(1)
        return c(f"[{cite_num}]", Colors.CYAN + Colors.BOLD, config)
    return re.sub(r"\[(\d+)\]", replace_cite, text)


def format_response(state: dict[str, Any], config: CLIConfig) -> None:
    """Format and print the main response."""
    auditor_response = state.get("auditor_response", {})
    answer_status = auditor_response.get("answer_status", "partial")
    primary_answer = auditor_response.get("primary_answer", "")
    is_comparison_query = auditor_response.get("is_comparison_query", False)
    topic_comparisons = auditor_response.get("topic_comparisons", [])
    risks = auditor_response.get("risks", [])
    citations = auditor_response.get("citations", [])
    suggested_followup = auditor_response.get("suggested_followup")
    needs_human_review = auditor_response.get("needs_human_review", False)
    review_reason = auditor_response.get("review_reason", "")

    print()
    print_divider(config)

    # ─────────────────────────────────────────────────────────────
    # 1. Primary Answer
    # ─────────────────────────────────────────────────────────────
    if answer_status == "cannot_answer":
        print_warning(primary_answer, config)
    elif answer_status == "needs_clarification":
        print_info(primary_answer, config)
    else:
        formatted_answer = format_inline_citations(primary_answer, config)
        print(formatted_answer)

    # ─────────────────────────────────────────────────────────────
    # 2. Comparison Table (cross-document queries)
    # ─────────────────────────────────────────────────────────────
    if is_comparison_query and topic_comparisons:
        print()
        for comp in topic_comparisons:
            topic = comp.get("topic", "Unknown Topic")
            positions = comp.get("positions", [])
            has_conflict = comp.get("has_conflict", False)
            conflict_desc = comp.get("conflict_description", "")

            conflict_marker = c("⚠ CONFLICT", Colors.RED + Colors.BOLD, config) if has_conflict else c("✓ Aligned", Colors.GREEN, config)
            print(c(f"\n{topic}", Colors.BOLD, config) + f" — {conflict_marker}")

            for pos in positions:
                file_name = pos.get("file_name", "unknown")
                position_text = pos.get("position", "")
                print(f"  • {c(file_name, Colors.CYAN, config)}: {position_text}")

            if has_conflict and conflict_desc:
                print(c(f"  → {conflict_desc}", Colors.YELLOW, config))

    # ─────────────────────────────────────────────────────────────
    # 3. Key Findings (risk queries)
    # ─────────────────────────────────────────────────────────────
    if risks and not is_comparison_query:
        top_risk = risks[0]
        title = top_risk.get("title", "")
        desc = top_risk.get("description", "")

        print_header(f"Key Finding: {title}", config)
        formatted_desc = format_inline_citations(desc, config)
        print(formatted_desc)

        if len(risks) > 1:
            print(c(f"\n  (+{len(risks) - 1} more findings, use /verbose to see all)", Colors.DIM, config))

            if config.verbose:
                for risk in risks[1:]:
                    r_title = risk.get("title", "")
                    r_desc = risk.get("description", "")
                    formatted_r_desc = format_inline_citations(r_desc, config)
                    print(f"\n  • {c(r_title, Colors.BOLD, config)}: {formatted_r_desc}")

    # ─────────────────────────────────────────────────────────────
    # 4. Sources
    # ─────────────────────────────────────────────────────────────
    if citations:
        print_header("Sources:", config)
        for cit in citations:
            cit_id = cit.get("id", "?")
            file_name = cit.get("file_name", "unknown")
            section_num = cit.get("section_number", "")
            section_header = cit.get("section_header", "")
            quote = cit.get("verbatim_quote", "")

            print(f"  {c(f'[{cit_id}]', Colors.CYAN + Colors.BOLD, config)} {file_name} §{section_num} {section_header}")

            if config.verbose and quote:
                truncated = quote[:150] + "..." if len(quote) > 150 else quote
                print(c(f'      > "{truncated}"', Colors.DIM, config))

    # ─────────────────────────────────────────────────────────────
    # 5. Follow-up & Review flags
    # ─────────────────────────────────────────────────────────────
    if suggested_followup:
        print()
        print_info(f"💡 Suggested: {suggested_followup}", config)

    if needs_human_review:
        print()
        print_warning(f"⚠ Human review recommended: {review_reason}", config)

    print_divider(config)
    print()


def print_chunks(state: dict[str, Any], config: CLIConfig) -> None:
    """Print retrieved document chunks."""
    chunks = state.get("retrieved_clauses", [])
    if not chunks:
        print_debug("Chunks", "None retrieved", config)
        return

    print_header(f"Retrieved Chunks ({len(chunks)}):", config)
    for i, chunk in enumerate(chunks, 1):
        file_name = chunk.get("file_name", "unknown")
        section = f"§{chunk.get('section_number', '')} {chunk.get('section_header', '')}".strip()
        score = chunk.get("score", chunk.get("fused_score", 0))
        text = chunk.get("text", "")[:200]

        print(f"\n  {c(f'[{i}]', Colors.CYAN, config)} {file_name} {section}")
        print(f"      Score: {c(f'{score:.3f}', Colors.YELLOW, config)}")
        if config.verbose:
            print(c(f'      "{text}..."', Colors.DIM, config))


def print_reflection_info(state: dict[str, Any], config: CLIConfig) -> None:
    """Print guardian/reflection metrics."""
    faithfulness = state.get("faithfulness_score", 0.0)
    relevancy = state.get("answer_relevancy_score", 0.0)
    correction_needed = state.get("correction_needed", False)
    correction_reason = state.get("correction_reason", "")

    print_header("Guardian Evaluation:", config)

    # Color-code scores
    def score_color(score: float) -> str:
        if score >= 0.8:
            return Colors.GREEN
        elif score >= 0.5:
            return Colors.YELLOW
        return Colors.RED

    print(f"  Faithfulness:      {c(f'{faithfulness:.2f}', score_color(faithfulness), config)}")
    print(f"  Answer Relevancy:  {c(f'{relevancy:.2f}', score_color(relevancy), config)}")

    if correction_needed:
        print(c(f"  ⚠ Correction needed: {correction_reason}", Colors.YELLOW, config))


def print_verbose_debug(state: dict[str, Any], config: CLIConfig) -> None:
    """Print all debug information in verbose mode."""
    print_header("Debug Info:", config)
    print_debug("Rewritten Query", state.get("rewritten_query", "N/A"), config)
    print_debug("Retrieval Confidence", f"{state.get('retrieval_confidence', 0.0):.2f}", config)
    print_debug("Warning Threshold", state.get("retrieval_warning_threshold", 0.35), config)
    print_debug("Current Doc Focus", state.get("current_doc_focus", "N/A"), config)
    print_debug("Needs Clarification", state.get("needs_clarification", False), config)

    if state.get("retrieval_warning"):
        print_warning(state.get("retrieval_warning"), config)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_query(query: str, config: CLIConfig) -> dict[str, Any]:
    """Execute the graph and return the final state."""
    app = build_graph()
    result = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "enable_reflection": config.enable_reflection,
        },
        config={"configurable": {"thread_id": config.thread_id}},
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Command Handling
# ─────────────────────────────────────────────────────────────────────────────

def print_help(config: CLIConfig) -> None:
    """Print available commands."""
    print_header("Available Commands:", config)
    commands = [
        ("/help", "Show this help message"),
        ("/clear", "Start new conversation (reset thread)"),
        ("/chunks", f"Toggle chunk display (currently: {config.show_chunks})"),
        ("/reflection", f"Toggle reflection details (currently: {config.show_reflection})"),
        ("/verbose", f"Toggle verbose mode (currently: {config.verbose})"),
        ("/thread [ID]", f"Change thread ID (currently: {config.thread_id})"),
        ("/status", "Show current configuration"),
        ("/exit, /quit", "Exit CLI"),
    ]
    for cmd, desc in commands:
        print(f"  {c(cmd, Colors.CYAN, config):20} {desc}")
    print()


def print_status(config: CLIConfig) -> None:
    """Print current configuration status."""
    guardian_config = load_guardian_config()

    print_header("Current Configuration:", config)
    print(f"  Thread ID:       {c(config.thread_id, Colors.CYAN, config)}")
    print(f"  Messages:        {config.conversation_count}")
    print(f"  Guardian Mode:   {guardian_config.mode}")
    print(f"  Reflection:      {c('ON' if config.enable_reflection else 'OFF', Colors.GREEN if config.enable_reflection else Colors.DIM, config)}")
    print(f"  Show Chunks:     {c('ON' if config.show_chunks else 'OFF', Colors.GREEN if config.show_chunks else Colors.DIM, config)}")
    print(f"  Verbose:         {c('ON' if config.verbose else 'OFF', Colors.GREEN if config.verbose else Colors.DIM, config)}")
    print()


def handle_command(command: str, config: CLIConfig) -> bool:
    """
    Handle slash commands.
    Returns True if REPL should continue, False to exit.
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    if cmd in ("/exit", "/quit", "/q"):
        print(c("\nGoodbye!\n", Colors.DIM, config))
        return False

    elif cmd == "/help":
        print_help(config)

    elif cmd == "/clear":
        import uuid
        config.thread_id = f"cli-{uuid.uuid4().hex[:8]}"
        config.conversation_count = 0
        print_info(f"Conversation cleared. New thread: {config.thread_id}", config)

    elif cmd == "/chunks":
        config.show_chunks = not config.show_chunks
        print_info(f"Chunk display: {'ON' if config.show_chunks else 'OFF'}", config)

    elif cmd == "/reflection":
        config.show_reflection = not config.show_reflection
        config.enable_reflection = config.show_reflection
        print_info(f"Reflection: {'ON' if config.show_reflection else 'OFF'}", config)

    elif cmd == "/verbose":
        config.verbose = not config.verbose
        print_info(f"Verbose mode: {'ON' if config.verbose else 'OFF'}", config)

    elif cmd == "/thread":
        if arg:
            config.thread_id = arg
            config.conversation_count = 0
            print_info(f"Thread changed to: {config.thread_id}", config)
        else:
            print_info(f"Current thread: {config.thread_id}", config)

    elif cmd == "/status":
        print_status(config)

    else:
        print_error(f"Unknown command: {cmd}. Type /help for available commands.", config)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main REPL
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(config: CLIConfig) -> None:
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║           Contract Analysis CLI                           ║
║           Multi-Agent RAG System                          ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(c(banner, Colors.CYAN, config))
    print(c("  Type your question or /help for commands.\n", Colors.DIM, config))


def repl(config: CLIConfig) -> None:
    """Main REPL loop."""
    print_banner(config)
    print_status(config)

    while True:
        try:
            # Prompt
            prompt = c("You: ", Colors.GREEN + Colors.BOLD, config)
            user_input = input(prompt).strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                if not handle_command(user_input, config):
                    break
                continue

            # Execute query
            config.conversation_count += 1
            print(c("\n  Analyzing...", Colors.DIM, config))

            try:
                state = run_query(user_input, config)

                # Print debug info if enabled
                if config.verbose:
                    print_verbose_debug(state, config)

                # Print chunks if enabled
                if config.show_chunks:
                    print_chunks(state, config)

                # Print reflection if enabled
                if config.show_reflection:
                    print_reflection_info(state, config)

                # Print main response
                format_response(state, config)

            except Exception as e:
                print_error(f"Query failed: {e}", config)
                if config.verbose:
                    import traceback
                    traceback.print_exc()

        except KeyboardInterrupt:
            print(c("\n\n  Use /exit to quit.\n", Colors.DIM, config))

        except EOFError:
            print(c("\n\nGoodbye!\n", Colors.DIM, config))
            break


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> CLIConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Contract Analysis CLI — Interactive conversation with the Multi-Agent RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_cli.py                      Basic conversation
  python main_cli.py -v                   Verbose mode (show all debug info)
  python main_cli.py -c -r                Show chunks and reflection scores
  python main_cli.py -t my-session        Use custom thread ID

In-session commands:
  /help       Show available commands
  /clear      Start new conversation
  /chunks     Toggle chunk display
  /reflection Toggle reflection details
  /verbose    Toggle verbose mode
  /exit       Exit CLI
""",
    )

    parser.add_argument(
        "-t", "--thread",
        default="cli-session-1",
        help="Thread ID for conversation memory (default: cli-session-1)",
    )
    parser.add_argument(
        "-c", "--chunks",
        action="store_true",
        help="Show retrieved document chunks with scores",
    )
    parser.add_argument(
        "-r", "--reflection",
        action="store_true",
        help="Enable reflection loop and show guardian scores",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all intermediate steps (rewritten query, confidence, etc.)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args()

    return CLIConfig(
        thread_id=args.thread,
        show_chunks=args.chunks,
        show_reflection=args.reflection,
        verbose=args.verbose,
        use_color=not args.no_color,
        enable_reflection=args.reflection,
    )


def main() -> None:
    """Entry point."""
    config = parse_args()

    # Suppress noisy library warnings so CLI output stays clean.
    # These include:
    # - Chroma relevance score range warnings
    # - Pydantic serializer warnings when dumping AuditorResponse
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*Relevance scores must be between 0 and 1.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Pydantic serializer warnings.*",
    )

    # Check if running in a non-interactive environment
    if not sys.stdin.isatty():
        print_error("This CLI requires an interactive terminal.", config)
        sys.exit(1)

    try:
        repl(config)
    except Exception as e:
        print_error(f"Fatal error: {e}", config)
        sys.exit(1)


if __name__ == "__main__":
    main()
