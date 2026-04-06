import re
import sys
import logging
from pathlib import Path

from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config


logger = logging.getLogger(__name__)


PROMPTS_LIBRARY_DIRECTORY = Path(__file__).parent / "prompts"
INSTRUCTIONS_FILENAME = "instructions.txt"
VOICE_FILENAME = "voice.txt"
GLOBAL_TOOL_GUIDANCE = """

## CURRENT INFO RULES
When the web_search tool is available and the user asks for current, local, or fast-changing information
such as weather, news, restaurants, opening hours, schedules, prices, sports, or live events, use it
instead of guessing from memory.
Use web_search for recommendations that depend on freshness or local availability.
If a Home Assistant MCP server is configured, use it for smart-home status checks and device control
instead of guessing.
""".strip()
GLOBAL_MEMORY_GUIDANCE = """

## MEMORY RULES
Use remember_fact for durable user facts, preferences, relationships, routines, recurring projects, or explicit
"remember this" requests.
Use recall_memory when prior context could help.
Use forget_memory when the user asks you to forget or correct saved information.
Do not store secrets, temporary chatter, or one-off details unless the user explicitly asks.
""".strip()


def _expand_prompt_includes(content: str) -> str:
    """Expand [<name>] placeholders with content from prompts library files.

    Args:
        content: The template content with [<name>] placeholders

    Returns:
        Expanded content with placeholders replaced by file contents

    """
    # Pattern to match [<name>] where name is a valid file stem (alphanumeric, underscores, hyphens)
    # pattern = re.compile(r'^\[([a-zA-Z0-9_-]+)\]$')
    # Allow slashes for subdirectories
    pattern = re.compile(r'^\[([a-zA-Z0-9/_-]+)\]$')

    lines = content.split('\n')
    expanded_lines = []

    for line in lines:
        stripped = line.strip()
        match = pattern.match(stripped)

        if match:
            # Extract the name from [<name>]
            template_name = match.group(1)
            template_file = PROMPTS_LIBRARY_DIRECTORY / f"{template_name}.txt"

            try:
                if template_file.exists():
                    template_content = template_file.read_text(encoding="utf-8").rstrip()
                    expanded_lines.append(template_content)
                    logger.debug("Expanded template: [%s]", template_name)
                else:
                    logger.warning("Template file not found: %s, keeping placeholder", template_file)
                    expanded_lines.append(line)
            except Exception as e:
                logger.warning("Failed to read template '%s': %s, keeping placeholder", template_name, e)
                expanded_lines.append(line)
        else:
            expanded_lines.append(line)

    return '\n'.join(expanded_lines)


def get_session_instructions(user_id: str | None = None) -> str:
    """Get session instructions, loading from REACHY_MINI_CUSTOM_PROFILE if set."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        logger.info(f"Loading default prompt from {PROMPTS_LIBRARY_DIRECTORY / 'default_prompt.txt'}")
        instructions_file = PROMPTS_LIBRARY_DIRECTORY / "default_prompt.txt"
    else:
        if config.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            logger.info(
                "Loading prompt from external profile '%s' (root=%s)",
                profile,
                config.PROFILES_DIRECTORY,
            )
        else:
            logger.info(f"Loading prompt from profile '{profile}'")
        instructions_file = config.PROFILES_DIRECTORY / profile / INSTRUCTIONS_FILENAME

    try:
        if instructions_file.exists():
            instructions = instructions_file.read_text(encoding="utf-8").strip()
            if instructions:
                # Expand [<name>] placeholders with content from prompts library
                expanded_instructions = _expand_prompt_includes(instructions)
                memory_block = ""
                try:
                    from reachy_mini_conversation_app.memory import MemoryStore, get_default_memory_user_id

                    memory_block = MemoryStore().format_for_prompt(
                        user_id=(user_id or get_default_memory_user_id()),
                        limit=config.REACHY_MINI_MEMORY_PROMPT_LIMIT,
                    )
                except Exception as e:
                    logger.warning("Failed to load stored memory into prompt: %s", e)

                parts = [expanded_instructions, GLOBAL_TOOL_GUIDANCE, GLOBAL_MEMORY_GUIDANCE]
                if memory_block:
                    parts.append(memory_block)
                return "\n\n".join(parts)
            logger.error(f"Profile '{profile}' has empty {INSTRUCTIONS_FILENAME}")
            sys.exit(1)
        logger.error(f"Profile {profile} has no {INSTRUCTIONS_FILENAME}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load instructions from profile '{profile}': {e}")
        sys.exit(1)


def get_session_voice(default: str = "cedar") -> str:
    """Resolve the voice to use for the session.

    If a custom profile is selected and contains a voice.txt, return its
    trimmed content; otherwise return the provided default ("cedar").
    """
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        return default
    try:
        voice_file = config.PROFILES_DIRECTORY / profile / VOICE_FILENAME
        if voice_file.exists():
            voice = voice_file.read_text(encoding="utf-8").strip()
            return voice or default
    except Exception:
        pass
    return default
