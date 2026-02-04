import argparse
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import requests
import yaml
from docx import Document  # type: ignore

# Load environment variables from .env file (if present)
load_dotenv()


# =============================================================================
# LLM Provider Abstraction
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, provider_config: dict[str, Any]):
        """Initialize the provider with its configuration.
        
        Args:
            provider_config: Provider-specific configuration dict (api_key, model, etc.)
        """
        self.config = provider_config
    
    def _resolve_api_key(self) -> str:
        """Resolve the API key from config or environment variable.
        
        Priority:
        1. Environment variable (if api_key_env_var is set and the env var exists)
        2. Direct api_key from config
        
        Returns:
            The resolved API key.
            
        Raises:
            ValueError: If no API key can be resolved.
        """
        # Check for environment variable override first
        env_var_name = self.config.get("api_key_env_var")
        if env_var_name:
            env_key = os.environ.get(env_var_name)
            if env_key:
                return env_key
        
        # Fall back to direct api_key in config
        api_key = self.config.get("api_key")
        if api_key:
            return api_key
        
        # No key found - provide helpful error message
        if env_var_name:
            raise ValueError(
                f"{self.name} provider requires an API key. "
                f"Set the '{env_var_name}' environment variable or provide 'api_key' in configuration."
            )
        else:
            raise ValueError(
                f"{self.name} provider requires 'api_key' in configuration, "
                f"or set 'api_key_env_var' to specify an environment variable name."
            )
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the display name of this provider."""
        pass
    
    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model being used."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM.
            
        Returns:
            The text response from the LLM.
        """
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, provider_config: dict[str, Any]):
        super().__init__(provider_config)
        # Import here to avoid global dependency
        from google import genai
        
        api_key = self._resolve_api_key()
        self._client = genai.Client(api_key=api_key)
        # Require explicit model selection - no default since this costs money
        self._model = provider_config.get("model")
        if not self._model:
            raise ValueError(
                "Gemini provider requires 'model' to be specified in configuration. "
                "Example models: gemini-2.0-flash (cheap), gemini-2.0-pro"
            )
    
    @property
    def name(self) -> str:
        return "Gemini"
    
    @property
    def model(self) -> str:
        return self._model
    
    def generate(self, prompt: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                        print(f"  Rate limited. Waiting {wait_time}s before retry...", file=sys.stderr)
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise
        return ""


class ClaudeProvider(LLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(self, provider_config: dict[str, Any]):
        super().__init__(provider_config)
        # Import here to avoid global dependency
        import anthropic
        
        api_key = self._resolve_api_key()
        self._client = anthropic.Anthropic(api_key=api_key)
        # Require explicit model selection - no default since this costs money
        self._model = provider_config.get("model")
        if not self._model:
            raise ValueError(
                "Claude provider requires 'model' to be specified in configuration. "
                "Example models: claude-sonnet-4-20250514 (balanced), claude-opus-4-20250514 (best)"
            )
        self._max_tokens = provider_config.get("max_tokens", 16384)
        # Store anthropic module for exception handling
        self._anthropic = anthropic
    
    @property
    def name(self) -> str:
        return "Claude"
    
    @property
    def model(self) -> str:
        return self._model
    
    def generate(self, prompt: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use streaming to handle large prompts that take >10 minutes
                result_text = ""
                with self._client.messages.stream(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                ) as stream:
                    for text in stream.text_stream:
                        result_text += text
                return result_text
            except self._anthropic.RateLimitError:
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                    print(f"  Rate limited. Waiting {wait_time}s before retry...", file=sys.stderr)
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = 30 * (attempt + 1)
                        print(f"  Rate limited. Waiting {wait_time}s before retry...", file=sys.stderr)
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise
        return ""


class ChatGPTProvider(LLMProvider):
    """OpenAI ChatGPT LLM provider."""
    
    def __init__(self, provider_config: dict[str, Any]):
        super().__init__(provider_config)
        # Import here to avoid global dependency
        import openai
        
        api_key = self._resolve_api_key()
        self._client = openai.OpenAI(api_key=api_key)
        # Require explicit model selection - no default since this costs money
        self._model = provider_config.get("model")
        if not self._model:
            raise ValueError(
                "ChatGPT provider requires 'model' to be specified in configuration. "
                "Example models: gpt-4o-mini (cheap), gpt-4o, gpt-5.2 (best)"
            )
        self._max_tokens = provider_config.get("max_tokens", 16384)
        # Store openai module for exception handling
        self._openai = openai
    
    @property
    def name(self) -> str:
        return "ChatGPT"
    
    @property
    def model(self) -> str:
        return self._model
    
    def generate(self, prompt: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use streaming to handle large prompts
                result_text = ""
                stream = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        result_text += chunk.choices[0].delta.content
                return result_text
            except self._openai.AuthenticationError as e:
                raise ValueError(
                    f"OpenAI authentication failed. Check your API key. Error: {e}"
                ) from e
            except self._openai.PermissionDeniedError as e:
                raise ValueError(
                    f"OpenAI permission denied. Your account may lack billing setup or access to model '{self._model}'. Error: {e}"
                ) from e
            except self._openai.NotFoundError as e:
                raise ValueError(
                    f"OpenAI model '{self._model}' not found. Check the model name. Error: {e}"
                ) from e
            except self._openai.RateLimitError as e:
                # Check if it's actually a quota/billing issue vs true rate limit
                error_str = str(e).lower()
                if "quota" in error_str or "billing" in error_str or "insufficient" in error_str:
                    raise ValueError(
                        f"OpenAI quota exceeded or billing issue. Check your OpenAI account. Error: {e}"
                    ) from e
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                    print(f"  Rate limited. Waiting {wait_time}s before retry...", file=sys.stderr)
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                error_str = str(e).lower()
                # Only retry on actual rate limits, not other errors
                if "429" in str(e) and "rate" in error_str and "quota" not in error_str:
                    if attempt < max_retries - 1:
                        wait_time = 30 * (attempt + 1)
                        print(f"  Rate limited. Waiting {wait_time}s before retry...", file=sys.stderr)
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise
        return ""


def create_llm_provider(provider_name: str, provider_config: dict[str, Any]) -> LLMProvider:
    """Factory function to create an LLM provider instance.
    
    Args:
        provider_name: Name of the provider ("gemini", "claude", or "chatgpt").
        provider_config: Provider-specific configuration dict.
        
    Returns:
        An initialized LLMProvider instance.
        
    Raises:
        ValueError: If provider_name is not recognized.
    """
    providers = {
        "gemini": GeminiProvider,
        "claude": ClaudeProvider,
        "chatgpt": ChatGPTProvider,
    }
    
    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        valid_providers = ", ".join(providers.keys())
        raise ValueError(f"Unknown LLM provider '{provider_name}'. Valid options: {valid_providers}")
    
    return provider_class(provider_config)


# =============================================================================
# Configuration and File Loading
# =============================================================================

def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_resume_text(resume_path: str | Path) -> str:
    path = Path(resume_path)
    if not path.exists():
        print(f"Resume file '{path}' not found. Continuing with empty resume text.", file=sys.stderr)
        return ""

    if path.suffix.lower() == ".docx":
        try:
            doc = Document(path)
            # Combine paragraphs with blank lines preserved
            return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
        except Exception as exc:  # noqa: BLE001
            print(f"Could not read DOCX resume '{path}': {exc}. Using empty resume text.", file=sys.stderr)
            return ""

    # Fallback to plain-text read
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read resume file '{path}': {exc}. Using empty resume text.", file=sys.stderr)
        return ""


def build_date_posted_value(days_ago: int) -> str:
    """Map days_ago to valid JSearch date_posted values: today, 3days, week, month.
    
    Returns a value broad enough to cover the requested range (we filter locally after).
    """
    if days_ago <= 0:
        return "today"
    if days_ago <= 3:
        return "3days"
    if days_ago <= 7:
        return "week"
    return "month"


def filter_jobs_by_date(jobs: list, days_ago: int) -> list:
    """Filter jobs to only those posted within the last `days_ago` days."""
    if days_ago < 0:
        return jobs
    
    # Calculate cutoff timestamp (start of day, days_ago days back)
    now = datetime.now()
    cutoff = datetime(now.year, now.month, now.day) - timedelta(days=days_ago)
    cutoff_ts = cutoff.timestamp()
    
    filtered = []
    for job in jobs:
        posted_ts = job.get("job_posted_at_timestamp")
        if posted_ts and posted_ts >= cutoff_ts:
            filtered.append(job)
    return filtered


def filter_jobs_by_exclusions(jobs: list, config: dict) -> list:
    """Pre-filter jobs locally based on exclusion rules before sending to LLM.
    
    This is a defense-in-depth measure that:
    1. Reduces API costs by not sending obviously-excluded jobs
    2. Ensures exclusions are enforced even if LLM misses them
    
    Args:
        jobs: List of job dictionaries from JSearch API
        config: Config dict containing exclusion lists
    
    Returns:
        Filtered list of jobs that pass exclusion checks
    """
    exclude_techs = [t.lower() for t in (config.get("exclude_technologies") or [])]
    exclude_roles = [r.lower() for r in (config.get("exclude_role_types") or [])]
    exclude_keywords = [k.lower() for k in (config.get("exclude_keywords") or [])]
    
    # If no exclusions configured, return all jobs
    if not exclude_techs and not exclude_roles and not exclude_keywords:
        return jobs
    
    filtered = []
    excluded_count = 0
    
    for job in jobs:
        title = (job.get("job_title") or "").lower()
        description = (job.get("job_description") or "").lower()
        
        # Check role type exclusions (in title)
        role_excluded = any(role in title for role in exclude_roles)
        if role_excluded:
            excluded_count += 1
            continue
        
        # Check technology exclusions (in title - strong signal)
        # We check title for things like "React Developer", "TypeScript Engineer"
        tech_in_title = any(tech in title for tech in exclude_techs)
        if tech_in_title:
            excluded_count += 1
            continue
        
        # Check keyword exclusions
        keyword_excluded = any(kw in description for kw in exclude_keywords)
        if keyword_excluded:
            excluded_count += 1
            continue
        
        filtered.append(job)
    
    if excluded_count > 0:
        print(f"  Pre-filtered {excluded_count} jobs based on exclusion rules", file=sys.stderr)
    
    return filtered


def get_bulk_jobs(config: dict, rapidapi_key: str, *, debug: bool = False) -> list:
    """Fetches remote job listings for each configured query."""
    print("Fetching jobs from JSearch...", file=sys.stderr)
    url = "https://jsearch.p.rapidapi.com/search"

    queries = config.get("queries") or []
    days_ago = int(config.get("date_posted_days_ago", 0))
    date_posted = build_date_posted_value(days_ago)
    remote_only = bool(config.get("remote_jobs_only", True))

    all_jobs = []
    debug_responses = []

    pages_per_query = int(config.get("jsearch_pages_per_query", 5))
    if pages_per_query < 1:
        raise ValueError("'jsearch_pages_per_query' must be >= 1")

    for i, q in enumerate(queries):
        querystring = {
            "query": q,
            "page": "1",
            "num_pages": str(pages_per_query),
            "date_posted": date_posted,
            "remote_jobs_only": str(remote_only).lower(),
        }

        try:
            response = requests.get(
                url,
                headers={
                    "X-RapidAPI-Key": rapidapi_key,
                    "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
                },
                params=querystring,
            )

            response_json = response.json()
            if debug:
                debug_responses.append({
                    "query": q,
                    "querystring": querystring,
                    "response": response_json,
                })

            data = response_json.get("data", [])
            all_jobs.extend(data)
            print(f"  Got {len(data)} jobs for '{q}'", file=sys.stderr)
            time.sleep(1)  # Rate limit safety
        except Exception as e:  # noqa: BLE001
            print(f"  Error fetching {q}: {e}", file=sys.stderr)

    # Save debug output if enabled
    if debug and debug_responses:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        debug_file = Path(f"jsearch-debug-{timestamp}.json")
        debug_file.write_text(json.dumps(debug_responses, indent=2), encoding="utf-8")
        print(f"  Debug: saved raw API responses to {debug_file}", file=sys.stderr)

    # Filter locally to the exact day range requested
    filtered_jobs = filter_jobs_by_date(all_jobs, days_ago)
    if len(filtered_jobs) < len(all_jobs):
        print(f"  Filtered to {len(filtered_jobs)} jobs within last {days_ago} days", file=sys.stderr)
    
    return filtered_jobs


def build_exclusion_rules(config: dict) -> str:
    """Build dynamic exclusion rules section from config lists.
    
    Generates clear, explicit instructions for the LLM based on user-configured
    exclusion lists. These rules apply regardless of what's on the resume.
    """
    rules = []
    
    # Excluded technologies
    exclude_techs = config.get("exclude_technologies") or []
    if exclude_techs:
        tech_list = ", ".join(exclude_techs)
        rules.append(f"""EXCLUDED TECHNOLOGIES (reject even if on my resume):
  {tech_list}
  
  If a job REQUIRES any of these technologies, EXCLUDE IT immediately.
  - Job title mentions these → EXCLUDE
  - Requirements list these → EXCLUDE  
  - Description says "strong [tech]" or "[tech] required" → EXCLUDE
  - "N+ years of [tech]" for these → EXCLUDE""")
    
    # Excluded role types
    exclude_roles = config.get("exclude_role_types") or []
    if exclude_roles:
        role_list = ", ".join(f'"{r}"' for r in exclude_roles)
        rules.append(f"""EXCLUDED ROLE TYPES (reject based on job title):
  {role_list}
  
  If the job title contains ANY of these terms (case-insensitive), EXCLUDE IT.
  Examples: "Full-Stack Engineer" → EXCLUDE, "Senior Frontend Developer" → EXCLUDE""")
    
    # Excluded keywords
    exclude_keywords = config.get("exclude_keywords") or []
    if exclude_keywords:
        kw_list = ", ".join(f'"{k}"' for k in exclude_keywords)
        rules.append(f"""EXCLUDED KEYWORDS (reject if found in description):
  {kw_list}
  
  If the job description contains ANY of these phrases, EXCLUDE IT.""")
    
    if not rules:
        return "No additional exclusion rules configured."
    
    return "\n\n".join(rules)


def build_prompt(jobs: list, config: dict, resume_text: str, *, markdown: bool = False) -> str:
    """Build the analysis prompt for LLM providers."""
    jobs_text = ""
    for i, job in enumerate(jobs):
        jobs_text += f"\n--- JOB ID {i} ---\n"
        jobs_text += f"Title: {job.get('job_title')}\n"
        jobs_text += f"Link: {job.get('job_apply_link')}\n"
        jobs_text += f"Description: {job.get('job_description')}\n"

    # Build dynamic exclusion rules from config
    exclusion_rules = build_exclusion_rules(config)
    
    base_prompt = (config.get("prompt") or "").replace("{job_count}", str(len(jobs))).replace("{exclusion_rules}", exclusion_rules)

    resume_block = resume_text if resume_text else "[No resume text provided]"

    if markdown:
        output_instruction = "Return the results in markdown format."
    else:
        output_instruction = "Return the results in json format."

    # Double the prompt for increased accuracy (sends instructions twice)
    doubled_prompt = f"{base_prompt}\n\n{base_prompt}"

    prompt = f"""
{doubled_prompt}

{output_instruction}

=== JOB LISTINGS TO EVALUATE ===
{jobs_text}
=== END OF JOB LISTINGS ===

=== REFERENCE: MY RESUME (use ONLY to assess fit - NOT a job listing) ===
{resume_block}
=== END OF RESUME ===

CRITICAL VALIDATION:
- Every result MUST have a valid HTTPS application URL
- If a "job" has no link or says "based on my work experience", EXCLUDE IT
- Only return actual job postings from external companies
""".strip()

    return prompt


def analyze_jobs(
    jobs: list, provider: LLMProvider, config: dict, resume_text: str, *, markdown: bool = False
) -> str:
    """Sends all jobs to the configured LLM for analysis.
    
    Args:
        jobs: List of job dictionaries to analyze.
        provider: The LLM provider to use for analysis.
        config: Application configuration dict.
        resume_text: The user's resume text.
        markdown: Whether to return markdown (True) or JSON (False).
        
    Returns:
        The LLM's analysis response.
    """
    print(f"Sending {len(jobs)} jobs to {provider.name} ({provider.model})", file=sys.stderr)
    
    prompt = build_prompt(jobs, config, resume_text, markdown=markdown)
    return provider.generate(prompt)


def clean_duplicate_output(analysis: str, *, markdown: bool = False) -> str:
    """Remove duplicate format sections when LLM outputs both markdown and JSON.
    
    Sometimes Gemini outputs both formats even when only one is requested.
    This extracts just the requested format.
    
    Args:
        analysis: The raw analysis text from the LLM.
        markdown: Whether markdown (True) or JSON (False) was requested.
    
    Returns:
        Cleaned output containing only the requested format.
    """
    text = analysis.strip()
    
    if markdown:
        # If JSON array appears in markdown output, strip it
        has_code_fence = '```' in text
        has_markdown_content = any(
            text.startswith(prefix) for prefix in ('*', '-', '#', '##', '###')
        )
        
        if has_code_fence and has_markdown_content:
            # Keep only the markdown portion (before code fences)
            parts = text.split('```')
            markdown_part = parts[0].strip()
            
            # If the markdown part has actual content, use it
            if markdown_part and len(markdown_part) > 50:
                return markdown_part
            
            # Otherwise check if markdown comes after the code block
            if len(parts) > 2:
                after_code = parts[2].strip() if len(parts) > 2 else ''
                if after_code and any(
                    after_code.startswith(p) for p in ('*', '-', '#', '##', '###')
                ):
                    return after_code
    else:
        # JSON mode: extract just the JSON array
        
        # If it already starts with a valid JSON array, return as-is
        if text.startswith('['):
            return text
        
        # Try to find and extract JSON array from mixed output
        start = text.find('[')
        end = text.rfind(']')
        
        if start >= 0 and end > start:
            potential_json = text[start:end + 1]
            
            # Validate it's actually parseable JSON before returning
            try:
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass  # Fall through to return original
        
        # Check for JSON in code fences
        if '```' in text:
            # Extract content between fences
            fence_start = text.find('```')
            fence_content_start = text.find('\n', fence_start) + 1
            fence_end = text.find('```', fence_content_start)
            
            if fence_content_start > 0 and fence_end > fence_content_start:
                fenced_content = text[fence_content_start:fence_end].strip()
                if fenced_content.startswith('['):
                    try:
                        json.loads(fenced_content)
                        return fenced_content
                    except json.JSONDecodeError:
                        pass
    
    return analysis


def validate_results(analysis: str, *, markdown: bool = False) -> str:
    """Filter out invalid job matches like resume content mistaken for jobs.
    
    Args:
        analysis: The raw analysis text from the LLM.
        markdown: Whether the output is markdown (True) or JSON (False).
    
    Returns:
        Cleaned analysis with invalid entries removed.
    """
    # Markers that indicate resume content or invalid entries
    invalid_markers = [
        'based on my work experience',
        'based on my resume',
        'this position is based on',
        'from my background',
        'my professional experience',
        'link:  (',  # Empty link with parenthetical
        'link: (',
        'link: n/a',
        'link: none',
        'link: not available',
        'no link provided',
        'no application link',
    ]
    
    if markdown:
        # For markdown, remove sections with invalid content
        lines = analysis.split('\n')
        filtered_lines = []
        skip_section = False
        current_section_lines: list[str] = []
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if this is a section header (new job entry)
            is_section_start = (
                line.startswith('### ') or 
                line.startswith('## ') or
                line.startswith('---')
            )
            
            if is_section_start:
                # Flush previous section if valid
                if current_section_lines and not skip_section:
                    filtered_lines.extend(current_section_lines)
                
                # Start new section
                current_section_lines = [line]
                skip_section = False
            else:
                current_section_lines.append(line)
                
                # Check for invalid markers in this line
                if any(marker in line_lower for marker in invalid_markers):
                    skip_section = True
        
        # Flush final section
        if current_section_lines and not skip_section:
            filtered_lines.extend(current_section_lines)
        
        return '\n'.join(filtered_lines)
    
    else:
        # For JSON, parse and filter
        try:
            text = analysis.strip()
            
            # Strip markdown code fences if present (handle ```json, ```, etc.)
            if text.startswith('```'):
                # Remove opening fence (with optional language tag)
                first_newline = text.find('\n')
                if first_newline != -1:
                    text = text[first_newline + 1:]
                # Remove closing fence
                if '```' in text:
                    text = text.rsplit('```', 1)[0]
                text = text.strip()
            
            jobs = json.loads(text)
            
            if not isinstance(jobs, list):
                return analysis  # Unexpected format, return as-is
            
            valid_jobs = []
            for job in jobs:
                link = job.get('link', '') or ''
                # Check for title/position (prompt asks for "position" but also accept "title" for compatibility)
                title = job.get('position', '') or job.get('title', '') or job.get('job_title', '') or ''
                
                # Must have a valid HTTP(S) link
                if not (link.startswith('http://') or link.startswith('https://')):
                    continue
                
                # Check for invalid markers in any field
                job_text = json.dumps(job).lower()
                if any(marker in job_text for marker in invalid_markers):
                    continue
                
                # Skip entries with empty or placeholder titles
                if not title or title.lower() in ('n/a', 'none', 'unknown', ''):
                    continue
                
                valid_jobs.append(job)
            
            return json.dumps(valid_jobs, indent=2)
        
        except (json.JSONDecodeError, TypeError, KeyError):
            return analysis  # Return as-is if parsing fails


def main():
    parser = argparse.ArgumentParser(description="Fetch and analyze remote jobs.")
    parser.add_argument(
        "-c",
        "--config",
        default="find-jobs.yaml",
        help="Path to YAML config file (required)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        help="Override resume file path (optional). If provided, YAML resume_file is ignored.",
    )
    parser.add_argument(
        "-o",
        "--output-to-file",
        action="store_true",
        help="Write analysis to a timestamped filename. Otherwise printed to stdout.",
    )
    parser.add_argument(
        "-d",
        "--directory",
        help="Output directory for the file (used with -o). Defaults to current directory.",
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Name token used in default output filename (find-jobs-{name}-YYYYMMDD-HHMMSS.{json|md}).",
    )
    parser.add_argument(
        "-f",
        "--from-days",
        type=int,
        help="Override 'date_posted_days_ago'. Any value supported (filtered locally).",
    )
    parser.add_argument(
        "-m",
        "--markdown",
        action="store_true",
        help="Output results in markdown format instead of JSON.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save raw JSearch API responses to debug files.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    # Resolve RapidAPI key: environment variable overrides config file
    rapidapi_key = None
    rapidapi_env_var = config.get("rapidapi_key_env_var")
    if rapidapi_env_var:
        rapidapi_key = os.environ.get(rapidapi_env_var)
    if not rapidapi_key:
        rapidapi_key = config.get("rapidapi_key")
    if not rapidapi_key:
        env_hint = f" Set '{rapidapi_env_var}' environment variable or" if rapidapi_env_var else " "
        raise ValueError(f"RapidAPI key required.{env_hint} set 'rapidapi_key' in config file.")

    # Determine which LLM provider to use
    llm_provider_name = config.get("llm_provider", "gemini").lower()
    
    # Get provider-specific configuration from llm_providers section
    llm_providers_config = config.get("llm_providers", {})
    provider_config = llm_providers_config.get(llm_provider_name, {})
    
    if not provider_config:
        raise ValueError(
            f"No configuration found for LLM provider '{llm_provider_name}'. "
            f"Add 'llm_providers.{llm_provider_name}' section to your config file."
        )

    # Create the LLM provider instance
    llm_provider = create_llm_provider(llm_provider_name, provider_config)
    print(f"Using LLM provider: {llm_provider.name} ({llm_provider.model})", file=sys.stderr)

    queries = config.get("queries")
    prompt = config.get("prompt")

    if not queries:
        raise ValueError("Config key 'queries' is required and must be non-empty.")
    if not prompt:
        raise ValueError("Config key 'prompt' is required and must be non-empty.")

    resume_path = args.resume or config.get("resume_file")
    if not resume_path:
        raise ValueError("Resume path is required. Provide --resume or set 'resume_file' in config.")

    # Override date window if provided via CLI
    if args.from_days is not None:
        if args.from_days < 0:
            raise ValueError("--from-days must be >= 0")
        config["date_posted_days_ago"] = args.from_days

    resume_text = load_resume_text(resume_path)

    print("Starting downloading jobs", file=sys.stderr)
    jobs = get_bulk_jobs(config, rapidapi_key, debug=args.debug)
    print(f"{len(jobs)} jobs found.", file=sys.stderr)
    
    # Apply local exclusion filters before sending to LLM
    jobs = filter_jobs_by_exclusions(jobs, config)
    print(f"{len(jobs)} jobs remaining after exclusion filters.", file=sys.stderr)
    
    if jobs:
        analysis = analyze_jobs(jobs, llm_provider, config, resume_text, markdown=args.markdown)
        
        # Debug: show raw response from LLM
        print(f"DEBUG: Raw {llm_provider.name} response length: {len(analysis)} chars", file=sys.stderr)
        print(f"DEBUG: Raw {llm_provider.name} response preview (first 500 chars): {analysis[:500]}", file=sys.stderr)
        
        # Post-process: clean up mixed format output, then validate entries
        analysis_before_clean = analysis
        analysis = clean_duplicate_output(analysis, markdown=args.markdown)
        print(f"DEBUG: After clean_duplicate_output length: {len(analysis)} chars", file=sys.stderr)
        
        analysis_before_validate = analysis
        analysis = validate_results(analysis, markdown=args.markdown)
        print(f"DEBUG: After validate_results length: {len(analysis)} chars", file=sys.stderr)

        # Count actual results returned by Gemini
        selected_count = 0
        if not args.markdown:
            try:
                selected_count = len(json.loads(analysis))
            except (json.JSONDecodeError, TypeError):
                selected_count = 0  # Can't parse, unknown count
        else:
            # For markdown, count section headers as a rough estimate
            selected_count = analysis.count('\n## ') + analysis.count('\n### ')

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        ext = ".md" if args.markdown else ".json"
        if args.output_to_file:
            if args.name:
                filename = f"find-jobs-{args.name}-{timestamp}{ext}"
            else:
                filename = f"find-jobs-{timestamp}{ext}"
            if args.directory:
                output_path = Path(args.directory) / filename
            else:
                output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(analysis, encoding="utf-8")
            print(f"{selected_count} jobs written to {output_path} on {timestamp}", file=sys.stderr)
        else:
            # Emit results to stdout when not writing to file.
            print(analysis)

        print(f"{selected_count} of {len(jobs)} jobs selected by {llm_provider.name}.", file=sys.stderr)
    else:
        print("No jobs found for the configured filters.", file=sys.stderr)


if __name__ == "__main__":
    main()