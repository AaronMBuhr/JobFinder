import argparse
import hashlib
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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


# =============================================================================
# Resume Caching and Compression
# =============================================================================

RESUME_CACHE_FILE = Path("resume.cached")

RESUME_COMPRESSION_PROMPT = """You are a resume compressor for job-matching.

INPUT: A resume in plain text or Markdown.
OUTPUT: A compact plain-text "LLM Matching Payload" suitable to compare against job descriptions.

Hard constraints:
- Output must be <= 2400 characters.
- Output must be plain text only (no code fences).
- Do NOT invent skills, tools, or experience not explicitly present in the resume.
- Preserve important duty wording and all technologies mentioned in duties.

What to KEEP (in this exact structure):

NAME | Location
Headline: <short role label>

Skills: <single line, comma-separated; include languages, cloud, infra, databases, core backend keywords found in resume>

Experience
<Most recent company> — <title> (<dates>, <location/remote>)
• <3–6 duty bullets: copy or lightly compress the resume's real duties; MUST retain architecture/scale/reliability details and tech stack>

<Second most recent company> — <title> (<dates>, <location/remote>)
• <3–5 duty bullets: same rules>

Continue through last minimum 10 years of jobs 

Earlier roles (high level)
• <Company> — <Title>: <ONE sentence duty/impact summary copied/condensed from resume>
• <Company> — <Title>: <ONE sentence duty/impact summary copied/condensed from resume>

Education: <one line, degree + school>

What to REMOVE:
- phone, email, LinkedIn, GitHub URLs
- long summaries, soft-skill fluff, repeated headings
- projects unless they contain unique technologies not shown elsewhere
- any duty bullets that are purely generic (e.g., "collaborated with team") unless they include unique tech or measurable impact

Normalization rules:
- Collapse whitespace; keep the bullet marker "•"
- Prefer concrete duties that indicate fit: backend services, infra, reliability, distributed systems, data pipelines, CI/CD, production ops

Now produce the compact payload from the resume below:

"""

RESUME_VERIFICATION_PROMPT = """You are a resume verification assistant.

Your task is to verify that a condensed resume accurately represents the original full resume.

IMPORTANT RULES:
- The condensed version should capture all key skills, technologies, job titles, companies, and dates.
- It should NOT invent or add anything not in the original.
- It should NOT omit critical technical skills, job roles, or significant achievements.
- Minor wording differences are acceptable if the meaning is preserved.
- The condensed version must be <= 2400 characters.

Below is the ORIGINAL FULL RESUME:
---
{full_resume}
---

Below is the CONDENSED VERSION:
---
{condensed_resume}
---

If the condensed version is an accurate representation of the original:
- Reply with EXACTLY: ACCURATE

If the condensed version has issues (missing critical info, invented content, or significant misrepresentation):
- Reply with: NEEDS_CORRECTION
- Then provide a corrected condensed version on the next line (plain text, no code fences, <= 2400 chars)

Your response:
"""


def compute_resume_checksum(resume_text: str) -> str:
    """Compute SHA256 checksum of the resume text.
    
    Args:
        resume_text: The full resume text content.
        
    Returns:
        Hex string of the SHA256 hash.
    """
    return hashlib.sha256(resume_text.encode("utf-8")).hexdigest()


def load_resume_cache(cache_path: Path = RESUME_CACHE_FILE) -> dict | None:
    """Load the cached compressed resume if it exists.
    
    Args:
        cache_path: Path to the cache file.
        
    Returns:
        Dict with 'checksum', 'payload', 'timestamp' or None if not found.
    """
    if not cache_path.exists():
        return None
    
    try:
        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
        # Validate required fields
        if all(k in cache_data for k in ("checksum", "payload", "timestamp")):
            return cache_data
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: Could not read resume cache: {e}", file=sys.stderr)
    
    return None


def save_resume_cache(
    checksum: str, 
    payload: str, 
    cache_path: Path = RESUME_CACHE_FILE
) -> None:
    """Save the compressed resume to cache.
    
    Args:
        checksum: SHA256 checksum of the original resume.
        payload: The compressed resume text.
        cache_path: Path to the cache file.
    """
    cache_data = {
        "checksum": checksum,
        "payload": payload,
        "timestamp": datetime.now().isoformat(),
    }
    cache_path.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
    print(f"  Resume cache saved to {cache_path}", file=sys.stderr)


def compress_resume(resume_text: str, provider: "LLMProvider") -> str:
    """Compress the resume using the LLM for job-matching.
    
    This is a two-stage process:
    1. Initial compression to create a condensed version
    2. Verification that the condensed version accurately represents the original
    
    Args:
        resume_text: The full resume text.
        provider: The LLM provider to use.
        
    Returns:
        Compressed resume payload (<= 2400 chars).
    """
    print(f"  Stage 1: Compressing resume using {provider.name} ({provider.model})...", file=sys.stderr)
    
    prompt = RESUME_COMPRESSION_PROMPT + resume_text
    compressed = provider.generate(prompt)
    
    # Strip any code fences the LLM might have added
    compressed = _strip_code_fences(compressed)
    
    # Warn if over limit
    if len(compressed) > 2400:
        print(f"  Warning: Initial compressed resume is {len(compressed)} chars (target: 2400)", file=sys.stderr)
    else:
        print(f"  Initial compressed resume: {len(compressed)} chars", file=sys.stderr)
    
    # Stage 2: Verify the compressed resume
    compressed = verify_compressed_resume(resume_text, compressed, provider)
    
    return compressed


def _strip_code_fences(text: str) -> str:
    """Strip code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    return text


def verify_compressed_resume(
    full_resume: str, 
    compressed_resume: str, 
    provider: "LLMProvider"
) -> str:
    """Verify and potentially correct the compressed resume.
    
    Stage 2 of the compression process: asks the LLM to verify that the 
    condensed version accurately represents the original full resume.
    
    Args:
        full_resume: The original full resume text.
        compressed_resume: The initially compressed resume.
        provider: The LLM provider to use.
        
    Returns:
        The verified (or corrected) compressed resume.
    """
    print(f"  Stage 2: Verifying compressed resume accuracy...", file=sys.stderr)
    
    prompt = RESUME_VERIFICATION_PROMPT.format(
        full_resume=full_resume,
        condensed_resume=compressed_resume
    )
    
    response = provider.generate(prompt)
    response = response.strip()
    
    # Check if the response indicates accuracy
    if response.upper().startswith("ACCURATE"):
        print(f"  Verification: Condensed resume is accurate", file=sys.stderr)
        return compressed_resume
    
    # The LLM indicated corrections are needed
    if response.upper().startswith("NEEDS_CORRECTION"):
        print(f"  Verification: Corrections needed, applying...", file=sys.stderr)
        
        # Extract the corrected version (everything after "NEEDS_CORRECTION" line)
        lines = response.split("\n", 1)
        if len(lines) > 1:
            corrected = lines[1].strip()
            corrected = _strip_code_fences(corrected)
            
            if len(corrected) > 2400:
                print(f"  Warning: Corrected resume is {len(corrected)} chars (target: 2400)", file=sys.stderr)
            else:
                print(f"  Corrected compressed resume: {len(corrected)} chars", file=sys.stderr)
            
            return corrected
        else:
            print(f"  Warning: LLM indicated corrections needed but provided none, using original", file=sys.stderr)
            return compressed_resume
    
    # Unexpected response format - log and return original
    print(f"  Warning: Unexpected verification response format, using original compressed version", file=sys.stderr)
    print(f"  Response preview: {response[:200]}...", file=sys.stderr)
    return compressed_resume


def get_resume_for_matching(
    resume_path: str | Path, 
    provider: "LLMProvider",
    *,
    force_refresh: bool = False,
) -> tuple[str, str]:
    """Get the compressed resume for job matching, using cache if valid.
    
    This function:
    1. Loads the full resume text
    2. Computes its checksum
    3. Checks if a valid cache exists (matching checksum)
    4. If cache valid, returns cached payload
    5. If cache invalid/missing, compresses via LLM and caches result
    
    Args:
        resume_path: Path to the resume file.
        provider: LLM provider for compression.
        force_refresh: If True, bypass cache and recompress.
        
    Returns:
        Tuple of (compressed_resume, full_resume_text).
        The compressed resume is used for job matching,
        the full text is available if needed.
    """
    # Load full resume
    full_resume = load_resume_text(resume_path)
    if not full_resume:
        return "", ""
    
    # Compute checksum
    checksum = compute_resume_checksum(full_resume)
    print(f"Resume checksum: {checksum[:16]}...", file=sys.stderr)
    
    # Check cache
    if not force_refresh:
        cache = load_resume_cache()
        if cache and cache.get("checksum") == checksum:
            print(f"  Using cached compressed resume (from {cache.get('timestamp', 'unknown')})", file=sys.stderr)
            return cache["payload"], full_resume
        elif cache:
            print("  Resume changed - recompressing...", file=sys.stderr)
        else:
            print("  No cached resume found - compressing...", file=sys.stderr)
    else:
        print("  Force refresh requested - recompressing...", file=sys.stderr)
    
    # Compress and cache
    compressed = compress_resume(full_resume, provider)
    save_resume_cache(checksum, compressed)
    
    return compressed, full_resume


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


def filter_blocked_companies(jobs: list, config: dict) -> list:
    """Filter out jobs posted by known aggregator/repost sites.
    
    Aggregator sites scrape legitimate job postings from real companies
    and repost them under their own name to collect applicant data.
    Common offenders include Flexionis, Taskium, Jobgether, Lensa, etc.
    """
    blocked = [c.lower().strip() for c in (config.get("blocked_companies") or [])]
    if not blocked:
        return jobs
    
    filtered = []
    blocked_count = 0
    for job in jobs:
        employer = (job.get("employer_name") or "").lower().strip()
        if any(b in employer for b in blocked):
            blocked_count += 1
            continue
        filtered.append(job)
    
    if blocked_count:
        print(f"  Filtered {blocked_count} jobs from blocked companies (aggregators)", file=sys.stderr)
    return filtered


def filter_by_country(jobs: list, config: dict) -> list:
    """Filter jobs to only include those in configured target countries.
    
    If target_countries is not set or empty, all jobs pass through.
    Jobs with no country data are kept (benefit of the doubt).
    """
    target_countries = [c.lower().strip() for c in (config.get("target_countries") or [])]
    if not target_countries:
        return jobs
    
    filtered = []
    excluded_count = 0
    for job in jobs:
        country = (job.get("job_country") or "").lower().strip()
        if not country or country in target_countries:
            filtered.append(job)
        else:
            excluded_count += 1
    
    if excluded_count:
        countries_str = ", ".join(c.upper() for c in target_countries)
        print(f"  Filtered {excluded_count} jobs outside target countries ({countries_str})", file=sys.stderr)
    return filtered


def filter_blocked_link_domains(jobs: list, config: dict) -> list:
    """Filter out jobs whose apply links point to known aggregator domains."""
    blocked_domains = [d.lower().strip() for d in (config.get("blocked_link_domains") or [])]
    if not blocked_domains:
        return jobs
    
    filtered = []
    blocked_count = 0
    for job in jobs:
        link = (job.get("job_apply_link") or "").lower()
        is_blocked = False
        if link:
            try:
                domain = urlparse(link).netloc
                is_blocked = any(bd in domain for bd in blocked_domains)
            except Exception:
                pass
        
        if is_blocked:
            blocked_count += 1
        else:
            filtered.append(job)
    
    if blocked_count:
        print(f"  Filtered {blocked_count} jobs with blocked link domains", file=sys.stderr)
    return filtered


def filter_staffing_agencies(jobs: list, config: dict) -> list:
    """Filter out jobs from staffing/recruiting agencies.
    
    Uses two methods:
    1. Explicit list of agency names from config (staffing_agency_names)
    2. Heuristic detection based on description patterns
    """
    if not config.get("filter_staffing_agencies", False):
        return jobs
    
    agency_names = [n.lower().strip() for n in (config.get("staffing_agency_names") or [])]
    
    agency_description_patterns = [
        "on behalf of our client",
        "our client is seeking",
        "our client is looking",
        "w2/c2c",
        "w2 or c2c",
        "c2c/w2",
        "corp-to-corp",
        "corp to corp",
        "contract to hire",
        "contract-to-hire",
        "staffing agency",
        "recruiting agency",
        "talent acquisition firm",
    ]
    
    filtered = []
    excluded_count = 0
    
    for job in jobs:
        employer = (job.get("employer_name") or "").lower().strip()
        description = (job.get("job_description") or "").lower()
        
        if any(name in employer for name in agency_names):
            excluded_count += 1
            continue
        
        if any(pattern in description for pattern in agency_description_patterns):
            excluded_count += 1
            continue
        
        filtered.append(job)
    
    if excluded_count:
        print(f"  Filtered {excluded_count} jobs from staffing agencies", file=sys.stderr)
    return filtered


def deduplicate_jobs(jobs: list) -> list:
    """Remove duplicate job listings based on job_id.
    
    When running multiple search queries, the same job can appear in
    results from different queries. This keeps only the first occurrence.
    """
    seen_ids: set[str] = set()
    unique = []
    for job in jobs:
        job_id = job.get("job_id")
        if job_id and job_id in seen_ids:
            continue
        if job_id:
            seen_ids.add(job_id)
        unique.append(job)
    
    dedup_count = len(jobs) - len(unique)
    if dedup_count:
        print(f"  Removed {dedup_count} duplicate job listings", file=sys.stderr)
    return unique


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
    exclusion lists. These are defense-in-depth: most are also enforced as
    pre-filters, but the LLM should catch anything that slips through.
    """
    rules = []
    
    # Blocked companies (aggregators/repost sites)
    blocked_companies = config.get("blocked_companies") or []
    if blocked_companies:
        company_list = ", ".join(blocked_companies)
        rules.append(f"""BLOCKED COMPANIES (known aggregator/repost sites — REJECT immediately):
  {company_list}
  
  These are resume-farming sites that scrape real job postings and repost them.
  If the employer name matches any of these, EXCLUDE the job.
  Also EXCLUDE any job where the employer appears to be a generic aggregator but
  the description clearly describes a role at a different well-known company
  (e.g., employer is "Flexionis" but description says "at Netflix").""")
    
    # Target countries
    target_countries = config.get("target_countries") or []
    if target_countries:
        country_list = ", ".join(target_countries)
        rules.append(f"""TARGET COUNTRIES (reject jobs outside these):
  {country_list}
  
  Only include jobs located in or remote-eligible for these countries.
  If a job is tagged with a country NOT in this list, EXCLUDE it.""")
    
    # Staffing agencies
    if config.get("filter_staffing_agencies", False):
        agency_names = config.get("staffing_agency_names") or []
        agency_str = ", ".join(agency_names) if agency_names else "(auto-detected)"
        rules.append(f"""STAFFING AGENCIES (reject unless role details are exceptionally clear):
  Known agencies: {agency_str}
  
  Reject postings from staffing firms, recruiting agencies, or talent marketplaces.
  Indicators: "on behalf of our client", "W2/C2C", "contract-to-hire", vague role
  descriptions with no named end client.""")
    
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


def build_scoring_guidance() -> str:
    """Build scoring calibration instructions for the LLM.
    
    These are generic quality-check instructions that apply regardless of the
    user's specific profile. They address common failure modes in job-matching:
    surface-level keyword matching, aggregator reposts, and domain mismatches.
    """
    return """
============================================================
SCORING CALIBRATION — READ BEFORE SCORING ANY JOB
============================================================

Your scoring must go BEYOND surface-level keyword overlap. Apply these checks:

1. AGGREGATOR / REPOST DETECTION:
   If the employer name looks like a generic staffing or aggregator company but the
   job description clearly describes a role at a DIFFERENT well-known company,
   this is a scraped repost. REJECT it (score 0). Signs of reposts:
   - Employer name doesn't match the company described in the body
   - Apply link domain doesn't match the employer
   - Description says "About [Famous Company]" but employer is something else

2. DOMAIN EXPERTISE FIT (most important calibration):
   Do NOT give high scores based on programming language overlap alone.
   The DOMAIN and CONTEXT of the work must match the candidate's background.
   Examples of BAD matches despite keyword overlap:
   - "Python" in QA automation ≠ "Python" in backend architecture
   - "C++" in embedded/RTOS ≠ "C++" in distributed systems
   - "Systems Engineer" in aerospace/defense ≠ "Systems Engineer" in software
   - "Data Engineer" doing ETL scripts ≠ "Data Platform Engineer" building infra
   Ask: "Would this candidate's actual work history prepare them for THIS role's
   day-to-day responsibilities?" If not, score LOW.

3. UNREALISTIC REQUIREMENTS CHECK:
   If a job requires specific domain expertise, certifications, or clearances that
   are NOT evident in the candidate's resume, score it LOW regardless of tool overlap.
   Examples: security clearance, specific industry certifications, niche domain
   knowledge (aerospace, biotech, quant finance) that doesn't appear in the resume.

4. SENIORITY ALIGNMENT:
   A Staff/Principal role should not score high for a mid-level candidate, and vice
   versa. Check years-of-experience requirements against the resume timeline.

5. SCORE MEANING:
   95-100: Near-perfect fit — skills, domain, seniority, and tech stack all align
   90-94:  Strong fit — most requirements match with minor gaps
   85-89:  Good fit — solid overlap but some requirements are a stretch
   Below 85: Do not include in results
"""


def extract_whitelisted_job_fields(job: dict) -> dict:
    """Extract only the whitelisted fields from a job for LLM analysis.
    
    This reduces the amount of data sent to the LLM, lowering costs while
    preserving the information needed for job matching and quality checks.
    """
    highlights = job.get("job_highlights") or {}
    
    return {
        "job_id": job.get("job_id"),
        "job_title": job.get("job_title"),
        "employer_name": job.get("employer_name"),
        "job_publisher": job.get("job_publisher"),
        "job_employment_type": job.get("job_employment_type"),
        "job_apply_link": job.get("job_apply_link"),
        "apply_options": job.get("apply_options"),
        "job_description": job.get("job_description"),
        "job_is_remote": job.get("job_is_remote"),
        "qualifications": highlights.get("Qualifications"),
        "responsibilities": highlights.get("Responsibilities"),
    }


def build_prompt(jobs: list, config: dict, resume_text: str, *, markdown: bool = False) -> str:
    """Build the analysis prompt for LLM providers.
    
    Assembles the complete prompt from:
    1. User's custom prompt (from YAML) with exclusion rules injected
    2. Scoring calibration guidance (generic, code-generated)
    3. Output format instructions
    4. Job listings (whitelisted fields only)
    5. Resume text
    """
    # Extract whitelisted fields for each job
    jobs_text = ""
    for job in jobs:
        whitelisted = extract_whitelisted_job_fields(job)
        jobs_text += f"\n--- JOB: {whitelisted.get('job_id')} ---\n"
        jobs_text += f"Job ID: {whitelisted.get('job_id')}\n"
        jobs_text += f"Title: {whitelisted.get('job_title')}\n"
        jobs_text += f"Employer: {whitelisted.get('employer_name')}\n"
        if whitelisted.get('job_publisher'):
            jobs_text += f"Publisher: {whitelisted.get('job_publisher')}\n"
        jobs_text += f"Employment Type: {whitelisted.get('job_employment_type')}\n"
        jobs_text += f"Application URL: {whitelisted.get('job_apply_link') or ''}\n"
        jobs_text += f"Apply Options: {json.dumps(whitelisted.get('apply_options') or [])}\n"
        jobs_text += f"Remote: {whitelisted.get('job_is_remote')}\n"
        if whitelisted.get('qualifications'):
            jobs_text += f"Qualifications: {whitelisted.get('qualifications')}\n"
        if whitelisted.get('responsibilities'):
            jobs_text += f"Responsibilities: {whitelisted.get('responsibilities')}\n"
        jobs_text += f"Description: {whitelisted.get('job_description')}\n"

    # Build dynamic exclusion rules from config
    exclusion_rules = build_exclusion_rules(config)
    
    base_prompt = (
        (config.get("prompt") or "")
        .replace("{job_count}", str(len(jobs)))
        .replace("{exclusion_rules}", exclusion_rules)
    )

    resume_block = resume_text if resume_text else "[No resume text provided]"

    scoring_guidance = build_scoring_guidance()

    output_instruction = """
============================================================
IMPORTANT: OUTPUT FORMAT OVERRIDE
============================================================

Return ONLY a JSON array containing the job_id, score, reason, job_apply_link, and apply_options for each matching job.
DO NOT include any other job details in your response.

Format your response as a JSON array like this:
[
  {
    "job_id": "abc123==",
    "score": 95,
    "reason": "Brief reason",
    "job_apply_link": "https://example.com/apply/abc123",
    "apply_options": [{"publisher": "LinkedIn", "apply_link": "https://..."}]
  },
  {
    "job_id": "xyz789==",
    "score": 92,
    "reason": "Brief reason",
    "job_apply_link": "https://example.com/apply/xyz789",
    "apply_options": []
  }
]

Each object must have exactly these fields:
- "job_id": The exact job_id string from the job listing (preserve the == suffix if present)
- "score": Integer match score (85-100)
- "reason": Brief 1-2 sentence explanation of why this job matches
- "job_apply_link": Copy exactly from the matching job listing (string or null)
- "apply_options": Copy exactly from the matching job listing (array; include all options, use [] if none)

Return ONLY this JSON array - no markdown, no additional text, no other job details.
"""

    # Double the user's prompt for increased accuracy (sends instructions twice)
    doubled_prompt = f"{base_prompt}\n\n{base_prompt}"

    prompt = f"""
{doubled_prompt}

{scoring_guidance}

{output_instruction}

=== JOB LISTINGS TO EVALUATE ===
{jobs_text}
=== END OF JOB LISTINGS ===

=== REFERENCE: MY RESUME (use ONLY to assess fit - NOT a job listing) ===
{resume_block}
=== END OF RESUME ===

CRITICAL REMINDER:
- Return ONLY a JSON array with job_id, score, reason, job_apply_link, and apply_options for each matching job
- Use the EXACT job_id from the listings (including any == suffix)
- Copy job_apply_link and apply_options exactly from each matching listing
- Do not include any other fields or formatting
- Apply the SCORING CALIBRATION rules above — reject surface-level keyword matches
""".strip()

    return prompt


def analyze_jobs(
    jobs: list, provider: LLMProvider, config: dict, resume_text: str, *, markdown: bool = False, debug: bool = False
) -> str:
    """Sends all jobs to the configured LLM for analysis.
    
    Args:
        jobs: List of job dictionaries to analyze.
        provider: The LLM provider to use for analysis.
        config: Application configuration dict.
        resume_text: The user's resume text.
        markdown: Whether to return markdown (True) or JSON (False).
        debug: Whether to save the LLM query to a debug file.
        
    Returns:
        The LLM's analysis response (JSON array of job_id, score, reason, job_apply_link, apply_options).
    """
    print(f"Sending {len(jobs)} jobs to {provider.name} ({provider.model})", file=sys.stderr)
    
    prompt = build_prompt(jobs, config, resume_text, markdown=markdown)
    
    # Debug: save the LLM query to a file for review
    if debug:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        debug_filename = f"llmquery-debug-{timestamp}.json"
        debug_data = {
            "timestamp": timestamp,
            "provider": provider.name,
            "model": provider.model,
            "job_count": len(jobs),
            "prompt_length_chars": len(prompt),
            "prompt": prompt,
        }
        Path(debug_filename).write_text(json.dumps(debug_data, indent=2), encoding="utf-8")
        print(f"  Debug: saved LLM query to {debug_filename}", file=sys.stderr)
        
        # TEMPORARY: Hard break to review query before sending to LLM
        input("Press Enter to continue and send query to LLM (or Ctrl+C to abort)...")
    
    return provider.generate(prompt)


def parse_llm_job_scores(llm_response: str) -> list[dict]:
    """Parse the LLM response to extract matched job entries.
    
    Args:
        llm_response: Raw LLM response text containing JSON array.
        
    Returns:
        List of dicts with job_id, score, reason, and optional link/apply options.
    """
    text = llm_response.strip()
    
    # Strip markdown code fences if present
    if text.startswith('```'):
        first_newline = text.find('\n')
        if first_newline != -1:
            text = text[first_newline + 1:]
        if '```' in text:
            text = text.rsplit('```', 1)[0]
        text = text.strip()
    
    # Try to find and extract JSON array
    if not text.startswith('['):
        start = text.find('[')
        end = text.rfind(']')
        if start >= 0 and end > start:
            text = text[start:end + 1]
    
    try:
        results = json.loads(text)
        if isinstance(results, list):
            return results
    except json.JSONDecodeError:
        pass
    
    return []


def merge_scores_with_jobs(
    llm_results: list[dict], 
    jobs: list[dict], 
    *, 
    markdown: bool = False
) -> str:
    """Merge LLM scoring results back with full job data.
    
    Takes LLM match entries and combines them with
    the full job data to produce the final output.
    
    Args:
        llm_results: List of dicts with job_id, score, reason, and optional apply metadata from LLM.
        jobs: Full list of job dictionaries from JSearch.
        markdown: Whether to output markdown (True) or JSON (False).
        
    Returns:
        Complete analysis output with full job details and scores.
    """
    # Build lookup dict for jobs by job_id
    jobs_by_id = {job.get("job_id"): job for job in jobs if job.get("job_id")}
    
    # Build output by matching LLM results to full job data
    matched_jobs = []
    
    for result in llm_results:
        job_id = result.get("job_id")
        score = result.get("score")
        reason = result.get("reason", "")
        
        if not job_id or score is None:
            continue
        
        full_job = jobs_by_id.get(job_id)
        if not full_job:
            print(f"  Warning: job_id '{job_id}' from LLM not found in job list", file=sys.stderr)
            continue
        
        # Build the complete job record with score
        matched_job = {
            "position": full_job.get("job_title"),
            "company": full_job.get("employer_name"),
            "score": score,
            "reason": reason,
            "requirements": _extract_requirements(full_job),
            "short_description": _build_short_description(full_job),
            "link": result.get("job_apply_link", full_job.get("job_apply_link")),
            "job_apply_link": result.get("job_apply_link", full_job.get("job_apply_link")),
            "apply_options": result.get("apply_options"),
            # Include additional useful fields
            "job_id": job_id,
            "employment_type": full_job.get("job_employment_type"),
            "is_remote": full_job.get("job_is_remote"),
            "location": full_job.get("job_location"),
            "city": full_job.get("job_city"),
            "state": full_job.get("job_state"),
            "country": full_job.get("job_country"),
        }
        if not isinstance(matched_job.get("apply_options"), list):
            matched_job["apply_options"] = full_job.get("apply_options") or []
        matched_jobs.append(matched_job)
    
    # Sort by score descending
    matched_jobs.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    if markdown:
        return _format_jobs_as_markdown(matched_jobs)
    else:
        return json.dumps(matched_jobs, indent=2)


def _extract_requirements(job: dict) -> list[str]:
    """Extract key requirements from job highlights."""
    highlights = job.get("job_highlights") or {}
    qualifications = highlights.get("Qualifications") or []
    
    # Return first few qualifications as requirements
    if isinstance(qualifications, list):
        return qualifications[:5]  # Limit to first 5
    return []


def _build_short_description(job: dict) -> str:
    """Build a short description from job data."""
    description = job.get("job_description") or ""
    
    # Take first ~500 chars as summary
    if len(description) > 500:
        # Try to cut at a sentence boundary
        cut_point = description[:500].rfind('. ')
        if cut_point > 200:
            return description[:cut_point + 1]
        return description[:500] + "..."
    
    return description


def _format_jobs_as_markdown(jobs: list[dict]) -> str:
    """Format matched jobs as markdown output."""
    lines = []
    
    for job in jobs:
        company = job.get("company", "Unknown Company")
        position = job.get("position", "Unknown Position")
        score = job.get("score", 0)
        reason = job.get("reason", "")
        link = job.get("link", "")
        requirements = job.get("requirements", [])
        short_desc = job.get("short_description", "")
        is_remote = job.get("is_remote", False)
        location = job.get("location") or ""
        
        lines.append(f"## {company} - {position}")
        lines.append("")
        lines.append(f"**Match Score:** {score}/100")
        lines.append("")
        lines.append(f"**Why it fits:** {reason}")
        lines.append("")
        
        if requirements:
            lines.append("**Requirements:**")
            for req in requirements:
                lines.append(f"- {req}")
            lines.append("")
        
        lines.append(f"**Remote:** {'Yes' if is_remote else 'No'}")
        if location:
            lines.append(f"**Location:** {location}")
        lines.append("")
        
        if short_desc:
            lines.append(f"**Description:** {short_desc}")
            lines.append("")
        
        lines.append(f"**[Apply Here]({link})**")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


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
    parser.add_argument(
        "--refresh-resume",
        action="store_true",
        help="Force recompression of resume even if cached version exists.",
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

    # Load and compress resume (uses cache if valid)
    print("Loading resume...", file=sys.stderr)
    compressed_resume, full_resume = get_resume_for_matching(
        resume_path, 
        llm_provider,
        force_refresh=args.refresh_resume,
    )
    
    if not compressed_resume:
        print("Warning: No resume available for matching.", file=sys.stderr)

    print("Fetching job listings...", file=sys.stderr)
    jobs = get_bulk_jobs(config, rapidapi_key, debug=args.debug)
    print(f"{len(jobs)} raw jobs fetched.", file=sys.stderr)
    
    # Pre-LLM filtering pipeline (cheapest/most aggressive filters first)
    jobs = deduplicate_jobs(jobs)
    jobs = filter_blocked_companies(jobs, config)
    jobs = filter_by_country(jobs, config)
    jobs = filter_blocked_link_domains(jobs, config)
    jobs = filter_staffing_agencies(jobs, config)
    jobs = filter_jobs_by_exclusions(jobs, config)
    print(f"{len(jobs)} jobs remaining after all pre-filters.", file=sys.stderr)
    
    if jobs:
        # Step 1: Send whitelisted job data to LLM, get back job_id + score pairs
        # Use compressed resume for efficient matching
        llm_response = analyze_jobs(jobs, llm_provider, config, compressed_resume, markdown=args.markdown, debug=args.debug)
        
        # Debug: show raw response from LLM
        print(f"DEBUG: Raw {llm_provider.name} response length: {len(llm_response)} chars", file=sys.stderr)
        print(f"DEBUG: Raw {llm_provider.name} response preview (first 500 chars): {llm_response[:500]}", file=sys.stderr)
        
        # Step 2: Parse LLM response to extract job_id and score pairs
        llm_results = parse_llm_job_scores(llm_response)
        print(f"DEBUG: Parsed {len(llm_results)} job scores from LLM response", file=sys.stderr)
        
        # Step 3: Merge LLM scores with full job data to produce final output
        analysis = merge_scores_with_jobs(llm_results, jobs, markdown=args.markdown)
        print(f"DEBUG: Final analysis length: {len(analysis)} chars", file=sys.stderr)

        # Count actual results
        selected_count = len(llm_results)

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