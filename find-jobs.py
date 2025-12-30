import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from google import genai
import requests
import yaml
from docx import Document  # type: ignore


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


def get_bulk_jobs(config: dict, rapidapi_key: str, *, debug: bool = False) -> list:
    """Fetches ~100-200 fresh remote systems jobs based on config queries."""
    print("Fetching jobs from JSearch...", file=sys.stderr)
    url = "https://jsearch.p.rapidapi.com/search"

    queries = config.get("queries") or []
    days_ago = int(config.get("date_posted_days_ago", 0))
    date_posted = build_date_posted_value(days_ago)
    remote_only = bool(config.get("remote_jobs_only", True))

    all_jobs = []
    debug_responses = []

    for i, q in enumerate(queries):
        querystring = {
            "query": q,
            "page": "1",
            "num_pages": "5",  # 5 pages = ~50 jobs per query
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


def analyze_jobs(
    jobs: list, client: genai.Client, config: dict, resume_text: str, *, markdown: bool = False
) -> str:
    """Sends all jobs to Gemini Flash in one massive context window."""
    print(f"Sending {len(jobs)} jobs to Gemini", file=sys.stderr)

    jobs_text = ""
    for i, job in enumerate(jobs):
        jobs_text += f"\n--- JOB ID {i} ---\n"
        jobs_text += f"Title: {job.get('job_title')}\n"
        jobs_text += f"Link: {job.get('job_apply_link')}\n"
        jobs_text += f"Description: {job.get('job_description')}\n"

    base_prompt = (config.get("prompt") or "").replace("{job_count}", str(len(jobs)))

    resume_block = resume_text if resume_text else "[No resume text provided]"

    if markdown:
        output_instruction = "Return the results in markdown format."
    else:
        output_instruction = "Return the results in json format."

    prompt = f"""
{base_prompt}

{output_instruction}

=== MY RESUME (this is MY background - do NOT include this as a job match) ===
{resume_block}
=== END OF RESUME ===

=== JOB LISTINGS TO EVALUATE (only evaluate jobs from this section) ===
{jobs_text}
=== END OF JOB LISTINGS ===

IMPORTANT: Only return matches from the JOB LISTINGS section above. 
Do NOT include my own work experience from my resume as a job match.
""".strip()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
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

    rapidapi_key = config.get("rapidapi_key")
    google_api_key = config.get("google_api_key")

    if not rapidapi_key or not google_api_key:
        raise ValueError("Both 'rapidapi_key' and 'google_api_key' must be set in find-jobs.yaml.")

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

    client = genai.Client(api_key=google_api_key)

    print("Starting downloading jobs", file=sys.stderr)
    jobs = get_bulk_jobs(config, rapidapi_key, debug=args.debug)
    print(f"{len(jobs)} jobs found.", file=sys.stderr)
    if jobs:
        analysis = analyze_jobs(jobs, client, config, resume_text, markdown=args.markdown)

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
            print(f"{len(jobs)} jobs written to {output_path} on {timestamp}", file=sys.stderr)
        else:
            # Emit results to stdout when not writing to file.
            print(analysis)

        print(f"{len(jobs)} jobs selected.", file=sys.stderr)
    else:
        print("No jobs found for the configured filters.", file=sys.stderr)


if __name__ == "__main__":
    main()