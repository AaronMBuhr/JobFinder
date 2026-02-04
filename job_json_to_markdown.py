#!/usr/bin/env python3
"""
Convert job listing JSON (from JSearch API) to formatted Markdown.

Usage:
    python job_json_to_markdown.py input.json -o output.md
    python job_json_to_markdown.py input.json > output.md
    cat input.json | python job_json_to_markdown.py -o output.md
    python job_json_to_markdown.py -h
"""

import json
import sys
import re
from typing import Optional


def extract_location(job: dict) -> str:
    """Extract location string from job data."""
    parts = []
    if job.get("job_city"):
        parts.append(job["job_city"])
    if job.get("job_state"):
        parts.append(job["job_state"])
    if not parts and job.get("job_location"):
        return job["job_location"]
    if not parts and job.get("job_is_remote"):
        return "Remote"
    return ", ".join(parts) if parts else "Location not specified"


def extract_salary(job: dict) -> Optional[str]:
    """Extract salary range if available."""
    min_sal = job.get("job_min_salary")
    max_sal = job.get("job_max_salary")
    period = job.get("job_salary_period", "YEAR")
    
    if min_sal and max_sal:
        return f"${min_sal:,} - ${max_sal:,} / {period.title()}"
    elif min_sal:
        return f"${min_sal:,}+ / {period.title()}"
    elif max_sal:
        return f"Up to ${max_sal:,} / {period.title()}"
    return None


def summarize_description(description: str, max_sentences: int = 5, max_chars: int = 800) -> str:
    """Extract first few meaningful sentences from description."""
    if not description:
        return "No description available."
    
    # Clean up the description
    lines = description.split('\n')
    
    # Skip header-like lines and find actual content
    content_lines = []
    for line in lines:
        line = line.strip()
        # Skip short lines, headers, or list markers
        if len(line) < 30:
            continue
        if line.endswith(':'):
            continue
        if line.startswith('•') or line.startswith('-'):
            # Clean bullet points for potential use
            line = line.lstrip('•- ').strip()
        if len(line) > 40:
            content_lines.append(line)
    
    # Get sentences from content
    text = ' '.join(content_lines[:10])
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    result = ' '.join(sentences[:max_sentences])
    if len(result) > max_chars:
        result = result[:max_chars - 3] + "..."
    
    return result if result else "See full job description for details."


def extract_requirements(job: dict, max_items: int = 8) -> list[str]:
    """Extract top requirements/qualifications from job data."""
    qualifications = job.get("job_highlights", {}).get("Qualifications", [])
    
    if not qualifications:
        return []
    
    # Filter out non-technical or administrative requirements
    skip_patterns = [
        r'background check',
        r'must be available to',
        r'misconduct',
        r'contingent',
        r'travel as required',
    ]
    
    filtered = []
    for qual in qualifications:
        # Skip administrative/legal requirements
        if any(re.search(pattern, qual, re.IGNORECASE) for pattern in skip_patterns):
            continue
        # Truncate very long requirements
        if len(qual) > 150:
            qual = qual[:147] + "..."
        filtered.append(qual)
    
    return filtered[:max_items]


def extract_key_technologies(job: dict) -> list[str]:
    """Extract key technologies/skills from job data."""
    technologies = set()
    
    # Common tech keywords to look for
    tech_patterns = [
        r'\b(Python|Java|JavaScript|TypeScript|Go|Golang|Rust|C\+\+|C#|Ruby|PHP|Swift|Kotlin)\b',
        r'\b(React|Vue\.?js|Angular|Node\.?js|Django|Flask|Spring|\.NET)\b',
        r'\b(AWS|Azure|GCP|Google Cloud|Kubernetes|Docker|Terraform)\b',
        r'\b(PostgreSQL|MySQL|MongoDB|Redis|Kafka|Elasticsearch)\b',
        r'\b(Git|Jenkins|GitHub Actions|CI/CD)\b',
        r'\b(Linux|Unix|Embedded|CUDA)\b',
        r'\b(Machine Learning|ML|AI|NLP|Deep Learning)\b',
        r'\b(REST|GraphQL|API|Microservices)\b',
    ]
    
    description = job.get("job_description", "")
    qualifications = job.get("job_highlights", {}).get("Qualifications", [])
    
    text_to_search = description + " " + " ".join(qualifications)
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, text_to_search, re.IGNORECASE)
        technologies.update(m if isinstance(m, str) else m[0] for m in matches)
    
    return list(technologies)[:8]  # Limit to top 8


def generate_fit_analysis(job: dict, profile: Optional[dict] = None) -> str:
    """Generate a 'Why it fits' analysis based on job requirements."""
    techs = extract_key_technologies(job)
    
    if not techs:
        description = job.get("job_description", "").lower()
        if "remote" in description:
            techs.append("Remote")
        if "senior" in job.get("job_title", "").lower():
            techs.append("Senior-level position")
    
    if techs:
        return f"Involves {', '.join(techs[:5])}. " + (
            "Remote possible." if job.get("job_is_remote") else ""
        )
    
    return "See job description for requirements."


def job_to_markdown(job: dict, match_score: Optional[int] = None) -> str:
    """Convert a single job entry to markdown format."""
    title = job.get("job_title", "Unknown Position")
    company = job.get("employer_name", "Unknown Company")
    location = extract_location(job)
    description = summarize_description(job.get("job_description", ""))
    apply_link = job.get("job_apply_link", "#")
    salary = extract_salary(job)
    fit_analysis = generate_fit_analysis(job)
    requirements = extract_requirements(job)
    
    # Build markdown
    lines = [
        f'### **{title}**',
        '',
        f'**Company:** {company} ({location})',
    ]
    
    if match_score is not None:
        lines.append(f'**Match Score:** {match_score}')
    
    if salary:
        lines.append(f'**Salary:** {salary}')
    
    lines.extend([
        '',
        f'**Technologies:** {fit_analysis}',
        '',
        f'**Description:** {description}',
    ])
    
    if requirements:
        lines.append('')
        lines.append('**Requirements:**')
        for req in requirements:
            lines.append(f'  - {req}')
    
    lines.extend([
        '',
        f'**[Apply Here]({apply_link})**',
        '',
        '---',
        '',
    ])
    
    return '\n'.join(lines)


def extract_jobs_from_json(json_data: dict | list) -> list:
    """Recursively extract job listings from various JSON structures."""
    jobs = []
    
    if isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, dict):
                # Check if this item looks like a job (has job_title or job_id)
                if "job_title" in item or "job_id" in item:
                    jobs.append(item)
                else:
                    # Recurse into nested structures
                    jobs.extend(extract_jobs_from_json(item))
    elif isinstance(json_data, dict):
        # Check if this dict is a job itself
        if "job_title" in json_data or "job_id" in json_data:
            jobs.append(json_data)
        else:
            # Check common nested structures
            if "response" in json_data and isinstance(json_data["response"], dict):
                if "data" in json_data["response"]:
                    jobs.extend(extract_jobs_from_json(json_data["response"]["data"]))
            elif "data" in json_data:
                jobs.extend(extract_jobs_from_json(json_data["data"]))
            else:
                # Try all values in case jobs are nested somewhere
                for value in json_data.values():
                    if isinstance(value, (dict, list)):
                        jobs.extend(extract_jobs_from_json(value))
    
    return jobs


def convert_json_to_markdown(json_data: dict | list) -> str:
    """Convert job listing JSON to markdown."""
    jobs = extract_jobs_from_json(json_data)
    
    if not jobs:
        return "No jobs found in the provided JSON."
    
    markdown_parts = []
    for job in jobs:
        markdown_parts.append(job_to_markdown(job))
    
    return '\n'.join(markdown_parts)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert job listing JSON (from JSearch API) to formatted Markdown."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Input JSON file (use - for stdin, default: stdin)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output Markdown file (default: stdout)"
    )

    args = parser.parse_args()

    # Read input
    if args.input == "-":
        input_data = sys.stdin.read()
    else:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = f.read()

    try:
        json_data = json.loads(input_data)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)

    markdown_output = convert_json_to_markdown(json_data)

    # Write output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(markdown_output)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(markdown_output)


if __name__ == "__main__":
    main()