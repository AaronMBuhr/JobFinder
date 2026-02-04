#!/usr/bin/env python3
"""
Convert job listing JSON data to an aesthetic Markdown file.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def score_to_badge(score: int) -> str:
    """Return an emoji badge based on score threshold."""
    if score >= 95:
        return "🔥"
    elif score >= 90:
        return "⭐"
    elif score >= 85:
        return "✅"
    else:
        return "📌"


def score_to_color_hint(score: int) -> str:
    """Return a text hint for the score range."""
    if score >= 95:
        return "Excellent Match"
    elif score >= 90:
        return "Strong Match"
    elif score >= 85:
        return "Good Match"
    else:
        return "Potential Match"


def job_to_markdown(job: dict, index: int) -> str:
    """Convert a single job entry to Markdown format."""
    position = job.get("position", "Unknown Position")
    company = job.get("company", "Unknown Company")
    score = job.get("score", 0)
    reason = job.get("reason", "No reason provided.")
    requirements = job.get("requirements", "")
    description = job.get("short_description", "No description available.")
    link = job.get("link", "")

    badge = score_to_badge(score)
    color_hint = score_to_color_hint(score)

    lines = [
        f"### {index}. {position}",
        f"**{company}** {badge}",
        "",
        f"| Match Score | {score}/100 ({color_hint}) |",
        "|-------------|---------------------------|",
        "",
        f"> {description}",
        "",
    ]

    if requirements:
        lines.append(f"**Requirements:** {requirements}")
        lines.append("")

    lines.extend([
        f"**Why it fits:** {reason}",
        "",
    ])

    if link:
        lines.append(f"🔗 [View Job Listing]({link})")
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def convert_jobs_to_markdown(jobs: list, title: str = None) -> str:
    """Convert a list of job entries to a full Markdown document."""
    if title is None:
        title = "Job Listings"

    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # Header
    lines = [
        f"# {title}",
        "",
        f"*Generated: {timestamp}*",
        "",
    ]

    # Summary stats
    if jobs:
        avg_score = sum(j.get("score", 0) for j in jobs) / len(jobs)
        top_score = max(j.get("score", 0) for j in jobs)
        companies = set(j.get("company", "") for j in jobs)

        lines.extend([
            "## Summary",
            "",
            f"- **Total Positions:** {len(jobs)}",
            f"- **Unique Companies:** {len(companies)}",
            f"- **Top Score:** {top_score}",
            f"- **Average Score:** {avg_score:.1f}",
            "",
            "### Score Legend",
            "",
            "| Badge | Range | Meaning |",
            "|-------|-------|---------|",
            "| 🔥 | 95+ | Excellent Match |",
            "| ⭐ | 90-94 | Strong Match |",
            "| ✅ | 85-89 | Good Match |",
            "| 📌 | <85 | Potential Match |",
            "",
            "---",
            "",
            "## Listings",
            "",
        ])

    # Individual jobs
    for i, job in enumerate(jobs, 1):
        lines.append(job_to_markdown(job, i))

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert job listing JSON to Markdown format."
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
    parser.add_argument(
        "-t", "--title",
        default="Job Listings",
        help="Title for the Markdown document"
    )

    args = parser.parse_args()

    # Read input
    if args.input == "-":
        data = json.load(sys.stdin)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Handle both list and dict with 'jobs' key
    if isinstance(data, dict):
        jobs = data.get("jobs", data.get("results", []))
    else:
        jobs = data

    # Convert to Markdown
    markdown = convert_jobs_to_markdown(jobs, title=args.title)

    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(markdown)


if __name__ == "__main__":
    main()