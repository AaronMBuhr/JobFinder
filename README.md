# JobFinder

AI-powered job search tool that fetches remote job listings and uses Google's Gemini to filter and rank them against your resume.

## Features

- **Bulk job fetching** via [JSearch API](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch) (RapidAPI)
- **AI-powered filtering** using Google Gemini 2.0 Flash
- **Resume matching** — supports `.txt`, `.md`, and `.docx` formats
- **Configurable search** — queries, date range, remote-only filter
- **Custom prompts** — tailor the LLM instructions to your profile
- **Flexible output** — JSON or Markdown, to stdout or timestamped files

## Prerequisites

- Python 3.10+
- [RapidAPI key](https://rapidapi.com/) with JSearch subscription
- [Google AI API key](https://aistudio.google.com/apikey)

## Installation

```bash
git clone https://github.com/yourusername/JobFinder.git
cd JobFinder
pip install -r requirements.txt
```

## Configuration

Create a `find-jobs.yaml` file (or copy and edit the example below):

```yaml
rapidapi_key: "YOUR_RAPIDAPI_KEY"
google_api_key: "YOUR_GOOGLE_API_KEY"

# Search queries — at least one required
queries:
  - "Senior Software Engineer Remote"
  - "Backend Developer Remote"
  - "Systems Engineer Remote"

# Date window: 0 = today only, 1 = since yesterday, 3 = last 3 days
date_posted_days_ago: 1

# Limit to remote jobs only
remote_jobs_only: true

# Path to your resume (.txt, .md, or .docx)
# Use forward slashes or escape backslashes on Windows
resume_file: "resume.docx"

# REQUIRED: Prompt sent to Gemini (use {job_count} placeholder if needed)
prompt: |
  I am a Senior Software Engineer.
  Attached is my Resume and a list of {job_count} recent job postings.

  TASK:
  Identify the jobs that match my profile with a score of 90/100 or higher.

  STRICT FILTERING CRITERIA:
  1. MUST involve Python, Go, or systems programming.
  2. MUST be backend, infrastructure, or platform engineering.
  3. REJECT all Frontend, React, or Web Design roles.
  4. REJECT all staffing agencies.

  Return the results in descending order by score.
```

### Required Fields

| Field | Description |
|-------|-------------|
| `rapidapi_key` | Your RapidAPI key for JSearch |
| `google_api_key` | Your Google AI API key |
| `queries` | List of search terms (at least one) |
| `prompt` | Instructions for Gemini filtering |
| `resume_file` | Path to resume (or use `--resume` flag) |

### Optional Fields

| Field | Default | Description |
|-------|---------|-------------|
| `date_posted_days_ago` | `0` | How far back to search |
| `remote_jobs_only` | `true` | Filter to remote positions |

## Usage

```bash
python find-jobs.py [options]
```

### Options

| Flag | Description |
|------|-------------|
| `-c, --config FILE` | Path to YAML config (default: `find-jobs.yaml`) |
| `-r, --resume FILE` | Override resume file path |
| `-f, --from-days N` | Override date window (0=today, 1=yesterday, etc.) |
| `-o, --output-to-file` | Write to timestamped file instead of stdout |
| `-d, --directory DIR` | Output directory for the file (used with `-o`) |
| `-n, --name TOKEN` | Add name token to output filename |
| `-m, --markdown` | Output in Markdown format instead of JSON |

### Examples

**Basic run** — fetch today's jobs, print JSON to stdout:
```bash
python find-jobs.py
```

**Last 3 days, save to file:**
```bash
python find-jobs.py -f 3 -o
# Output: find-jobs-20251230-143022.json
```

**Named output in Markdown:**
```bash
python find-jobs.py -f 1 -o -n backend -m
# Output: find-jobs-backend-20251230-143022.md
```

**Save to specific directory:**
```bash
python find-jobs.py -f 3 -o -d ./results
# Output: ./results/find-jobs-20251230-143022.json
```

**Different config and resume:**
```bash
python find-jobs.py -c custom-config.yaml -r my-resume.docx
```

**Redirect stdout, monitor progress on stderr:**
```bash
python find-jobs.py -f 3 > results.json
```

## Output

### JSON Format (default)

```json
[
  {
    "job_id": 0,
    "title": "Senior Backend Engineer",
    "company": "Acme Corp",
    "score": 95,
    "reason": "Strong C++ and systems experience matches role requirements",
    "short_description": "Backend systems development in C++",
    "link": "https://example.com/apply"
  }
]
```

### Markdown Format (`-m`)

Human-readable format with job details, scores, and reasoning.

## Status Messages

Progress is printed to stderr so you can monitor while redirecting stdout:

```
Starting downloading jobs
Fetching jobs from JSearch...
  Got 50 jobs for 'Senior Software Engineer Remote'
  Got 42 jobs for 'Backend Developer Remote'
140 jobs found.
Sending 140 jobs to Gemini
140 jobs written to find-jobs-20251230-143022.json on 20251230-143022
140 jobs selected.
```

## Rate Limits

- **JSearch API**: Depends on your RapidAPI plan
- **Gemini API**: Free tier has token/request limits; enable billing for higher quotas

If you hit rate limits, wait ~60 seconds or reduce the number of queries.

## License

MIT

