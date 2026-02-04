# Prompts Guide for Job Finder

This guide documents lessons learned about optimizing the Gemini prompt in `find-jobs.yaml` to get better job matches.

---

## How the Pipeline Works

1. **JSearch API** fetches raw job listings based on `queries` in config
2. **Date filter** excludes jobs older than `date_posted_days_ago`
3. **Gemini** evaluates ALL remaining jobs against your resume and prompt criteria
4. **Post-processing** validates URLs and removes invalid entries

**Key insight:** Gemini sees hundreds of jobs and must rank them. Your prompt criteria heavily influence what makes the cut.

---

## Common Problems & Solutions

### Problem 1: Gemini Violates Rejection Criteria

**Symptom:** Jobs with Java, Node.js, or TypeScript appearing in results despite "REJECT Java" in prompt.

**Cause:** Vague rejection language like "Java-primary roles" is interpreted loosely.

**Solution:** Use explicit, unambiguous rejection lists:

```yaml
HARD REJECTIONS (exclude regardless of other fit):
- Frontend, React, Angular, or Web Design roles
- Java-primary or Kotlin-primary roles (Java as secondary is OK)
- Node.js-primary or TypeScript-primary roles
```

### Problem 2: Title Bias ("Software Engineer" vs "Systems Engineer")

**Symptom:** Gemini only returns "Software Engineer" titles, ignoring "Systems Engineer" roles that match your skills.

**Cause:** LLMs have implicit biases toward common phrasings.

**Solution:** Explicitly state title parity:

```yaml
PREFERRED: "Systems Engineer" titles are EQUALLY valuable as "Software Engineer"
```

### Problem 3: AI/ML Over-Weighting

**Symptom:** Results dominated by AI/ML roles when you want general backend/systems work.

**Cause:** AI/ML is a "hot" keyword that LLMs tend to prioritize.

**Solution:** Add criteria to balance:

```yaml
- Roles where AI/ML is the ONLY focus without backend systems work
```

Or if you DO want AI roles, be explicit about the balance you want.

### Problem 4: Duplicate Listings

**Symptom:** Same company/role appears 3-4 times (e.g., job posted on multiple boards).

**Cause:** No deduplication instruction.

**Solution:** Add explicit rule:

```yaml
6. DO NOT include duplicate positions (same company + similar role = pick one)
```

### Problem 5: Good Jobs Scoring Just Below Threshold

**Symptom:** Jobs you'd want are in the raw data but not in results.

**Cause:** Score threshold too high (e.g., >= 90), and Gemini's scoring is subjective.

**Solution:** 
- Lower threshold: `>= 85/100`
- Request more results: `Return UP TO 25 results`
- Add scoring guidance so Gemini knows what deserves what score

### Problem 6: Date Filter Excludes Good Jobs

**Symptom:** Jobs you manually found in the debug JSON aren't in results.

**Cause:** `date_posted_days_ago` is too restrictive.

**Solution:** Increase the window:

```yaml
date_posted_days_ago: 7   # or 14 for two weeks
```

---

## Prompt Structure Best Practices

### 1. Use Hierarchical Rules

Structure rules from most to least important:

```yaml
CRITICAL RULES:     # Must always be followed
MATCHING CRITERIA:  # What to look for
HARD REJECTIONS:    # What to absolutely exclude
SCORING GUIDANCE:   # How to rank what's left
```

### 2. Be Specific About Preferences

Bad:
```yaml
- Backend roles preferred
```

Good:
```yaml
- PREFERRED: Infrastructure roles (Kubernetes, Linux, Docker, Terraform, DevOps)
- PREFERRED: Enterprise SaaS, multi-tenant architecture, backend services
- PREFERRED: Game backend/server architecture
```

### 3. Provide Scoring Rubrics

Gemini's scoring is arbitrary without guidance:

```yaml
SCORING GUIDANCE:
- Score 95+: C++/C# backend + systems/infrastructure focus
- Score 90-94: Python backend/infrastructure OR C#/.NET enterprise
- Score 85-89: Strong match but less emphasis on core languages
```

### 4. Specify Output Quantity

Without limits, Gemini may return too few results:

```yaml
Return UP TO 25 results in descending order by score.
```

### 5. Handle Edge Cases Explicitly

```yaml
- AI/ML roles OK if they include substantial backend systems work
- Soft requirements (preferred/nice-to-have) are OK even if you don't have the tech
```

### 6. Use Resume-Driven Filtering (Recommended)

Instead of hardcoding specific technologies to exclude, have the LLM extract your skills 
from your resume and filter dynamically:

```yaml
STEP 1 - BUILD MY SKILLS LIST:
First, read my resume and extract all programming languages, frameworks, tools, and 
technologies mentioned. This is my "skills list" for filtering jobs.

STEP 2 - FILTER JOBS:
EXCLUDE if job has HARD requirement (strong/required/must have/X+ years) for tech NOT on skills list.
KEEP if job has SOFT requirement (preferred/nice-to-have/a plus) for tech NOT on skills list.
```

**Benefits:**
- Anyone can use the tool with their own resume
- No need to maintain hardcoded exclusion lists
- Automatically adapts as your resume changes
- Distinguishes between hard requirements you can't meet vs. nice-to-haves

---

## Debugging Tips

### Check the Debug JSON

Run with `--debug` flag to save raw JSearch responses:

```bash
python find-jobs.py --debug
```

This creates `jsearch-debug-YYYYMMDD-HHMMSS.json` with all jobs before Gemini filtering.

### Compare Your Picks vs Gemini's

1. Manually review the debug JSON and note jobs you'd want
2. Run the pipeline and compare results
3. If your picks aren't selected, analyze:
   - Were they filtered by date? (check `job_posted_at_timestamp`)
   - Do they match your stated criteria? (check job description vs prompt)
   - What did Gemini select instead? (look for patterns)

### Common Patterns in Gemini's Behavior

| Pattern | Meaning |
|---------|---------|
| All results are AI/ML focused | Prompt doesn't explicitly deprioritize AI-only roles |
| Java/Node.js jobs appearing | Rejection criteria too vague |
| Only "Software Engineer" titles | Title bias - add explicit parity statement |
| Many duplicates | No deduplication rule |
| Results seem random | Scoring guidance missing |

---

## Example: Optimized Prompt (Resume-Driven)

The key insight is to make the prompt **resume-driven** rather than hardcoding specific technologies.
This way anyone can use the tool with their own resume.

```yaml
prompt: |
  Below are {job_count} job postings to evaluate against my resume.

  STEP 1 - BUILD MY SKILLS LIST:
  First, read my resume and extract all programming languages, frameworks, tools, and 
  technologies mentioned. This is my "skills list" for filtering jobs.

  STEP 2 - FILTER JOBS (do this BEFORE scoring):
  For each job, check if it has HARD REQUIREMENTS for technologies NOT on my skills list.
  
  EXCLUDE a job if it says any of these for a technology NOT on my skills list:
  - "strong [technology]" or "[technology] strong"
  - "[technology] required" or "required: [technology]"
  - "[technology] experience required"
  - "must have [technology]" or "must know [technology]"
  - "X+ years of [technology]" (where technology is not on my skills list)
  
  DO NOT exclude if the job merely says:
  - "[technology] preferred" or "[technology] nice to have"
  - "[technology] a plus" or "bonus: [technology]"
  - "exposure to [technology]" or "familiarity with [technology]"
  
  STEP 3 - APPLY THESE RULES:
  1. ONLY evaluate items from the JOB LISTINGS section
  2. My resume is PROVIDED FOR REFERENCE ONLY - it is NOT a job listing
  3. Every valid match MUST have an actual application URL starting with http
  4. If something looks like my own work history, EXCLUDE it
  5. If the "link" field is empty or contains parenthetical notes, EXCLUDE it
  6. DO NOT include duplicate positions (same company + similar role = pick one)

  STEP 4 - MATCH AND SCORE remaining jobs:
  - REQUIRED: Must be backend, systems, infrastructure, or data engineering
  - PREFERRED: "Systems Engineer" titles are EQUALLY valuable as "Software Engineer"
  - PREFERRED: Infrastructure roles (Kubernetes, Linux, Docker, Terraform, DevOps)
  - PREFERRED: Enterprise SaaS, multi-tenant architecture, backend services
  - PREFERRED: Game backend/server architecture
  
  HARD REJECTIONS (exclude regardless of other fit):
  - Frontend-primary, React-primary, Angular-primary, or Web Design roles
  - Roles where AI/ML is the ONLY focus without backend systems work

  SCORING GUIDANCE:
  - Score 95+: Strong match on multiple resume technologies + systems/infrastructure focus
  - Score 90-94: Good match on resume technologies + backend/infrastructure
  - Score 85-89: Decent match but fewer resume technology overlaps
  - Include all jobs scoring >= 85/100
  
  Return UP TO 25 results in descending order by score.
```

### Why This Works

| Job Requirement | Resume Has Tech? | Result |
|-----------------|------------------|--------|
| "Strong Go required" | No | ❌ EXCLUDE |
| "5+ years Java experience" | No | ❌ EXCLUDE |
| "Go preferred" | No | ✅ Keep (soft requirement) |
| "Java a plus" | No | ✅ Keep (soft requirement) |
| "Strong Python required" | Yes | ✅ Keep |
| "Must have C++" | Yes | ✅ Keep |

---

## Quick Reference: Config Knobs

| Setting | Purpose | Recommendation |
|---------|---------|----------------|
| `date_posted_days_ago` | Filter old jobs | 7 for weekly runs, 3 for daily |
| `jsearch_pages_per_query` | How many pages to fetch | 10 for broad search |
| `queries` | Search terms | Mix of specific + broad terms |
| Score threshold (in prompt) | Minimum match quality | 85 for more results, 90 for precision |
| Result count (in prompt) | Max jobs returned | 20-25 for variety |

---

## Iterating on Your Prompt

1. **Run with --debug** to capture raw data
2. **Manually review** debug JSON for jobs you'd want
3. **Compare** to actual results
4. **Identify gaps** - what criteria caused good jobs to be excluded?
5. **Adjust prompt** - be more explicit about what you want
6. **Re-run** and compare again

Prompt optimization is iterative. Each run teaches you what Gemini misunderstands.
