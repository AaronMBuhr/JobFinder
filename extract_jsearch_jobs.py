import json
import argparse
import sys
import yaml

def extract_jobs(data):
    extracted_jobs = []

    # Parse the JSON structure
    # The JSON is a list containing search queries, each with a 'response' -> 'data' array
    for query_result in data:
        # Safely get the list of jobs, defaulting to an empty list if not found
        jobs_list = query_result.get('response', {}).get('data', [])
        
        for job in jobs_list:
            # 3. Extract the requested fields
            job_info = {
                'job_id': job.get('job_id'),
                'job_title': job.get('job_title'),
                'employer_name': job.get('employer_name'),
                'job_apply_link': job.get('job_apply_link'),
                'job_description': job.get('job_description')
            }
            extracted_jobs.append(job_info)

    return extracted_jobs


def extract_jobs_to_yaml(input_json_path, output_yaml_path):
    # Read JSON from file
    with open(input_json_path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    extracted_jobs = extract_jobs(data)

    # Write YAML to file
    with open(output_yaml_path, 'w', encoding='utf-8') as output_file:
        # default_flow_style=False ensures it prints in block YAML format (highly readable)
        # sort_keys=False preserves the dictionary order of the fields
        # allow_unicode=True prevents escaping issues with special characters in descriptions
        yaml.dump(extracted_jobs, output_file, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Successfully extracted {len(extracted_jobs)} jobs to '{output_yaml_path}'")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract selected job fields from JSearch JSON and output YAML."
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input JSON file path. If omitted, read JSON from stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output YAML file path. If omitted, write YAML to stdout.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.input:
        with open(args.input, 'r', encoding='utf-8') as input_file:
            data = json.load(input_file)
    else:
        data = json.load(sys.stdin)

    extracted_jobs = extract_jobs(data)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as output_file:
            yaml.dump(extracted_jobs, output_file, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"Successfully extracted {len(extracted_jobs)} jobs to '{args.output}'", file=sys.stderr)
    else:
        yaml.dump(extracted_jobs, sys.stdout, default_flow_style=False, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()
    