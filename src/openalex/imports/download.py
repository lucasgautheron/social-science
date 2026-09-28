#!/usr/bin/env python3
import argparse
import gzip
import json
import time
from collections.abc import Sequence
from datetime import date, datetime
from os import makedirs
from os.path import exists

from pyalex import Subfields, Topics, Works

# Social Science domain ID
domain_id = "https://openalex.org/domains/3"

DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2025-12-31"


def parse_date(value, end_of_year=False):
    """Parse YYYY or YYYY-MM-DD command-line dates."""
    try:
        if len(value) == 4:
            year = int(value)
            if end_of_year:
                return date(year, 12, 31)
            return date(year, 1, 1)
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} must be a year or ISO date like 2005 or 2005-01-31"
        ) from exc


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Fetch Social Science works from OpenAlex.")
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=parse_date(DEFAULT_START_DATE),
        help=f"Start publication date, as YYYY or YYYY-MM-DD. Default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        type=lambda value: parse_date(value, end_of_year=True),
        default=parse_date(DEFAULT_END_DATE, end_of_year=True),
        help=f"End publication date, as YYYY or YYYY-MM-DD. Default: {DEFAULT_END_DATE}",
    )
    parser.add_argument(
        "--failure-wait",
        type=float,
        default=60.0,
        help="Seconds to wait after a failed request. This doubles after each recent failure.",
    )
    parser.add_argument(
        "--failure-reattempts",
        type=int,
        default=3,
        help="Total number of times to try a failed request before raising.",
    )
    args = parser.parse_args(argv)

    if args.end_date < args.start_date:
        parser.error("--end-date must be on or after --start-date")
    if args.failure_wait < 0:
        parser.error("--failure-wait must be non-negative")
    if args.failure_reattempts < 1:
        parser.error("--failure-reattempts must be at least 1")

    return args


def retry_request(operation, failure_wait, failure_reattempts, description):
    """Run an OpenAlex request with exponential backoff for recent failures."""
    for attempt in range(1, failure_reattempts + 1):
        try:
            return operation()
        except StopIteration:
            raise
        except Exception as exc:
            if attempt == failure_reattempts:
                print(f"{description} failed after {attempt} attempts")
                raise

            wait = failure_wait * (2 ** (attempt - 1))
            print(
                f"{description} failed on attempt {attempt}/{failure_reattempts}: {exc}. "
                f"Waiting {wait:g} seconds before retrying."
            )
            time.sleep(wait)


def retry_pages(pager, failure_wait, failure_reattempts, description):
    """Yield paginated OpenAlex results, retrying transient page failures."""
    iterator = iter(pager)
    while True:
        try:
            yield retry_request(
                lambda: next(iterator),
                failure_wait,
                failure_reattempts,
                description,
            )
        except StopIteration:
            return


def get_subfields(failure_wait, failure_reattempts):
    """Get all subfields within the Social Science domain."""
    subfields = []

    res_subfields = Subfields().paginate(per_page=200, method="cursor")
    for page in retry_pages(
        res_subfields,
        failure_wait,
        failure_reattempts,
        "Fetching subfields page",
    ):
        print(page)
        for subfield in page:
            print(subfield)
            if subfield["domain"]["id"] != domain_id:
                continue
            subfields.append(subfield["id"].replace('https://openalex.org/subfields/', ''))

    print(f"Subfields in Social Science: {subfields}")
    print(f"Number of subfields: {len(subfields)}")
    return subfields

oa_status = ["gold", "green", "hybrid", "bronze", "closed"]

# Cache for topics by subfield
topics_cache = {}


def get_topics_for_subfield(subfield_id, failure_wait, failure_reattempts):
    """Get all topics for a given subfield (cached)"""
    if subfield_id in topics_cache:
        print(f"Using cached topics for subfield {subfield_id}")
        return topics_cache[subfield_id]

    print(f"Fetching topics for subfield {subfield_id}")
    topics = []
    res_topics = Topics().filter(subfield={"id": subfield_id}).paginate(per_page=200, method="cursor")
    for page in retry_pages(
        res_topics,
        failure_wait,
        failure_reattempts,
        f"Fetching topics page for subfield {subfield_id}",
    ):
        for topic in page:
            topics.append(topic["id"].replace('https://openalex.org/topics/', ''))

    # Cache the result
    topics_cache[subfield_id] = topics
    print(f"Cached {len(topics)} topics for subfield {subfield_id}")
    return topics


def full_year_bounds(year):
    return date(year, 1, 1), date(year, 12, 31)


def effective_year_bounds(year, start_date, end_date):
    year_start, year_end = full_year_bounds(year)
    return max(start_date, year_start), min(end_date, year_end)


def year_output_dir(base_dir, year, start_date, end_date):
    effective_start, effective_end = effective_year_bounds(year, start_date, end_date)
    year_start, year_end = full_year_bounds(year)
    if effective_start == year_start and effective_end == year_end:
        return f"{base_dir}/{year}"
    return f"{base_dir}/{year}_{effective_start.isoformat()}_{effective_end.isoformat()}"


def date_filtered_query(query, year, start_date, end_date):
    effective_start, effective_end = effective_year_bounds(year, start_date, end_date)
    return query.filter(
        publication_year=year,
        from_publication_date=effective_start.isoformat(),
        to_publication_date=effective_end.isoformat(),
    )


def check_download_complete(base_dir, expected_count):
    """Check if download is complete by checking DONE file or expected files"""
    # First check for DONE file
    done_file = f"{base_dir}/DONE"
    if exists(done_file):
        print("DONE file found, download complete")
        return True

    if expected_count == 0:
        return True

    expected_pages = (expected_count + 199) // 200  # Round up division

    # Check if all expected page files exist
    for page_num in range(1, expected_pages + 1):
        folder = page_num // 100
        file_path = f"{base_dir}/{folder}/page_{page_num}.gz"
        if not exists(file_path):
            return False

    print(f"All {expected_pages} pages exist, download complete")
    # Create DONE file to speed up future checks
    if not exists(base_dir):
        makedirs(base_dir)
    with open(done_file, 'w') as f:
        f.write(f"Completed at {expected_pages} pages\n")
    return True


def fetch_papers_by_topic(year, subfield, topic, oa, start_date, end_date, failure_wait, failure_reattempts):
    """Fetch papers for a specific topic within a subfield"""
    output_dir = year_output_dir(
        f"output/openalex/{oa}/{subfield}/topics/{topic}",
        year,
        start_date,
        end_date,
    )

    if exists(f"{output_dir}/DONE"):
        return

    # Query by topic to get count
    query = date_filtered_query(
        Works().filter(has_abstract=True).filter(primary_topic={"id": topic}).filter(oa_status=oa),
        year,
        start_date,
        end_date,
    )
    n = retry_request(
        query.count,
        failure_wait,
        failure_reattempts,
        f"Counting works for year={year}, topic={topic}, oa={oa}",
    )

    print(f"  Topic {topic}: {n} results found")

    if check_download_complete(output_dir, n):
        print(f"  Topic {topic}: Already complete, skipping")
        return

    # Get the paginated results
    pager = query.paginate(per_page=200, method="cursor")

    page_num = 0
    for page in retry_pages(
        pager,
        failure_wait,
        failure_reattempts,
        f"Fetching works page for year={year}, topic={topic}, oa={oa}",
    ):
        page_num += 1
        print(f"  Processing topic {topic} page {page_num} with {len(page)} papers")

        folder = page_num // 100
        folder_path = f"{output_dir}/{folder}"

        if not exists(folder_path):
            makedirs(folder_path)

        file = gzip.GzipFile(f"{folder_path}/page_{page_num}.gz", "wb")
        file.write(json.dumps(page).encode())
        file.close()

    # Create DONE file after successful completion
    with open(f"{output_dir}/DONE", 'w') as f:
        f.write(f"Completed at {page_num} pages\n")


def fetch_papers(year, subfield, oa, start_date, end_date, failure_wait, failure_reattempts):
    output_dir = year_output_dir(f"output/openalex/{oa}/{subfield}", year, start_date, end_date)

    if exists(f"{output_dir}/DONE"):
        return

    # Build the query filtering by subfield
    query = date_filtered_query(
        Works().filter(has_abstract=True).filter(primary_topic={"subfield": {"id": subfield}}).filter(oa_status=oa),
        year,
        start_date,
        end_date,
    )
    n = retry_request(
        query.count,
        failure_wait,
        failure_reattempts,
        f"Counting works for year={year}, subfield={subfield}, oa={oa}",
    )

    print(f"{n} results found for year={year}, subfield={subfield}, oa={oa}")

    if n == 0:
        return

    # If results exceed 10,000, break down by topics
    if n > 10000:
        print(f"Results exceed 10,000 ({n}). Breaking down by topics...")
        topics = get_topics_for_subfield(subfield, failure_wait, failure_reattempts)
        print(f"Found {len(topics)} topics in subfield {subfield}")

        for topic in topics:
            print(f"Processing topic {topic} in subfield {subfield}")
            fetch_papers_by_topic(
                year,
                subfield,
                topic,
                oa,
                start_date,
                end_date,
                failure_wait,
                failure_reattempts,
            )
        return

    # Check if subfield-level download is complete
    if check_download_complete(output_dir, n):
        print("Subfield-level download already complete, skipping")
        return

    # If results are manageable, proceed with subfield-level fetching
    pager = query.paginate(per_page=200, method="cursor")

    page_num = 0
    for page in retry_pages(
        pager,
        failure_wait,
        failure_reattempts,
        f"Fetching works page for year={year}, subfield={subfield}, oa={oa}",
    ):
        page_num += 1
        print(f"Processing page {page_num} with {len(page)} papers")

        folder = page_num // 100
        folder_path = f"{output_dir}/{folder}"

        if not exists(folder_path):
            makedirs(folder_path)

        file = gzip.GzipFile(f"{folder_path}/page_{page_num}.gz", "wb")
        file.write(json.dumps(page).encode())
        file.close()

    # Create DONE file after successful completion
    with open(f"{output_dir}/DONE", 'w') as f:
        f.write(f"Completed at {page_num} pages\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    subfields = get_subfields(args.failure_wait, args.failure_reattempts)

    for year in range(args.end_date.year, args.start_date.year - 1, -1):
        for subfield in subfields:
            for oa in oa_status:
                print(f"Fetching: year={year}, subfield={subfield}, oa={oa}")
                fetch_papers(
                    year,
                    subfield,
                    oa,
                    args.start_date,
                    args.end_date,
                    args.failure_wait,
                    args.failure_reattempts,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
