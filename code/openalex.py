from pyalex import Works, Subfields, Topics
import json
import gzip
from os.path import exists
from os import makedirs
import os

# Social Science domain ID
domain_id = "https://openalex.org/domains/2"

# Get all subfields within the Social Science domain
subfields = []

# Get all subfields in the Social Science domain
res_subfields = Subfields().paginate(per_page=200, method="cursor")
for page in res_subfields:
    print(page)
    for subfield in page:
        print(subfield)
        if subfield["domain"]["id"] != domain_id:
            continue
        subfields.append(subfield["id"].replace('https://openalex.org/subfields/', ''))

print(f"Subfields in Social Science: {subfields}")
print(f"Number of subfields: {len(subfields)}")

oa_status = ["gold", "green", "hybrid", "bronze", "closed"]

# Cache for topics by subfield
topics_cache = {}

def get_topics_for_subfield(subfield_id):
    """Get all topics for a given subfield (cached)"""
    if subfield_id in topics_cache:
        print(f"Using cached topics for subfield {subfield_id}")
        return topics_cache[subfield_id]
    
    print(f"Fetching topics for subfield {subfield_id}")
    topics = []
    res_topics = Topics().filter(subfield={"id": subfield_id}).paginate(per_page=200, method="cursor")
    for page in res_topics:
        for topic in page:
            topics.append(topic["id"].replace('https://openalex.org/topics/', ''))
    
    # Cache the result
    topics_cache[subfield_id] = topics
    print(f"Cached {len(topics)} topics for subfield {subfield_id}")
    return topics

def check_download_complete(base_dir, expected_count):
    """Check if download is complete by checking DONE file or expected files"""
    # First check for DONE file
    done_file = f"{base_dir}/DONE"
    if exists(done_file):
        print(f"DONE file found, download complete")
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

def fetch_papers_by_topic(year, subfield, topic, oa):
    """Fetch papers for a specific topic within a subfield"""
    output_dir = f"output/social_science/{oa}/{subfield}/topics/{topic}/{year}"

    if exists(f"{output_dir}/DONE"):
        return

    # Query by topic to get count
    query = Works().filter(has_abstract=True).filter(primary_topic={"id": topic}).filter(publication_year=year).filter(oa_status=oa)
    n = query.count()
    
    print(f"  Topic {topic}: {n} results found")
    
    if check_download_complete(output_dir, n):
        print(f"  Topic {topic}: Already complete, skipping")
        return

    # Get the paginated results
    pager = query.paginate(per_page=200, method="cursor")

    page_num = 0
    for page in pager:
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

def fetch_papers(year, subfield, oa):
    output_dir = f"output/social_science/{oa}/{subfield}/{year}"

    if exists(f"{output_dir}/DONE"):
        return

    # Build the query filtering by subfield
    query = Works().filter(has_abstract=True).filter(primary_topic={"subfield": {"id": subfield}}).filter(publication_year=year).filter(oa_status=oa)
    n = query.count()
    
    print(f"{n} results found for year={year}, subfield={subfield}, oa={oa}")

    if n == 0:
        return

    # If results exceed 10,000, break down by topics
    if n > 10000:
        print(f"Results exceed 10,000 ({n}). Breaking down by topics...")
        topics = get_topics_for_subfield(subfield)
        print(f"Found {len(topics)} topics in subfield {subfield}")
        
        for topic in topics:
            print(f"Processing topic {topic} in subfield {subfield}")
            fetch_papers_by_topic(year, subfield, topic, oa)
        return

    # Check if subfield-level download is complete
    if check_download_complete(output_dir, n):
        print(f"Subfield-level download already complete, skipping")
        return

    # If results are manageable, proceed with subfield-level fetching
    pager = query.paginate(per_page=200, method="cursor")

    page_num = 0
    for page in pager:
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


# Fetch papers from 2019 onwards (including 2019)
for year in range(2025, 2018, -1):  # 2025 down to 2019
    for subfield in subfields:
        for oa in oa_status:
            print(f"Fetching: year={year}, subfield={subfield}, oa={oa}")
            fetch_papers(year, subfield, oa)