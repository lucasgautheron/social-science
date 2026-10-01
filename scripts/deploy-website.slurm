#!/bin/bash
# Submit from the repository root:
#   sbatch scripts/deploy-website.slurm
#
# --incidence-workers is set explicitly. The builder does not read the Slurm
# allocation. 16 workers and 64G keep one uncompressed shard per process.
#SBATCH --job-name=event-website
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-website-%j.out

cd "${SLURM_SUBMIT_DIR}"

export OPENALEX_CLUSTERS_DIR="${SLURM_SUBMIT_DIR}/output/event_clusters"
export OPENALEX_NEW_LINK_VISUALIZATIONS_DIR="${SLURM_SUBMIT_DIR}/output/new_link_visualizations"
export OPENALEX_DB_PATH="${SLURM_SUBMIT_DIR}/articles.db"
export OPENALEX_INCIDENCE_WORKERS=16

CLUSTER_TRENDS="${OPENALEX_CLUSTERS_DIR}/cluster_trends.csv"
if [[ -f "$CLUSTER_TRENDS" ]]; then
  export OPENALEX_CLUSTER_TRENDS="$CLUSTER_TRENDS"
fi

scripts/deploy-gh-pages.sh "${SLURM_SUBMIT_DIR}/output/events"
