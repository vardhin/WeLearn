import arxiv
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_arxiv_papers(query: str, start_date: str, end_date: str, num_papers: int, save_path: str):
    """
    Searches for and downloads papers from arXiv based on a query and date range.

    Note: The arXiv API does not support sorting by citation count. 
    This function sorts by relevance as a proxy for impact.

    Args:
        query (str): The search query (e.g., 'quantum computing').
        start_date (str): The start date for the search in 'YYYY-MM-DD' format.
        end_date (str): The end date for the search in 'YYYY-MM-DD' format.
        num_papers (int): The maximum number of papers to download.
        save_path (str): The directory path where papers will be saved.
    """
    if not os.path.exists(save_path):
        logging.info(f"Creating directory: {save_path}")
        os.makedirs(save_path)

    try:
        # Format dates for the arXiv API query
        start_date_formatted = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
        end_date_formatted = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
        
        # Construct the full query including the date range
        full_query = f'({query}) AND submittedDate:[{start_date_formatted} TO {end_date_formatted}]'
        
        logging.info(f"Searching arXiv with query: '{full_query}'")

        # Search for papers, sorting by relevance
        search = arxiv.Search(
            query=full_query,
            max_results=num_papers,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = list(search.results())
        
        if not results:
            logging.warning("No papers found for the given query and date range.")
            return

        logging.info(f"Found {len(results)} papers. Attempting to download top {min(num_papers, len(results))}.")

        for i, paper in enumerate(results):
            try:
                logging.info(f"Downloading paper {i+1}/{len(results)}: '{paper.title}'")
                # Download the PDF to the specified directory.
                # The default filename is '{entry_id}.{title}.pdf'.
                paper.download_pdf(dirpath=save_path)
            except Exception as e:
                logging.error(f"Failed to download paper '{paper.title}'. Reason: {e}")

        logging.info("Download process finished.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == '__main__':
    # --- Example Usage ---
    # field expert:
    # Parameters
    # PEFT
    # efficient transformers
    # QLoRA
    # quantization
    # mixture of experts

    # collaboration expert:
    # multi-agent
    # MARL
    # decentralized learning
    # peer-to-peer learning
    # federeated learning
    # swarm intelligence
    search_query = 'swarm intelligence'
    number_of_papers = 20
    search_start_date = '2024-06-01'
    search_end_date = '2025-08-31'
    download_path = './collaboration_expert'

    # Call the function
    download_arxiv_papers(
        query=search_query,
        start_date=search_start_date,
        end_date=search_end_date,
        num_papers=number_of_papers,
        save_path=download_path
    )