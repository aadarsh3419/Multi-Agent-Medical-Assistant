import sys
import json
import logging
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


sys.path.append(str(Path(__file__).parent.parent))


from agents.rag_agent import MedicalRAG
from config import Config

import argparse


parser = argparse.ArgumentParser(description="Process some command-line arguments.")


parser.add_argument("--file", type=str, required=False, help="Enter file path to ingest")
parser.add_argument("--dir", type=str, required=False, help="Enter directory path of files to ingest")


args = parser.parse_args()


config = Config()

rag = MedicalRAG(config)


def data_ingestion():

    if args.file:

        file_path = args.file

        result = rag.ingest_file(file_path)
    elif args.dir:

        dir_path = args.dir

        result = rag.ingest_directory(dir_path)

    print("Ingestion result:", json.dumps(result, indent=2))

    return result["success"]


if __name__ == "__main__":

    print("\nIngesting document(s)...")

    ingestion_success = data_ingestion()

    if ingestion_success:
        print("\nSuccessfully ingested the documents.")