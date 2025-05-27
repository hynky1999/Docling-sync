from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.media.extractors.extractors import PyMuPDFExtractor, DoclingExtractor
from datatrove.pipeline.media.readers.zstd_threaded import ZstdThreadedReader
from datatrove.pipeline.media.readers.warc_threaded import WarcReaderFast
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from dotenv import load_dotenv

load_dotenv()

NON_TRUNCATED_INPUT_DIR = "./Docling-sync/indices"
from docling.datamodel.settings import settings
settings.debug.visualize_layout = True

# Set AWS credentials

def create_pipeline():
    reader = WarcReaderFast(data_folder="s3://commoncrawl", workers=1)

    extractor = DoclingExtractor(timeout=5*60)

    pipeilne = [
        JsonlReader(
            data_folder=f"{NON_TRUNCATED_INPUT_DIR}",
            limit=10,
            glob_pattern="*.jsonl.gz",
            shuffle_paths=True,
            doc_progress=True,
        ),
        reader,
        extractor,
    ]
    # pipeilne.append(
    #     JsonlWriter(
    #         output_folder=f"{OUTPUT_DIR}/{truncation_str}/success",
    #     )
    # )

    return pipeilne

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Flag
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pipeline = create_pipeline()

    truncated_executor = LocalPipelineExecutor(pipeline, tasks=1)
    truncated_executor.run()