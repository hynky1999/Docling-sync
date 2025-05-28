from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.media.extractors.extractors import PyMuPDFExtractor, DoclingExtractor
from datatrove.pipeline.media.readers.zstd_threaded import ZstdThreadedReader
from datatrove.pipeline.media.readers.warc_threaded import WarcReaderFast
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor

NON_TRUNCATED_INPUT_DIR = "./Docling-sync/indices"
from docling.datamodel.settings import settings

# Set AWS credentials

def create_pipeline(samples: int = 10):
    reader = WarcReaderFast(data_folder="s3://commoncrawl", workers=1)

    extractor = DoclingExtractor(timeout=5*60)

    pipeilne = [
        JsonlReader(
            data_folder=f"{NON_TRUNCATED_INPUT_DIR}",
            limit=samples,
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
    parser.add_argument("--samples", default=10)
    args = parser.parse_args()

    pipeline = create_pipeline(args.samples)

    truncated_executor = LocalPipelineExecutor(pipeline, tasks=1)
    truncated_executor.run()