from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.media.extractors.extractors import PyMuPDFExtractor, DoclingExtractor
from datatrove.pipeline.media.readers.zstd_threaded import ZstdThreadedReader
from datatrove.pipeline.media.readers.warc_threaded import WarcReaderFast
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor

NON_TRUNCATED_INPUT_DIR = "s3://fine-pdfs/data/non_truncated_scanned_classified"

class FailedDocsFilter(BaseFilter):
    name = "Failed Docs Filter"

    def filter(self, doc) -> bool:
        if not "pdf_metadata" in doc.media[0].metadata or not "garbled_text_ratio" in doc.media[0].metadata["pdf_metadata"]:
            return False

        encrypted_or_password = doc.media[0].metadata["pdf_metadata"]["is_encrypted"] or doc.media[0].metadata["pdf_metadata"]["needs_password"]
        if encrypted_or_password:
            print(f"Skipping {doc.id} because it is encrypted or has a password")
        return not encrypted_or_password

class OCRFilter(BaseFilter):
    name = "OCR Filter"
    
    def filter(self, doc) -> bool:
        metadata = doc.media[0].metadata["pdf_metadata"]
        ocr_prob = metadata["ocr_prob"]
        garbled_text_ratio = metadata["garbled_text_ratio"]
        needs_ocr = (ocr_prob >= 0.2 or garbled_text_ratio > 0.0)
        return not needs_ocr

pipeline = [
    JsonlReader(
        data_folder=f"{NON_TRUNCATED_INPUT_DIR}/records",
        glob_pattern="05000.jsonl.gz",
        limit=10000,
        shuffle_paths=True,
        doc_progress=True,
    ),
    FailedDocsFilter(),
    OCRFilter(),
    JsonlWriter("/admin/home/hynek_kydlicek/fsx/projects/pdf_project/docling-modfs/indices")
]




LocalPipelineExecutor(pipeline).run()