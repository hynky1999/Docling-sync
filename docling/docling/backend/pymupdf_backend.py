import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import fitz
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import BoundingRectangle, SegmentedPdfPage, TextCell
from PIL import Image
from pymupdf import Page as PyMuPDFPage

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class PyMuPdfPageBackend(PdfPageBackend):
    def __init__(self, doc_obj: fitz.Document, document_hash: str, page_no: int):
        super().__init__()
        self.valid = True

        try:
            self._fpage: fitz.Page = doc_obj.load_page(page_no)
        except Exception as e:
            _log.info(
                f"An exception occured when loading page {page_no} of document {document_hash}.",
                exc_info=True,
            )
            self.valid = False

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        if not self.valid:
            return ""

        if bbox.coord_origin != CoordOrigin.TOPLEFT:
            bbox = bbox.to_top_left_origin(self.get_size().height)

        rect = fitz.Rect(*bbox.as_tuple())
        text_piece = self._fpage.get_text("text", clip=rect)

        return text_piece

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        return None

    def get_text_cells(self) -> Iterable[TextCell]:
        cells = []

        if not self.valid:
            return cells

        cell_counter = 0

        blocks = self._fpage.get_text(
            "dict"  # , flags=fitz.TEXTFLAGS_DICT | ~fitz.TEXT_CID_FOR_UNKNOWN_UNICODE #&
        )["blocks"]
        for b in blocks:
            for l in b.get("lines", []):  # noqa: E741
                for s in l["spans"]:
                    text = s["text"]
                    bbox = s["bbox"]
                    x0, y0, x1, y1 = bbox

                    cells.append(
                        TextCell(
                            index=cell_counter,
                            text=text,
                            orig=text,
                            from_ocr=False,
                            rect=BoundingRectangle.from_bounding_box(
                                BoundingBox(
                                    l=x0,
                                    b=y0,
                                    r=x1,
                                    t=y1,
                                )
                            ),
                        )
                    )
                    cell_counter += 1

        return cells

    def get_bitmap_rects(self, scale: int = 1) -> Iterable["BoundingBox"]:
        AREA_THRESHOLD = 32 * 32

        images = self._fpage.get_image_info()

        for im in images:
            cropbox = BoundingBox.from_tuple(im["bbox"], origin=CoordOrigin.TOPLEFT)
            if cropbox.area() > AREA_THRESHOLD:
                cropbox = cropbox.scaled(scale=scale)

                yield cropbox

    def get_page_image(
        self, scale: int = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        if not self.valid:
            return None

        if not cropbox:
            pix = self._fpage.get_pixmap(matrix=fitz.Matrix(scale, scale))
        else:
            page_height = self.get_size().height
            cropbox = cropbox.to_top_left_origin(page_height)
            pix = self._fpage.get_pixmap(
                matrix=fitz.Matrix(scale, scale), clip=cropbox.as_tuple()
            )

        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples_mv)
        return image

    def get_size(self) -> Size:
        return Size(width=self._fpage.cropbox.width, height=self._fpage.cropbox.height)

    def is_valid(self) -> bool:
        return self.valid

    def unload(self):
        self._fpage = None


class PyMuPdfDocumentBackend(PdfDocumentBackend):
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        success = False
        if isinstance(self.path_or_stream, Path):
            self._fdoc: fitz.Document = fitz.open(str(self.path_or_stream))
            success = True
        elif isinstance(self.path_or_stream, BytesIO):
            self._fdoc: fitz.Document = fitz.open(
                filename=str(uuid.uuid4()), filetype="pdf", stream=path_or_stream
            )
            success = True

        if not success:
            raise RuntimeError(
                f"PyMuPdf could not load document with hash {self.document_hash}."
            )

    def page_count(self) -> int:
        return self._fdoc.page_count

    def load_page(self, page_no: int) -> PyMuPDFPage:
        return PyMuPdfPageBackend(self._fdoc, self.document_hash, page_no)

    def is_valid(self) -> bool:
        return self.page_count() > 0

    def unload(self):
        self._fdoc.close()
        self._fdoc = None