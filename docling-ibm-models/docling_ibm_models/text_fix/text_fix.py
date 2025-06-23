"""
List Item Marker Processor for Docling Documents

This module provides a rule-based model to identify list item markers and
merge marker-only TextItems with their content to create proper ListItems.
"""

import logging
import fitz
import re
import pymupdf
from fitz import Rect
from typing import List, Optional, Tuple, Union
from docling_core.types.doc.base import CoordOrigin
from docling.backend.pymupdf_backend import blocks_to_cells
from docling.models.page_assemble_model import sanitize_mineru_cells,  sanitize_cells_docling, is_broken_header, DELETE_TEXT_FLAG
from hashlib import sha256

from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel

_log = logging.getLogger(__name__)

# Set it to errror mode
logging.basicConfig(level=logging.INFO)

LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？', ')', '）', '"', '”', ':', '：', ';', '；')

REPROCESS_FLAG = "<--- to_reprocess --->"

class OverlapingTextFixer:

    def process_document(self, doc: DoclingDocument, allow_multi_prov: bool = False) -> DoclingDocument:
        """
        Process the entire document to identify and convert list markers.

        Args:
            doc: The DoclingDocument to process

        Returns:
            The processed document (modified in-place)
        """

        # Iterate over all items in the document, and delete those that are fully inside the preceeding or following item.
        deleted_items = []
        for page in range(doc.num_pages()):
            prev_item: Optional[TextItem] = None
            prev_level = 0
            for item, level in doc.iterate_items(page_no=page+1, with_groups=False):
                prev_level = level
                is_text_item = isinstance(item, TextItem)
                # We can only merge single provenance items.
                if not prev_item or not is_text_item or (len(item.prov) > 1 and not allow_multi_prov) or prev_level != level or item.label != prev_item.label:
                    if is_text_item and len(item.prov) == 1:
                        prev_item = item
                    else:
                        prev_item = None
                    continue

                if prev_item.prov[-1].bbox.is_inside(item.prov[0].bbox):
                    deleted_items.append(prev_item)
                    item.text = REPROCESS_FLAG
                    prev_item = item
                elif item.prov[0].bbox.is_inside(prev_item.prov[-1].bbox):
                    deleted_items.append(item)
                    prev_item.text = REPROCESS_FLAG
                    prev_item = prev_item

                elif item.prov[0].bbox.width == 0 or item.prov[0].bbox.height == 0:
                    deleted_items.append(item)
                    prev_item = None
                else:
                    prev_item = item


        if len(deleted_items) > 0:
            doc.delete_items(node_items=deleted_items)
        return doc

                



class TextFix:
    def process_document(self, doc: DoclingDocument, pymupdf_doc: pymupdf.Document, fix_overlapping_text: bool = True) -> DoclingDocument:
        """
        Process the entire document to identify and convert list markers.

        Args:
            doc: The DoclingDocument to process

        Returns:
            The processed document (modified in-place)
        """

        if fix_overlapping_text:
            doc = OverlapingTextFixer().process_document(doc)

        deleted_nodes = []

        for page in range(doc.num_pages()):
            pymupdf_page = pymupdf_doc.load_page(page)
            
            # OPTIMIZATION: Get textpage once for the entire page
            text_page = pymupdf_page.get_textpage(flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_CID_FOR_UNKNOWN_UNICODE)
            dct = text_page.extractDICT()
            
            # Get all cells for the page once
            all_page_cells = blocks_to_cells(dct["blocks"], page_height=pymupdf_page.cropbox.height)
            
            for item, level in doc.iterate_items(page_no=page+1, with_groups=False):
                full_text = ""
                if isinstance(item, (TextItem)):
                    for i, provenance_item in enumerate(item.prov):
                        if i != 0:
                            offset = len(full_text)
                            full_text += f" {item.text[provenance_item.charspan[0]:provenance_item.charspan[1]]}"
                            # Update provenance item
                            provenance_item.charspan = (offset + 1, len(full_text))
                            continue

                        if provenance_item.bbox.coord_origin != CoordOrigin.TOPLEFT:
                            prov_rect = provenance_item.bbox.to_top_left_origin(pymupdf_page.cropbox.height)
                        else:
                            prov_rect = provenance_item.bbox

                        orig_text = item.text[provenance_item.charspan[0]:provenance_item.charspan[1]]

                        # OPTIMIZATION: Filter cells based on bounding box instead of cropping textpage
                        cells = self._filter_cells_by_bbox(all_page_cells, prov_rect)

                        original_sanitized_text = sanitize_cells_docling(cells)
                        if orig_text != REPROCESS_FLAG and original_sanitized_text != orig_text:
                            # _log.warning(f"Failed to reconstruct cells (item-width: {item.prov[-1].bbox.width}, item-height: {item.prov[-1].bbox.height}) for (original)\n```{item.text}```\nvs (docling reconstructed)\n```{original_sanitized_text}```")
                            sanitized_text = orig_text
                        else:
                            sanitized_text, median_char_width, last_line_bbox, contains_superscript = sanitize_mineru_cells(cells, ignore_rotated=True)
                            if sanitized_text == "":
                                # _log.warning(f"Deleting {item.text} because it is rotated")
                                deleted_nodes.append(item)
                            provenance_item.media_char_width = median_char_width
                            item.prov[-1].last_line_bbox = last_line_bbox
                            item.prov[-1].contains_superscript = contains_superscript

                        if item.label in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.SECTION_HEADER] and len(cells) == 1:
                            if is_broken_header(sanitized_text):
                                sanitized_text = re.sub(r" (?! )", "", sanitized_text)
                                sanitized_text = re.sub(r" +", " ", sanitized_text)
                                # _log.warning(f"Fixed broken header original {orig_text} -> {sanitized_text}")
                            else:
                                pass
                                # _log.warning(f"Valid header {sanitized_text} not fixed")

                        provenance_item.charspan = (0, len(sanitized_text))
                        full_text += sanitized_text
                    
                    # if item.text != full_text:
                        # _log.warning(f"Sanitized text: {item.text} -> {full_text}")
                    item.text = full_text


        if len(deleted_nodes) > 0:
            doc.delete_items(node_items=list(deleted_nodes))

        return doc

    def _filter_cells_by_bbox(self, all_cells, target_bbox):
        """
        Filter cells that are contained within the target bounding box.
        
        Args:
            all_cells: List of all cells from the page
            target_bbox: The bounding box to filter by
            
        Returns:
            List of cells that are within the target bounding box
        """
        filtered_cells = []
        
        for cell in all_cells:
            cell_bbox = cell.rect.to_bounding_box()
            
            # Check if cell is fully contained within the target bbox
            if (target_bbox.l <= cell_bbox.l and 
                target_bbox.t <= cell_bbox.t and 
                target_bbox.r >= cell_bbox.r and 
                target_bbox.b >= cell_bbox.b):
                filtered_cells.append(cell)
                
        return filtered_cells







                




