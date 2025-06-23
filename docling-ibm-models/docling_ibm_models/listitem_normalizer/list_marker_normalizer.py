"""
List Item Marker Normalizer for Docling Documents

This module provides a rule-based model to normalize list item markers
by converting various bullet symbols to standard formats.
"""

import logging
import re
from typing import List, Optional, Tuple, Union

from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    GroupItem,
    GroupLabel,
    ListItem,
    OrderedList,
    ProvenanceItem,
    RefItem,
    TextItem,
    UnorderedList,
)
from docling_core.types.doc.labels import DocItemLabel

_log = logging.getLogger(__name__)


class ListItemMarkerProcessor:
    """
    A rule-based processor for normalizing list item markers.

    This class normalizes various bullet symbols:
    - Bullets and arrows → "-"
    - Checkmarks → "[x]"
    """

    def __init__(self):
        """Initialize the processor with normalization mappings."""
        # Define bullet symbols to normalize to "-"
        self.bullet_symbols = [
            # Various bullet symbols
            '\u0002','\uf0fc',
            '\u2022', '\u2023', '\u25E6', '\u2043', '\u204C', '\u204D', 
            '\u2219', '\u25AA', '\u25AB', '\u25CF', '\u25CB', '\uF0B7', "",
            # Common ASCII and Unicode bullets
            '‣', '⁃',
            # Arrow-like bullets
            '►', '▶', '▸', '‣', '➤', '➢', '›'
        ]
        self.star_symbols = [
            '•', '·',
        ]
        
        # Define checkmark symbols to normalize to "[x]"
        self.checkmark_symbols = ['✓', '✔', '✗', '✘', ""]
        
        # Compile patterns for normalization
        bullet_pattern = '|'.join(rf"(?:\s*[{re.escape(symbol)}-](?:\s+{re.escape(symbol)})*)" for symbol in self.bullet_symbols)
        start_pattern = '|'.join(rf"\s*{re.escape(symbol)}" for symbol in self.star_symbols)
        checkmark_patterns = '|'.join(rf"\s*{re.escape(symbol)}" for symbol in self.checkmark_symbols)
        
        self.bullet_regex = re.compile(f'^({bullet_pattern})')
        self.checkmark_regex = re.compile(f'^({checkmark_patterns})')
        self.star_regex = re.compile(f'^({start_pattern})')

    def _normalize_symbols(self, text: str) -> str:
        """
        Normalize bullet and checkmark symbols in text.
        
        Args:
            text: Input text containing symbols to normalize
            
        Returns:
            Text with normalized symbols
        """
        # Replace checkmarks with [x]
        text = self.checkmark_regex.sub('[x]', text)

        text = self.star_regex.sub('*', text)
        
        # Replace bullets with -
        text = self.bullet_regex.sub('-', text)
        
        return text

    def _normalize_list_item(self, item: TextItem) -> TextItem:
        """
        Normalize a single list item's marker and text.
        
        Args:
            item: ListItem to normalize
            
        Returns:
            Normalized ListItem
        """
        # Normalize the text content
        new_text = self._normalize_symbols(item.text)
        if new_text != item.text:
            _log.warning(f"Updating text: {item.text} to {new_text}")
            # update charsspans (since we truncate from start it's easy just sub)
            len_diff = len(item.text) - len(new_text)
            for prov in item.prov:
                prov.charspan = (prov.charspan[0] - len_diff, prov.charspan[1] - len_diff)
            
            item.text = new_text
        return item

    def process_document(self, doc: DoclingDocument) -> DoclingDocument:
        """
        Process the entire document to normalize list markers.

        Args:
            doc: The DoclingDocument to process

        Returns:
            The processed document (modified in-place)
        """
        _log.info("Starting list marker normalization")
        
        # Process all items in the document
        for item, level in doc.iterate_items(with_groups=False):
            if isinstance(item, TextItem):
                self._normalize_list_item(item)

        return doc