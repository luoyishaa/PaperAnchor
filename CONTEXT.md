# PaperAnchor domain language

PaperAnchor is a local research workspace in which a researcher collects papers and asks questions whose answers can be inspected in the original PDF.

## Language

**Research space**:
A named scope containing papers and research activity for one topic. A paper may belong to multiple spaces.
_Avoid_: Folder, project, collection

**Paper**:
A research work represented by one indexed PDF and its metadata. The PDF is the source of truth for quoted evidence.
_Avoid_: Document, file

**Passage**:
A searchable part of a paper tied to a PDF page and rectangle.
_Avoid_: Chunk, paragraph

**Evidence**:
A passage selected for a particular question and supplied to the answer model. Its label, such as `E1`, is valid only within that answer.
_Avoid_: Source, citation

**Citation**:
A reference in an answer to an evidence label, which can open the paper at the selected passage.

**Annotation**:
A note attached to a PDF page or passage, attributed to the researcher or the assistant.

**Discovery candidate**:
External paper metadata found by a scholarly search before its PDF has been imported and indexed.
_Avoid_: Paper (until the PDF enters the library)
