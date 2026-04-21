import os
import sys
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.file import PptxReader, PDFReader, MarkdownReader 
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter

from llama_index.core import Settings

# Set the global ceiling to your maximum expected chunk size
Settings.chunk_size = 8192 
Settings.chunk_overlap = 100


def load_xlsx_with_stdlib(calendar_path):
    """Parse a simple .xlsx workbook without requiring openpyxl."""
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    def column_index(cell_ref):
        letters = "".join(ch for ch in cell_ref if ch.isalpha())
        index = 0
        for letter in letters:
            index = (index * 26) + (ord(letter.upper()) - ord("A") + 1)
        return index - 1

    def excel_serial_to_datetime(value):
        base = pd.Timestamp("1899-12-30")
        return base + pd.to_timedelta(float(value), unit="D")

    with zipfile.ZipFile(calendar_path) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("x:si", namespace):
                text = "".join(node.text or "" for node in item.iterfind(".//x:t", namespace))
                shared_strings.append(text)

        sheet_root = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows = []
        max_columns = 0

        for row in sheet_root.findall(".//x:sheetData/x:row", namespace):
            values = {}
            for cell in row.findall("x:c", namespace):
                cell_ref = cell.attrib.get("r", "")
                col_idx = column_index(cell_ref)
                value_node = cell.find("x:v", namespace)
                cell_type = cell.attrib.get("t")

                if value_node is None:
                    cell_value = ""
                elif cell_type == "s":
                    cell_value = shared_strings[int(value_node.text)]
                else:
                    raw_value = value_node.text or ""
                    try:
                        numeric_value = float(raw_value)
                        if numeric_value.is_integer():
                            numeric_value = int(numeric_value)
                        cell_value = numeric_value
                    except ValueError:
                        cell_value = raw_value

                values[col_idx] = cell_value
                max_columns = max(max_columns, col_idx + 1)

            if values:
                rows.append([values.get(i, "") for i in range(max_columns)])

    if not rows:
        return pd.DataFrame()

    header = [str(value).strip() for value in rows[0]]
    data_rows = rows[1:]
    calendar_df = pd.DataFrame(data_rows, columns=header)

    if "date" in calendar_df.columns:
        calendar_df["date"] = calendar_df["date"].apply(
            lambda value: excel_serial_to_datetime(value)
            if isinstance(value, (int, float)) and value != ""
            else value
        )

    return calendar_df


def load_calendar_table(calendar_path):
    """Load the course calendar whether it is CSV text or an Excel file with a wrong suffix."""
    file_signature = Path(calendar_path).read_bytes()[:4]

    # Excel .xlsx files are zip containers and begin with PK.
    if file_signature[:2] == b"PK":
        try:
            return pd.read_excel(calendar_path)
        except ImportError:
            return load_xlsx_with_stdlib(calendar_path)

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None
    for encoding in encodings_to_try:
        try:
            return pd.read_csv(calendar_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding if last_error else "utf-8",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 1,
        (
            f"Could not decode calendar file at {calendar_path}. "
            "Tried UTF-8, UTF-8-SIG, CP1252, and Latin-1."
        ),
    )


def material_handler(data_dir, calendar_map, material_type):
    all_nodes = []

    if not os.path.isdir(data_dir):
        print(f"Skipping missing folder: {data_dir}")
        return all_nodes

    # readers for lecture slides, handouts and textbook, and tutorials
    pptx_reader = PptxReader()
    pdf_reader = PDFReader()
    md_reader = MarkdownReader()
    
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        # Determine which reader to use based on extension
        if filename.endswith(".pptx"):
            reader = pptx_reader
        elif filename.endswith(".pdf"):
            reader = pdf_reader
        elif filename.endswith(".md"):
            reader = md_reader
        else:
            reader = SimpleDirectoryReader(input_files=[file_path])
            
        meta = calendar_map.get(filename, {})
        if material_type == "lectures":
            extra_info = {
                "source": meta.get('source', material_type), 
                "date": str(meta.get('date', 'unknown')),
                "topic": meta.get('topic', Path(filename).stem),
                "lecture_id": meta.get('lecture_id', Path(filename).stem), 
                "file_type": "PowerPoint" if filename.endswith(".pptx") else "PDF",
                "priority": 2
            }
        elif material_type == "labs":
            extra_info = {
                "source": meta.get('source', material_type), 
                "date": str(meta.get('date', 'unknown')),
                "topic": meta.get('topic', Path(filename).stem),
                "lab_id": meta.get('lab_id', Path(filename).stem), 
                "file_type": "PowerPoint" if filename.endswith(".pptx") else "PDF"
            }
        elif material_type == "tutorials":
            extra_info = {
                "source": meta.get('source', material_type), 
                "date": str(meta.get('date', 'unknown')),
                "topic": meta.get('topic', Path(filename).stem),
                "tutorial_id": meta.get('tutorial_id', Path(filename).stem), 
                "file_type": "Markdown" if filename.endswith(".md") else "Text"
            }
        elif material_type == "code":
            extra_info = {
                "source": meta.get('source', material_type), 
                "date": str(meta.get('date', 'unknown')),
                "topic": meta.get('topic', Path(filename).stem),
                "lecture_id": meta.get('lecture_id', Path(filename).stem), 
                "file_type": "C++" if filename.endswith(".cpp") else "Text"
            }
        elif material_type == "textbook":
            raw_topics = meta.get('topic', Path(filename).stem)
            topic_list = [t.strip() for t in str(raw_topics).split(',')]
            extra_info = {
                "source": meta.get('source', material_type), 
                "topic": topic_list,
                "file_type": "PDF",
                "priority": 1
            }
        else:
            continue

        print(f"Ingesting {filename} ...")
        documents = reader.load_data(file_path)

        # 1. "Personalities"
        textbook_parser = TokenTextSplitter(chunk_size=800, chunk_overlap=100)
        tutorial_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        for doc in documents:
            doc.metadata = {}
            doc.metadata = extra_info
            doc.metadata["source_file"] = filename 
            doc.metadata["page_label"] = doc.metadata.get("page_label", "N/A")
            doc.metadata.update(extra_info)
            print(f"TEXT LENGTH: { (int) (len(doc.text) / 4) } tokens")
            
            doc.excluded_llm_metadata_keys = [] # Let the LLM see everything else (date, topic)
            doc.excluded_embed_metadata_keys = ["file_name", "source_file"] # Don't waste vector space on filename

            doc.metadata_template = "{key}: {value}"
            doc.text_template = "Metadata: {metadata_str}\n\nContent: {content}"

            if material_type == "lectures":
                nodes = [doc]
            elif material_type == "tutorials":
                nodes = tutorial_parser.get_nodes_from_documents([doc])
            elif material_type == "textbook":
                if filename == "chapter_03.pdf":
                    high_chunk_textbook_parser = TokenTextSplitter(chunk_size=8192, chunk_overlap=100)
                    nodes = high_chunk_textbook_parser.get_nodes_from_documents([doc])
                else:
                    nodes = textbook_parser.get_nodes_from_documents([doc])
            else:
                nodes = tutorial_parser.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)
            # if len(nodes) > 0 and filename == "chapter_03.pdf":
            #     sample_node = nodes[0]
            #     print(f"\n--- DEBUG: Verifying {filename} ---")
            #     # This shows what the LLM actually receives
            #     print("LLM VIEW:\n", sample_node.get_content(metadata_mode="llm"))
            #     print("-" * 30)
            #     # --------------------
    return all_nodes
            
def build_course_index(data_dir="./materials", calendar_file="master_calendar.csv"):

    calendar_path = os.path.join(data_dir, calendar_file)  
    calendar_df = load_calendar_table(calendar_path)
    calendar_df.columns = calendar_df.columns.str.strip()
    if "primary_file" in calendar_df.columns:
        calendar_map = calendar_df.set_index('primary_file').to_dict('index')
    else:
        print(
            "Calendar file does not include 'primary_file'; "
            "ingesting with fallback metadata derived from filenames."
        )
        calendar_map = {}

    all_nodes = []

    material_types = ["lectures", "labs", "tutorials", "code", "textbook"]
    discovered_materials = [
        material for material in material_types
        if os.path.isdir(os.path.join(data_dir, material))
    ]

    if discovered_materials:
        material_sources = [(material, os.path.join(data_dir, material)) for material in discovered_materials]
    else:
        print(
            "No standard material subfolders found; "
            "treating files in the top-level directory as lecture materials."
        )
        material_sources = [("lectures", data_dir)]

    for material, this_data_dir in material_sources: 
        print (f"Ingesting {material} ...") 
        nodes = material_handler(this_data_dir, calendar_map, material)
        all_nodes.extend(nodes) 
        
    index = VectorStoreIndex.from_documents(all_nodes)
    index.storage_context.persist(persist_dir="./storage")
    return index



if len(sys.argv) < 2:
    print(f"No path to course materials provided")
    print(f"Usage: ")
    print(f"\t python ingest.py [course_material_folder]\n")
else: 
    path = Path(sys.argv[1])
    if path.exists() and path.is_dir():
        build_course_index(path)
    else:
        print(f"Course material folder not found. Please try again")

