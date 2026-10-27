
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

def load_codiesp_labels(split_path, split_name, top_n=50):
    """Load diagnosis labels and restrict to top-N most frequent codes.
    split_name should be 'train', 'dev', or 'test' -- used to build the correct filename.
    """
    label_file = f"{split_name}D.tsv"
    diag_labels = pd.read_csv(os.path.join(split_path, label_file), sep='\t', header=None)
    diag_labels.columns = ['doc_id', 'code']
    code_counts = diag_labels['code'].value_counts()
    top_codes = set(code_counts.head(top_n).index.tolist())
    return diag_labels, code_counts, top_codes

def build_label_matrix(diag_labels, top_codes):
    """Build multi-label binary matrix for documents with at least one top-N code."""
    filtered = diag_labels[diag_labels['code'].isin(top_codes)]
    valid_doc_ids = filtered['doc_id'].unique()
    doc_to_codes = filtered.groupby('doc_id')['code'].apply(set).to_dict()
    top_list = sorted(top_codes)

    label_matrix = []
    for doc_id in valid_doc_ids:
        row = [1 if code in doc_to_codes[doc_id] else 0 for code in top_list]
        label_matrix.append(row)
    return np.array(label_matrix), valid_doc_ids, top_list

def load_documents(split_path, valid_doc_ids):
    """Load English clinical note text for each valid document."""
    notes_folder = os.path.join(split_path, 'text_files_en')
    documents_text = []
    for doc_id in valid_doc_ids:
        file_path = os.path.join(notes_folder, doc_id + '.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            documents_text.append(f.read())
    return documents_text

def chunk_document(text, tokenizer, chunk_size=128, max_chunks=25):
    """Split long documents into consecutive, non-overlapping token chunks to avoid truncation."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)
        if len(chunks) >= max_chunks:
            break
    return chunks

def pad_chunks(chunks, chunk_size, max_chunks, pad_token_id):
    """Pad all documents to the same number of chunks for batching."""
    padded = []
    for chunk in chunks:
        padded_chunk = chunk + [pad_token_id] * (chunk_size - len(chunk))
        padded.append(padded_chunk)
    while len(padded) < max_chunks:
        padded.append([pad_token_id] * chunk_size)
    return padded
