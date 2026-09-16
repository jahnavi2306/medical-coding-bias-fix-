import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer


def load_codiesp_labels(split_path, split_name, top_n=50):
    """Load diagnosis labels and restrict to the top-N most frequent codes.
    split_name is 'train', 'dev', or 'test' -- used to build the filename (e.g. trainD.tsv).
    """
    label_file = f"{split_name}D.tsv"
    diag_labels = pd.read_csv(os.path.join(split_path, label_file), sep='\t', header=None)
    diag_labels.columns = ['doc_id', 'code']
    code_counts = diag_labels['code'].value_counts()
    top_codes = set(code_counts.head(top_n).index.tolist())
    return diag_labels, code_counts, top_codes


def build_label_matrix(diag_labels, top_codes):
    """Build a multi-hot label matrix for documents with at least one top-N code."""
    filtered = diag_labels[diag_labels['code'].isin(top_codes)]
    valid_doc_ids = filtered['doc_id'].unique()
    doc_to_codes = filtered.groupby('doc_id')['code'].apply(set).to_dict()
    top_list = sorted(top_codes)

    label_matrix = np.array([
        [1 if code in doc_to_codes[doc_id] else 0 for code in top_list]
        for doc_id in valid_doc_ids
    ])
    return label_matrix, valid_doc_ids, top_list


def load_documents(split_path, valid_doc_ids):
    """Load English clinical note text for each valid document."""
    notes_folder = os.path.join(split_path, 'text_files_en')
    documents_text = []
    for doc_id in valid_doc_ids:
        with open(os.path.join(notes_folder, doc_id + '.txt'), 'r', encoding='utf-8') as f:
            documents_text.append(f.read())
    return documents_text


def chunk_document(text, tokenizer, chunk_size=128, max_chunks=10):
    """Split a document into overlapping token chunks instead of truncating at 512."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunks.append(tokens[i:i + chunk_size])
        if len(chunks) >= max_chunks:
            break
    return chunks


def pad_chunks(chunks, chunk_size, max_chunks, pad_token_id):
    """Pad every document to the same (max_chunks, chunk_size) shape for batching."""
    padded = [chunk + [pad_token_id] * (chunk_size - len(chunk)) for chunk in chunks]
    while len(padded) < max_chunks:
        padded.append([pad_token_id] * chunk_size)
    return padded


def preprocess_split(split_path, split_name, tokenizer, top_codes=None, top_n=50,
                      chunk_size=128, max_chunks=10):
    """End-to-end: labels -> label matrix -> text -> chunked/padded token ids.
    Pass `top_codes` (from the train split) when processing dev/test, so all
    splits share the exact same 50-code label space.
    """
    diag_labels, code_counts, inferred_top_codes = load_codiesp_labels(split_path, split_name, top_n)
    codes_to_use = top_codes if top_codes is not None else inferred_top_codes

    label_matrix, valid_doc_ids, top_list = build_label_matrix(diag_labels, codes_to_use)
    documents_text = load_documents(split_path, valid_doc_ids)

    pad_id = tokenizer.pad_token_id
    input_ids = np.array([
        pad_chunks(chunk_document(doc, tokenizer, chunk_size, max_chunks), chunk_size, max_chunks, pad_id)
        for doc in documents_text
    ])

    return {
        'input_ids': input_ids,
        'labels': label_matrix,
        'doc_ids': list(valid_doc_ids),
        'top_codes': top_list,
    }, code_counts
