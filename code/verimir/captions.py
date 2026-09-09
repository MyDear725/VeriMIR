"""Validate raw training captions against a user-supplied frozen manifest."""
import hashlib
import json
from pathlib import Path
import re


def audit_training_captions(caption_path, manifest_path):
    caption_path, manifest_path = Path(caption_path), Path(manifest_path)
    raw = caption_path.read_bytes()
    payload = json.loads(raw)
    manifest = json.loads(manifest_path.read_bytes())
    if payload.get('split') != 'train-only':
        raise ValueError('Caption file must be explicitly train-only')
    if payload.get('dataset') != manifest.get('dataset'):
        raise ValueError('Dataset mismatch')
    rows = payload['captions']
    if len(rows) != payload['num_captions']:
        raise ValueError('Caption count mismatch')
    pairs = [(int(r['sample_id']), r['source_id']) for r in rows]
    if len({p[0] for p in pairs}) != len(rows) or len({p[1] for p in pairs}) != len(rows):
        raise ValueError('Duplicate caption ID')
    expected = {(int(r['sample_id']),r['source_id']) for r in manifest['splits']['train']}
    if set(pairs) != expected:
        raise ValueError('Caption IDs/source IDs must match the entire frozen train split')
    forbidden = [r for name in ('validation','test') for r in manifest['splits'][name]]
    leaked_ids = {p[0] for p in pairs} & {int(r['sample_id']) for r in forbidden}
    leaked_sources = {p[1] for p in pairs} & {r['source_id'] for r in forbidden}
    if leaked_ids or leaked_sources:
        raise ValueError('Validation/test caption overlap')
    patterns = {
        'email':r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        'absolute_local_path':r'(?:/home/|/sharefiles\w*/|[A-Za-z]:[\\/])',
        'credential':r'(?:sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|BEGIN [A-Z ]*PRIVATE KEY)',
        'ipv4':r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        'explicit_identifier':r'(?i)\b(?:patient name|patient id|medical record number|date of birth)\s*[:=]',
    }
    counts = {name:0 for name in patterns}
    for row in rows:
        if not isinstance(row.get('caption'),str) or not row['caption'].strip():
            raise ValueError('Missing caption text')
        if set(row) != {'sample_id','source_id','caption'}:
            raise ValueError('Unexpected metadata fields; review before publishing')
        text = json.dumps(row,ensure_ascii=False)
        for name,pattern in patterns.items():
            counts[name] += bool(re.search(pattern,text))
    if any(counts.values()):
        raise ValueError(f'Potential sensitive data requires manual review: {counts}')
    current_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {
        'dataset':payload['dataset'], 'num_captions':len(rows),
        'caption_json_sha256':hashlib.sha256(raw).hexdigest(),
        'matches_current_train_split':True,
        'validation_test_id_overlap':0,'validation_test_source_overlap':0,
        'class_name_prepended':payload.get('class_name_prepended'),
        'recorded_manifest_sha256':payload.get('manifest_sha256'),
        'current_manifest_sha256':current_hash,
        'recorded_manifest_hash_matches_current':payload.get('manifest_sha256')==current_hash,
        'sensitive_pattern_counts':counts,
        'privacy_scope':'Pattern scan only, not a comprehensive de-identification or licensing certificate.',
    }
