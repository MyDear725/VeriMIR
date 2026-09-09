import hashlib
import json
from pathlib import Path
import pytest
from verimir.captions import audit_training_captions


def fixture_files(tmp_path):
    records = [dict(sample_id=i, source_id=f'{i}.jpg', label=0) for i in range(3)]
    manifest = {'dataset':'synthetic','splits':{'train':records[:1], 'validation':records[1:2], 'test':records[2:]}}
    captions = {'dataset':'synthetic','split':'train-only','num_captions':1,
                'captions':[{'sample_id':0,'source_id':'0.jpg','caption':'A synthetic lesion description.'}]}
    m,c = tmp_path/'manifest.json',tmp_path/'captions.json'
    m.write_text(json.dumps(manifest)); c.write_text(json.dumps(captions))
    return m,c,manifest,captions


def test_valid_membership_with_historical_manifest_hash(tmp_path):
    m,c,_,_=fixture_files(tmp_path)
    report=audit_training_captions(c,m)
    assert report['matches_current_train_split']
    assert not report['recorded_manifest_hash_matches_current']


def test_evaluation_caption_is_rejected(tmp_path):
    m,c,_,payload=fixture_files(tmp_path)
    payload['captions'][0]['sample_id']=2
    payload['captions'][0]['source_id']='2.jpg'
    c.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='train split'):
        audit_training_captions(c,m)


def test_overlapping_manifest_is_rejected(tmp_path):
    m,c,manifest,_=fixture_files(tmp_path)
    manifest['splits']['test']=manifest['splits']['train']
    m.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='overlap'):
        audit_training_captions(c,m)


def test_sensitive_text_is_rejected(tmp_path):
    m,c,_,payload=fixture_files(tmp_path)
    payload['captions'][0]['caption']='Contact somebody@example.org'
    c.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='sensitive'):
        audit_training_captions(c,m)


def test_release_json_files_match_audited_bytes_and_counts():
    root=Path(__file__).resolve().parents[1]
    audit=json.loads((root/'docs/CAPTION_AUDIT.json').read_text())
    for item in audit['datasets']:
        raw=(root/'data/captions'/f"{item['dataset']}_train_only.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest()==item['caption_json_sha256']
        data=json.loads(raw)
        assert len(data['captions'])==item['num_captions']==data['num_captions']
        assert data['class_name_prepended'] is True
