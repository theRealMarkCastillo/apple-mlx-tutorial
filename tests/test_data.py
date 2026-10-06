import json
import subprocess
import sys
from pathlib import Path

from scripts.generate_synthetic_data import generate_lora_chat_data
from scripts.download_datasets import DatasetDownloader

ROOT = Path(__file__).resolve().parent.parent


def test_lora_labels_and_disjoint_prompts(tmp_path):
    train, val = generate_lora_chat_data(output_dir=str(tmp_path))
    splits = [[json.loads(line)['messages'] for line in p.read_text().splitlines()]
              for p in (train, val)]
    assert {r[1]['content'] for r in splits[0]}.isdisjoint({r[1]['content'] for r in splits[1]})
    for rows in splits:
        assert {r[-1]['content'] for r in rows} == {'greeting', 'question', 'command'}
    original = train.read_bytes(), val.read_bytes()
    generate_lora_chat_data(output_dir=str(tmp_path))
    assert original == (train.read_bytes(), val.read_bytes())


def test_samples_respect_data_dir_without_site_packages(tmp_path):
    destination = tmp_path / 'nested/data'
    subprocess.run([sys.executable, '-S', str(ROOT / 'scripts/download_datasets.py'),
                    '--samples', '--data-dir', str(destination)], check=True, capture_output=True)
    assert (destination / 'intent_samples/data.json').is_file()
    assert (destination / 'train.jsonl').is_file()


def test_offline_intents_split_each_class(tmp_path):
    DatasetDownloader(str(tmp_path))._create_snips_fallback()
    train = json.loads((tmp_path / 'snips/train.json').read_text())
    test = json.loads((tmp_path / 'snips/test.json').read_text())
    assert set(train['labels']) == set(test['labels'])
    assert set(train['texts']).isdisjoint(test['texts'])


def test_imdb_subset_shuffles_before_truncating(tmp_path, monkeypatch):
    from datasets import Dataset, DatasetDict
    import scripts.download_datasets as downloader
    # Source ordering must not turn a small download into one-class data.
    rows = Dataset.from_dict({'text': [f'review {i}' for i in range(100)],
                              'label': [0] * 50 + [1] * 50})
    monkeypatch.setattr(downloader, 'load_dataset',
                        lambda *args, **kwargs: DatasetDict(train=rows, test=rows))
    output = DatasetDownloader(str(tmp_path)).download_imdb(max_samples=20)
    for split in ('train', 'test'):
        data = json.loads((output / f'{split}.json').read_text())
        assert len(data['texts']) == 20
        assert set(data['labels']) == {0, 1}
