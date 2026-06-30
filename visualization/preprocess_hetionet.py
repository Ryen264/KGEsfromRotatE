"""Preprocess Hetionet v1.0 into RotatE-style dataset files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import bz2
import json
import os
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def load_hetionet(source_path):
    with bz2.open(source_path, 'rt') as fin:
        data = json.load(fin)
    return data


def build_metaedge_relation_name(source_kind, relation_kind, target_kind):
    return '{}:{}:{}'.format(source_kind, relation_kind, target_kind)


def build_entity_and_relation_maps(data):
    entities = {node['identifier']: idx for idx, node in enumerate(data['nodes'])}
    entity_identifiers = {str(node['identifier']) for node in data['nodes']}

    node_lookup = {
        (node['kind'], str(node['identifier'])): str(node['identifier'])
        for node in data['nodes']
    }

    relations = {}
    for idx, metaedge in enumerate(data['metaedge_tuples']):
        source_kind, target_kind, relation_kind = metaedge[0], metaedge[1], metaedge[2]
        relation_name = build_metaedge_relation_name(
            source_kind, relation_kind, target_kind,
        )
        relations[relation_name] = idx

    relation_kinds = sorted({edge['kind'] for edge in data['edges']})
    return entities, entity_identifiers, node_lookup, relations, relation_kinds


def extract_triples(data, node_lookup):
    triples = []
    for edge in data['edges']:
        source_kind = edge['source_id'][0]
        target_kind = edge['target_id'][0]
        source_key = (source_kind, str(edge['source_id'][1]))
        target_key = (target_kind, str(edge['target_id'][1]))
        head = node_lookup[source_key]
        tail = node_lookup[target_key]
        relation = build_metaedge_relation_name(
            source_kind, edge['kind'], target_kind,
        )
        triples.append((head, relation, tail))
    return triples


def split_triples(triples, valid_ratio, test_ratio, seed):
    if valid_ratio + test_ratio >= 1.0:
        raise ValueError('valid_ratio + test_ratio must be < 1.0')

    rng = random.Random(seed)
    shuffled = list(triples)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(n_total * test_ratio)
    n_valid = int(n_total * valid_ratio)

    test_triples = shuffled[:n_test]
    valid_triples = shuffled[n_test:n_test + n_valid]
    train_triples = shuffled[n_test + n_valid:]
    return train_triples, valid_triples, test_triples


def build_true_tail_map(triples):
    true_tail = defaultdict(set)
    for head, relation, tail in triples:
        true_tail[(head, relation)].add(tail)
    return true_tail


def sample_negative_tail(head, relation, true_tail, entity_list, rng, max_attempts=64):
    for _ in range(max_attempts):
        candidate = entity_list[rng.randrange(len(entity_list))]
        if candidate not in true_tail[(head, relation)]:
            return candidate

    for candidate in entity_list:
        if candidate not in true_tail[(head, relation)]:
            return candidate

    raise RuntimeError(
        'Unable to sample a negative tail for ({}, {})'.format(head, relation)
    )


def build_labeled_triples(triples, all_triples, entity_identifiers, seed):
    true_tail = build_true_tail_map(all_triples)
    entity_list = sorted(entity_identifiers)
    rng = random.Random(seed)
    labeled = []

    for head, relation, tail in triples:
        labeled.append((head, relation, tail, 1))
        negative_tail = sample_negative_tail(
            head, relation, true_tail, entity_list, rng,
        )
        labeled.append((head, relation, negative_tail, 0))

    return labeled


def write_entities(path, entities):
    with open(path, 'w') as fout:
        for identifier, idx in sorted(entities.items(), key=lambda item: item[1]):
            fout.write('{}\t{}\n'.format(idx, identifier))


def write_relations(path, relations):
    with open(path, 'w') as fout:
        for relation, idx in sorted(relations.items(), key=lambda item: item[1]):
            fout.write('{}\t{}\n'.format(idx, relation))


def write_triples(path, triples):
    with open(path, 'w') as fout:
        for head, relation, tail in triples:
            fout.write('{}\t{}\t{}\n'.format(head, relation, tail))


def write_labeled_triples(path, labeled_triples):
    with open(path, 'w') as fout:
        for head, relation, tail, label in labeled_triples:
            fout.write('{}\t{}\t{}\t{}\n'.format(head, relation, tail, label))


def positive_triples_from_labeled(labeled_triples):
    """Return unlabeled true triples (label == 1) from a labeled split."""

    return [
        (head, relation, tail)
        for head, relation, tail, label in labeled_triples
        if int(label) == 1
    ]


def preprocess_hetionet(
    source_path,
    output_dir,
    valid_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    source_path = resolve_path(source_path)
    output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print('Loading Hetionet from {}'.format(source_path))
    data = load_hetionet(source_path)

    entities, entity_identifiers, node_lookup, relations, relation_kinds = (
        build_entity_and_relation_maps(data)
    )
    triples = extract_triples(data, node_lookup)
    print(
        'Entities: {}  Relations (metaedges): {}  Relation kinds: {}  Triples: {}'.format(
            len(entities), len(relations), len(relation_kinds), len(triples),
        )
    )

    train_triples, valid_triples, test_triples = split_triples(
        triples, valid_ratio, test_ratio, seed,
    )
    all_triples = train_triples + valid_triples + test_triples

    print(
        'Split -> train: {}  valid: {}  test: {}  (sum: {})'.format(
            len(train_triples),
            len(valid_triples),
            len(test_triples),
            len(train_triples) + len(valid_triples) + len(test_triples),
        )
    )

    write_entities(os.path.join(output_dir, 'entities.dict'), entities)
    write_relations(os.path.join(output_dir, 'relations.dict'), relations)
    write_triples(os.path.join(output_dir, 'train.txt'), train_triples)

    valid_labeled = build_labeled_triples(
        valid_triples, all_triples, entity_identifiers, seed=seed + 1,
    )
    test_labeled = build_labeled_triples(
        test_triples, all_triples, entity_identifiers, seed=seed + 2,
    )
    write_labeled_triples(os.path.join(output_dir, 'valid_w_label.txt'), valid_labeled)
    write_labeled_triples(os.path.join(output_dir, 'test_w_label.txt'), test_labeled)

    valid_unlabeled = positive_triples_from_labeled(valid_labeled)
    test_unlabeled = positive_triples_from_labeled(test_labeled)
    write_triples(os.path.join(output_dir, 'valid.txt'), valid_unlabeled)
    write_triples(os.path.join(output_dir, 'test.txt'), test_unlabeled)

    print('Wrote files to {}'.format(output_dir))
    return {
        'entities': len(entities),
        'relations': len(relations),
        'relation_kinds': len(relation_kinds),
        'triples': len(triples),
        'train': len(train_triples),
        'valid': len(valid_triples),
        'test': len(test_triples),
        'valid_labeled': len(valid_labeled),
        'test_labeled': len(test_labeled),
    }


def parse_cli():
    parser = argparse.ArgumentParser(
        description='Preprocess Hetionet v1.0 into train and labeled valid/test splits.',
    )
    parser.add_argument(
        '--input',
        default='data/hetionet/hetionet-v1.0.json.bz2',
        help='Path to hetionet-v1.0.json.bz2',
    )
    parser.add_argument(
        '--output-dir',
        default='data/hetionet',
        help='Directory for train/valid/test splits, labeled valid/test, entities.dict, relations.dict',
    )
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_cli()
    stats = preprocess_hetionet(
        source_path=args.input,
        output_dir=args.output_dir,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(stats)


if __name__ == '__main__':
    main()
