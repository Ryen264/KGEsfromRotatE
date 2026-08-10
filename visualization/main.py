import argparse
import json
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'codes'))

import run as train_run
from loss import (
    KGAULoss,
    KGmAULoss,
    KGmAmULoss,
    compute_kge_loss,
    UniGammaController,
    build_training_optimizer,
    is_learnable_kgau_gammas,
    update_kgau_gamma_schedule,
)
from metrics.classification import classification_metrics_from_probs
from model import KGEModel, resolve_rotate_score_mode
from strategy import get_strategy, resolve_strategy_name

DEFAULT_UNIFORM_KEYS = ['query', 'target', 'entity', 'relation']
DEFAULT_UNIFORM_T = 4

UNIFORM_COLORS = {
    'query': '#1f77b4',    # blue
    'target': '#2ca02c',   # green
    'entity': '#9467bd',   # purple
    'relation': '#d62728', # red
    'head': '#8c564b',     # brown (secondary)
    'tail': '#bcbd22',     # olive (secondary)
}

LOSS_DISPLAY_NAMES = {
    'se': 'SE',
    'hinge': 'Hinge',
    'bce': 'BCE',
    'mr': 'MR',
    'bpr': 'BPR',
    'ce': 'CE',
    'sans': 'SA',
    'kgau': 'KGAU',
    'kgmau': 'KGmAU',
    'kgmamu': 'KGmAmU',
}


def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return '{}h {}m {}s'.format(hours, minutes, secs)
    if minutes:
        return '{}m {}s'.format(minutes, secs)
    return '{:.2f}s'.format(seconds)


def format_metric_value(value, digits=4):
    if value is None:
        return 'N/A'
    return '{:.{}f}'.format(float(value), digits)


def find_best_valid(history, valid_metric='MRR'):
    metric_values = [metrics[valid_metric] for metrics in history['valid_metric']]
    best_idx = int(np.argmax(metric_values))
    return history['epochs'][best_idx], metric_values[best_idx]


def clone_model_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def evaluate_triple_classification(model, test_triples, args):
    model.eval()
    sample = []
    y_true = []
    for head, relation, tail in test_triples:
        for candidate_region in args.regions:
            y_true.append(1 if candidate_region == tail else 0)
            sample.append((head, relation, candidate_region))

    sample = torch.LongTensor(sample)
    if args.cuda:
        sample = sample.cuda()

    with torch.no_grad():
        y_score = model(sample).squeeze(1).cpu().numpy()

    return classification_metrics_from_probs(np.array(y_true), y_score)


def dataset_display_name(data_path):
    name = os.path.basename(str(data_path or '').rstrip('/\\'))
    aliases = {
        'wn18rr': 'WN18RR',
        'wn18': 'WN18',
        'fb15k_237': 'FB15k-237',
        'fb15k-237': 'FB15k-237',  # legacy alias
        'fb15k': 'FB15k',
        'yago3_10': 'YAGO3-10',
        'yago3-10': 'YAGO3-10',  # legacy alias
        'countries_s1': 'Countries_S1',
        'countries_s2': 'Countries_S2',
        'countries_s3': 'Countries_S3',
    }
    return aliases.get(name.lower(), name or 'N/A')


def negative_samples_for_report(args):
    '''NegSamp: N; AllNeg: nentity; KGAU: 0.'''
    return candidates_per_positive(args)


def chunk_size_for_report(args):
    '''NegSamp: negative_chunk_size; KGAU: uniform_pair_chunk_size; AllNeg: None.'''
    strategy = getattr(args, 'strategy', 'uniform')
    loss = getattr(args, 'loss', '')
    if strategy in ('uniform', 'bernoulli', 'selfadv'):
        return str(int(getattr(args, 'negative_chunk_size', 256) or 0))
    if strategy == 'kgau' or loss in ('kgau', 'kgmau', 'kgmamu'):
        return str(int(getattr(args, 'uniform_pair_chunk_size', 256) or 0))
    return None


def steady_time_per_epoch(timing, num_epochs):
    '''Mean train-only epoch time; skip first epoch as allocator/cudnn warmup.'''
    train_epoch_times = timing.get('train_epoch_times') or []
    if len(train_epoch_times) > 1:
        steady = train_epoch_times[1:]
        return float(sum(steady) / len(steady))
    if train_epoch_times:
        return float(train_epoch_times[0])
    return float(timing.get('train_time', 0.0)) / max(num_epochs, 1)


def build_results_report(
    num_epochs,
    valid_metric,
    link_metrics,
    classification_metrics_dict,
    best_epoch,
    best_valid_value,
    timing,
    model_name,
    loss_name,
    strategy_name,
    dim,
    batch_size,
    config_path=None,
    dataset_name=None,
    negative_samples=None,
    args=None,
):
    mr = format_metric_value(link_metrics.get('MR') if link_metrics else None)
    mrr = format_metric_value(link_metrics.get('MRR') if link_metrics else None)
    hit_1 = format_metric_value(link_metrics.get('HITS@1') if link_metrics else None)
    hit_3 = format_metric_value(link_metrics.get('HITS@3') if link_metrics else None)
    hit_10 = format_metric_value(link_metrics.get('HITS@10') if link_metrics else None)

    acc = format_metric_value(classification_metrics_dict.get('accuracy') if classification_metrics_dict else None)
    prec = format_metric_value(classification_metrics_dict.get('precision') if classification_metrics_dict else None)
    rec = format_metric_value(classification_metrics_dict.get('recall') if classification_metrics_dict else None)
    f1 = format_metric_value(classification_metrics_dict.get('f1') if classification_metrics_dict else None)
    pr_auc = format_metric_value(classification_metrics_dict.get('pr_auc') if classification_metrics_dict else None)
    roc_auc = format_metric_value(classification_metrics_dict.get('roc_auc') if classification_metrics_dict else None)

    best_epoch = int(best_epoch)
    best_valid_value = format_metric_value(best_valid_value)

    train_time = format_duration(timing['train_time'])
    valid_time = format_duration(timing['valid_time'])
    test_time = format_duration(timing['test_time'])
    total_time = format_duration(timing['total_time'])

    time_per_epoch = format_duration(steady_time_per_epoch(timing, num_epochs))

    peak_value = timing.get('train_peak_gpu_memory_gb')
    if peak_value is None:
        peak_value = timing.get('peak_gpu_memory_gb')
    peak_gpu_memory = (
        None if peak_value is None else '{:.2f} GB'.format(float(peak_value))
    )
    reserved_value = timing.get('train_peak_gpu_memory_reserved_gb')
    peak_gpu_memory_reserved = (
        None if reserved_value is None else '{:.2f} GB'.format(float(reserved_value))
    )

    chunk_size = chunk_size_for_report(args) if args is not None else None

    lines = [
        'Config Path: {}'.format(config_path or 'N/A'),
        'Model: {}'.format(model_name),
        'Dataset: {}'.format(dataset_name or 'N/A'),
        'Loss: {}'.format(loss_name),
        'Strategy: {}'.format(strategy_name),
        'Dim: {}'.format(dim),
        'Batch Size: {}'.format(batch_size),
        'Epochs: {}'.format(num_epochs),
        'Negative Samples: {}'.format(
            0 if negative_samples is None else negative_samples
        ),
    ]
    if chunk_size is not None:
        lines.append('Chunk Size: {}'.format(chunk_size))

    loss_name_key = getattr(args, 'loss', '') if args is not None else ''
    if loss_name_key in ('kgau', 'kgmau', 'kgmamu') or str(strategy_name).lower() == 'kgau':
        uniform_t = getattr(args, 'uniform_t', DEFAULT_UNIFORM_T) if args is not None else DEFAULT_UNIFORM_T
        align_alpha = getattr(args, 'align_alpha', 2.0) if args is not None else 2.0
        gamma_q = getattr(args, 'uniform_gamma_q', 0.0) if args is not None else 0.0
        gamma_y = getattr(args, 'uniform_gamma_y', 0.0) if args is not None else 0.0
        gamma_e = getattr(args, 'uniform_gamma_e', 0.0) if args is not None else 0.0
        lines += [
            'Align alpha: {}'.format(align_alpha),
            'Uniform t: {}'.format(uniform_t),
            '(Gamma Query, Gamma Target, Gamma Entity): ({}, {}, {})'.format(
                gamma_q, gamma_y, gamma_e,
            ),
        ]

    lines.append('')

    if mrr is not None:
        lines += [
            'MR: {}'.format(mr),
            'MRR: {}'.format(mrr),
            'Hit@1: {}'.format(hit_1),
            'Hit@3: {}'.format(hit_3),
            'Hit@10: {}'.format(hit_10),
            ''
        ]

    if acc is not None and acc != 'N/A':
        lines += [
            'Acc: {}'.format(acc),
            'Prec: {}'.format(prec),
            'Rec: {}'.format(rec),
            'F1: {}'.format(f1),
            'PR-AUC: {}'.format(pr_auc),
            'ROC-AUC: {}'.format(roc_auc),
            '',
        ]

    lines += [
        'Best Epoch: {}'.format(best_epoch),
        'Best {}: {}'.format(valid_metric, best_valid_value),
        'Test checkpoint: best-valid (epoch {})'.format(best_epoch),
        '',
        'Training: {}'.format(train_time),
        'Valid: {}'.format(valid_time),
        'Test: {}'.format(test_time),
        'Total: {}'.format(total_time),
        ''
    ]

    lines.append('Time per epoch: {}'.format(time_per_epoch))
    if peak_gpu_memory is not None:
        lines.append('Peak GPU memory: {}'.format(peak_gpu_memory))
    if peak_gpu_memory_reserved is not None:
        lines.append('Peak GPU memory reserved: {}'.format(peak_gpu_memory_reserved))
    return '\n'.join(lines) + '\n'


def write_results_report(output_dir, report_text):
    results_path = os.path.join(output_dir, 'results.txt')
    with open(results_path, 'w') as fout:
        fout.write(report_text)
    print('Results saved to {}'.format(results_path))
    return results_path


def _sync_cuda(args):
    if getattr(args, 'cuda', False) and torch.cuda.is_available():
        torch.cuda.synchronize()


def candidates_per_positive(args):
    strategy = getattr(args, 'strategy', 'uniform')
    if strategy in ('1vsall', 'kvsall'):
        return int(getattr(args, 'nentity', 0) or 0)
    if strategy in ('uniform', 'bernoulli', 'selfadv'):
        return int(getattr(args, 'negative_sample_size', 0) or 0)
    return 0


def run_post_training_evaluation(model, args, test_triples, all_true_triples):
    if args.countries:
        return None, evaluate_triple_classification(model, test_triples, args)

    return KGEModel.test_step(model, test_triples, all_true_triples, args), None


def get_loss_display_name(args):
    loss_name = getattr(args, 'loss', 'sans')
    return LOSS_DISPLAY_NAMES.get(loss_name, loss_name.upper())


def format_training_postfix(valid_metric, metric_value, loss_name, loss_value, align, uniform):
    return '{metric}={metric_v:.4f}, {loss_name}={loss_v:.4f}, align={align:.4f}, uniform={uniform:.4f}'.format(
        metric=valid_metric,
        metric_v=metric_value,
        loss_name=loss_name,
        loss_v=loss_value,
        align=align,
        uniform=uniform,
    )


def load_config(config_path):
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config, config_path


def resolve_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def build_output_dir(config_path):
    config_path = resolve_path(config_path)
    config_stem = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_name = '{}_{}'.format(config_stem, timestamp)
    return os.path.join(ROOT, 'visualization', 'outputs', folder_name)


def default_save_path(config, save_id=0):
    dataset = os.path.basename(config['data_path'].rstrip('/'))
    return os.path.join(ROOT, 'models', '{}_{}_{}'.format(config['model'], dataset, save_id))


def build_args(config):
    args = train_run.parse_args([])

    for key, value in config.items():
        setattr(args, key, value)

    args.data_path = resolve_path(args.data_path)
    args.save_path = resolve_path(config.get('save_path') or default_save_path(config, 0))
    args.init_checkpoint = resolve_path(config.get('init_checkpoint'))
    args.cuda = torch.cuda.is_available()
    args.do_train = True
    args.do_valid = True
    args.do_test = False
    args.cpu_num = config.get('cpu_num', 4)
    # Resolve strategy defaults (sans -> selfadv, else uniform)
    args.strategy = resolve_strategy_name(args)
    # Configs omit unused keys; clear misleading argparse defaults on args.
    if args.strategy in ('1vsall', 'kvsall', 'kgau') and 'negative_sample_size' not in config:
        args.negative_sample_size = 0
    if args.strategy not in ('uniform', 'bernoulli', 'selfadv') and 'negative_chunk_size' not in config:
        args.negative_chunk_size = 0
    get_strategy(args)

    return args


def resolve_num_epochs(config):
    return int(config['epochs'])


def load_dataset(args):
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = {}
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = {}
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    if args.countries:
        regions = []
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                regions.append(entity2id[line.strip()])
        args.regions = regions

    args.nentity = len(entity2id)
    args.nrelation = len(relation2id)

    train_triples = train_run.read_triple(
        os.path.join(args.data_path, 'train.txt'), entity2id, relation2id
    )
    valid_triples = train_run.read_triple(
        os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id
    )
    test_triples = train_run.read_triple(
        os.path.join(args.data_path, 'test.txt'), entity2id, relation2id
    )
    all_true_triples = train_triples + valid_triples + test_triples
    return train_triples, valid_triples, test_triples, all_true_triples


def build_model_and_iterator(args, train_triples):
    score_mode = (
        resolve_rotate_score_mode(args) if args.model == 'RotatE' else 'distance'
    )
    model = KGEModel(
        model_name=args.model,
        nentity=args.nentity,
        nrelation=args.nrelation,
        dim=args.dim,
        margin_gamma=args.margin_gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        score_mode=score_mode,
    )
    if args.cuda:
        model = model.cuda()

    if is_learnable_kgau_gammas(args):
        UniGammaController(args).ensure_model_params(model)

    strategy = get_strategy(args)
    train_iterator = strategy.build_train_iterator(
        train_triples, args.nentity, args.nrelation
    )
    optimizer = build_training_optimizer(model, args)
    return model, train_iterator, optimizer


def validate_uniform_sets(uniform_sets):
    if not uniform_sets:
        raise ValueError('uniform_sets must be non-empty')
    allowed = set(DEFAULT_UNIFORM_KEYS)
    unknown = [key for key in uniform_sets if key not in allowed]
    if unknown:
        raise ValueError(
            'Unknown uniform_sets: {}. Allowed: {}'.format(
                unknown, ', '.join(DEFAULT_UNIFORM_KEYS)
            )
        )
    return list(uniform_sets)


def get_kgau_family_loss(args):
    loss_name = getattr(args, 'loss', 'kgau')
    if loss_name == 'kgmau':
        return KGmAULoss(args)
    if loss_name == 'kgmamu':
        return KGmAmULoss(args)
    return KGAULoss(args)


def compute_au_metrics(model, positive_sample, mode, args, uniform_sets):
    loss_fn = get_kgau_family_loss(args)
    loss_name = getattr(args, 'loss', 'kgau')
    head = model.entity_embedding[positive_sample[:, 0]]
    relation = model.relation_embedding[positive_sample[:, 1]]
    tail = model.entity_embedding[positive_sample[:, 2]]

    query_e = model.query_encoder(head, relation, tail, mode=mode)
    target_e = model.target_encoder(tail, head=head, relation=relation, mode=mode)

    align_margin = getattr(args, 'align_margin', 0.0)
    align_alpha = getattr(args, 'align_alpha', 2.0)
    uniform_margin = getattr(args, 'uniform_margin', 2.0)
    uniform_t = DEFAULT_UNIFORM_T

    if loss_name in ('kgmau', 'kgmamu'):
        align_loss = loss_fn.margin_alignment(
            query_e, target_e, align_margin=align_margin,
        ).item()
    else:
        align_loss = loss_fn.alignment(
            query_e, target_e, align_alpha=align_alpha,
        ).item()

    uniform_components = {}
    embeddings = {
        'query': query_e,
        'target': target_e,
        'head': head,   # already embedded by entity_embedding
        'tail': tail,   # already embedded by entity_embedding
        'entity': torch.cat([head, tail], dim=0),   # already embedded by entity_embedding
        'relation': relation,   # already embedded by relation_embedding
    }

    for key in uniform_sets:
        pair_chunk = int(getattr(args, 'uniform_pair_chunk_size', 0) or 0)
        if loss_name == 'kgmamu':
            uniform_components[key] = loss_fn.margin_uniformity(
                embeddings[key],
                uniform_margin=uniform_margin,
                uniform_t=uniform_t,
                pair_chunk_size=pair_chunk,
            ).item()
        else:
            uniform_components[key] = loss_fn.uniformity(
                embeddings[key],
                uniform_t=uniform_t,
                pair_chunk_size=pair_chunk,
            ).item()

    uniform_loss = float(np.mean(list(uniform_components.values())))
    return align_loss, uniform_components, uniform_loss


def train_step_with_metrics(model, optimizer, train_iterator, args, uniform_sets):
    model.train()
    optimizer.zero_grad()

    strategy = get_strategy(args)
    batch = next(train_iterator)
    (
        positive_score, negative_score, subsampling_weight,
        positive_sample, mode, negative_weights, scores, labels,
    ) = strategy.prepare_train_batch(batch, model)
    loss, log = compute_kge_loss(
        positive_score, negative_score, subsampling_weight, model, args,
        positive_sample=positive_sample, mode=mode,
        negative_weights=negative_weights,
        scores=scores, labels=labels,
    )
    loss.backward()
    optimizer.step()

    if is_learnable_kgau_gammas(args):
        UniGammaController(args).clamp_log_gammas(model)

    align_loss, uniform_components, uniform_loss = compute_au_metrics(
        model, positive_sample, mode, args, uniform_sets,
    )
    log['align_loss'] = align_loss
    log['uniform_loss'] = uniform_loss
    log['uniform'] = uniform_components
    return log


def train_and_collect_history(args, num_epochs, valid_metric='MRR', uniform_sets=None):
    uniform_sets = validate_uniform_sets(uniform_sets or DEFAULT_UNIFORM_KEYS)
    train_triples, valid_triples, test_triples, all_true_triples = load_dataset(args)
    model, train_iterator, optimizer = build_model_and_iterator(args, train_triples)
    epoch_steps = train_run.steps_per_epoch(len(train_triples), args.batch_size)
    loss_name = get_loss_display_name(args)

    history = {
        'epochs': [],
        'align_loss': [],
        'uniform_loss': [],
        'uniform': {key: [] for key in uniform_sets},
        'loss': [],
        'valid_metric': [],
    }
    train_time = 0.0
    valid_time = 0.0
    train_epoch_times = []
    train_peak_gpu_memory_gb = None
    train_peak_gpu_memory_reserved_gb = None
    best_epoch = None
    best_valid_value = float('-inf')
    best_model_state = None

    epoch_bar = tqdm(
        range(1, num_epochs + 1),
        desc='Training',
        unit='epoch',
        dynamic_ncols=True,
    )
    for epoch in epoch_bar:
        if is_learnable_kgau_gammas(args):
            args.current_epoch = epoch
            update_kgau_gamma_schedule(args)

        batch_logs = []
        if args.cuda:
            torch.cuda.reset_peak_memory_stats()
        _sync_cuda(args)
        train_start = time.perf_counter()
        step_bar = tqdm(
            range(epoch_steps),
            desc='  batches',
            unit='batch',
            leave=False,
            dynamic_ncols=True,
        )
        for _ in step_bar:
            batch_logs.append(
                train_step_with_metrics(model, optimizer, train_iterator, args, uniform_sets)
            )
        _sync_cuda(args)
        epoch_train_time = time.perf_counter() - train_start
        train_time += epoch_train_time
        train_epoch_times.append(epoch_train_time)

        if args.cuda:
            epoch_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            epoch_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            if train_peak_gpu_memory_gb is None:
                train_peak_gpu_memory_gb = epoch_peak
                train_peak_gpu_memory_reserved_gb = epoch_reserved
            else:
                train_peak_gpu_memory_gb = max(train_peak_gpu_memory_gb, epoch_peak)
                train_peak_gpu_memory_reserved_gb = max(
                    train_peak_gpu_memory_reserved_gb, epoch_reserved
                )

        history['epochs'].append(epoch)
        history['align_loss'].append(np.mean([log['align_loss'] for log in batch_logs]))
        history['uniform_loss'].append(np.mean([log['uniform_loss'] for log in batch_logs]))
        for key in uniform_sets:
            history['uniform'][key].append(
                np.mean([log['uniform'][key] for log in batch_logs])
            )
        history['loss'].append(np.mean([log['loss'] for log in batch_logs]))

        valid_start = time.perf_counter()
        metrics = KGEModel.test_step(model, valid_triples, all_true_triples, args)
        valid_time += time.perf_counter() - valid_start
        history['valid_metric'].append(metrics)

        metric_value = metrics[valid_metric]
        if metric_value > best_valid_value:
            best_valid_value = metric_value
            best_epoch = epoch
            best_model_state = clone_model_state(model)

        postfix = format_training_postfix(
            valid_metric,
            metrics[valid_metric],
            loss_name,
            history['loss'][-1],
            history['align_loss'][-1],
            history['uniform_loss'][-1],
        )
        epoch_bar.set_postfix_str(postfix, refresh=True)

    # Test / reports use best-valid weights, not the last training epoch.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            'Restored best model at epoch {} ({}={:.4f}) for evaluation'.format(
                best_epoch, valid_metric, best_valid_value,
            )
        )

    timing = {
        'train_time': train_time,
        'valid_time': valid_time,
        'test_time': 0.0,
        'total_time': train_time + valid_time,
        'peak_gpu_memory_gb': train_peak_gpu_memory_gb,
        'train_peak_gpu_memory_gb': train_peak_gpu_memory_gb,
        'train_peak_gpu_memory_reserved_gb': train_peak_gpu_memory_reserved_gb,
        'train_epoch_times': train_epoch_times,
        'epoch_steps': epoch_steps,
        'best_epoch': best_epoch,
        'best_valid_value': best_valid_value,
    }

    datasets = {
        'valid_triples': valid_triples,
        'test_triples': test_triples,
        'all_true_triples': all_true_triples,
    }
    return history, model, args, datasets, timing


def _truncate_history(history, num_epochs):
    n = min(num_epochs, len(history['epochs']))
    truncated = {}
    for key, value in history.items():
        if key == 'uniform':
            truncated[key] = {k: v[:n] for k, v in value.items()}
        else:
            truncated[key] = value[:n]
    return truncated


def _place_legend_bottom_right(ax_right, lines):
    ax_right.legend(
        lines,
        [line.get_label() for line in lines],
        loc='lower right',
        bbox_to_anchor=(1.0, 0.0),
        bbox_transform=ax_right.transAxes,
        frameon=True,
        framealpha=0.95,
    )


def plot_alignment_uniformity(history, num_epochs, uniform_sets, output_path=None):
    history = _truncate_history(history, num_epochs)
    epochs = history['epochs']

    fig, ax_left = plt.subplots(figsize=(6, 4))
    ax_right = ax_left.twinx()

    ax_left.plot(
        epochs, history['align_loss'],
        color='#e69138', linewidth=2, label=r'$l_{align}$'
    )

    uniform_lines = []
    for key in uniform_sets:
        color = UNIFORM_COLORS.get(key, '#1f77b4')
        line, = ax_right.plot(
            epochs, history['uniform'][key],
            color=color, linewidth=2,
            label=r'$l_{uniform}^{' + key + '}$',
        )
        uniform_lines.append(line)

    ax_left.set_xlabel('training epochs')
    ax_left.set_ylabel('alignment', color='#e69138')
    ax_right.set_ylabel('uniformity', color='#444444')
    ax_left.tick_params(axis='y', labelcolor='#e69138')
    ax_right.tick_params(axis='y', labelcolor='#444444')
    ax_left.set_xlim(min(epochs), max(epochs))

    fig.tight_layout()
    _place_legend_bottom_right(ax_right, uniform_lines)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_loss_and_metric(history, valid_metric, num_epochs, loss_label='loss', output_path=None):
    history = _truncate_history(history, num_epochs)
    epochs = history['epochs']
    metric_values = [m[valid_metric] for m in history['valid_metric']]

    fig, ax_left = plt.subplots(figsize=(6, 4))
    ax_right = ax_left.twinx()

    line_loss, = ax_left.plot(
        epochs, history['loss'],
        color='#4a86e8', linewidth=2, label=loss_label
    )
    line_metric, = ax_right.plot(
        epochs, metric_values,
        color='#cc0000', linewidth=2, label='performance'
    )

    ax_left.set_xlabel('training epochs')
    ax_left.set_ylabel(loss_label, color='#4a86e8')
    ax_right.set_ylabel(valid_metric, color='#cc0000')
    ax_left.tick_params(axis='y', labelcolor='#4a86e8')
    ax_right.tick_params(axis='y', labelcolor='#cc0000')
    ax_left.set_xlim(min(epochs), max(epochs))

    lines = [line_loss, line_metric]
    fig.tight_layout()
    _place_legend_bottom_right(ax_right, lines)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def configure_cuda_device(gpu=0):
    '''
    Bind the process to a CUDA device index among currently visible GPUs.

    Do not rewrite CUDA_VISIBLE_DEVICES after torch has been imported — that
    empties the visible device list and breaks Triton ("Invalid device id").
    '''
    if not torch.cuda.is_available():
        print('CUDA not available; running on CPU')
        return None
    n = torch.cuda.device_count()
    if gpu is None:
        gpu = 0
    gpu = int(gpu)
    if gpu < 0 or gpu >= n:
        print(
            'Warning: GPU {} unavailable (device_count={}); using GPU 0'.format(gpu, n)
        )
        gpu = 0
    torch.cuda.set_device(gpu)
    print('Using CUDA device {} ({})'.format(gpu, torch.cuda.get_device_name(gpu)))
    return gpu


def visualize_training(
    config_path,
    valid_metric='MRR',
    gpu=0,
    output_dir=None,
    show=True,
    uniform_sets=None):
    config, config_path = load_config(resolve_path(config_path))
    num_epochs = resolve_num_epochs(config)
    args = build_args(config)
    uniform_sets = validate_uniform_sets(uniform_sets or DEFAULT_UNIFORM_KEYS)
    configure_cuda_device(gpu)

    model_name = config.get('model')
    loss_name = getattr(args, 'loss', 'NoneLoss')
    strategy_name = getattr(args, 'strategy', 'NoneStrategy')
    dim = getattr(args, 'dim', 'NoneDim')
    batch_size = getattr(args, 'batch_size', 'NoneBatchSize')
    dataset_name = dataset_display_name(config.get('data_path') or getattr(args, 'data_path', ''))

    print('Config: {}'.format(config_path))
    print('Model: {}  Loss: {}  Strategy: {}  Dataset: {}'.format(
        model_name, loss_name, getattr(args, 'strategy', 'uniform'), dataset_name,
    ))
    if model_name == 'RotatE':
        print('RotatE score_mode: {}'.format(resolve_rotate_score_mode(args)))
    print('Training for {} epochs (from config)'.format(num_epochs))
    print('Uniform sets: {}'.format(', '.join(uniform_sets)))

    if output_dir is None:
        output_dir = build_output_dir(config_path)
    else:
        output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print('Output dir: {}'.format(output_dir))

    history, model, args, datasets, timing = train_and_collect_history(
        args, num_epochs, valid_metric=valid_metric, uniform_sets=uniform_sets,
    )

    loss_label = {
        'se': 'SE loss',
        'hinge': 'Hinge loss',
        'bce': 'BCE loss',
        'mr': 'MR loss',
        'bpr': 'BPR loss',
        'ce': 'CE loss',
        'sans': 'SANS loss',
        'kgau': 'KGAU loss',
        'kgmau': 'KGmAU loss',
        'kgmamu': 'KGmAmU loss',
    }.get(loss_name, get_loss_display_name(args) + ' loss')

    fig_au = plot_alignment_uniformity(
        history,
        num_epochs,
        uniform_sets,
        output_path=os.path.join(output_dir, 'alignment_uniformity.png'),
    )
    fig_curve = plot_loss_and_metric(
        history,
        valid_metric,
        num_epochs,
        loss_label=loss_label,
        output_path=os.path.join(output_dir, 'loss_and_{}.png'.format(valid_metric)),
    )

    if show:
        plt.show()
    else:
        plt.close(fig_au)
        plt.close(fig_curve)

    test_start = time.perf_counter()
    link_metrics, classification_metrics_dict = run_post_training_evaluation(
        model,
        args,
        datasets['test_triples'],
        datasets['all_true_triples'],
    )
    timing['test_time'] = time.perf_counter() - test_start
    timing['total_time'] = timing['train_time'] + timing['valid_time'] + timing['test_time']

    best_epoch, best_valid_value = find_best_valid(history, valid_metric=valid_metric)
    report_text = build_results_report(
        num_epochs=num_epochs,
        valid_metric=valid_metric,
        link_metrics=link_metrics,
        classification_metrics_dict=classification_metrics_dict,
        best_epoch=best_epoch,
        best_valid_value=best_valid_value,
        timing=timing,
        model_name=model_name,
        loss_name=loss_name,
        strategy_name=strategy_name,
        dim=dim,
        batch_size=batch_size,
        config_path=config_path,
        dataset_name=dataset_name,
        negative_samples=negative_samples_for_report(args),
        args=args,
    )
    write_results_report(output_dir, report_text)

    return history, fig_au, fig_curve


def parse_cli():
    parser = argparse.ArgumentParser(
        description='Train and visualize a KGE model from a JSON config file.'
    )
    parser.add_argument(
        'config',
        nargs='?',
        default='configs/ComplEx_WN18RR.json',
        help='Path to config JSON (default: configs/ComplEx_WN18RR.json)',
    )
    parser.add_argument('--valid-metric', default='MRR', help='Validation metric for learning curve')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index among visible GPUs (default: 0)')
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory to save PNG figures (default: visualization/outputs/<config_path>_<timestamp>)',
    )
    parser.add_argument('--no-show', action='store_true', help='Save figures without opening a window')
    parser.add_argument(
        '--uniform-sets', nargs='+', default=None,
        choices=list(DEFAULT_UNIFORM_KEYS),
        help='Uniformity embedding pools to track and plot (default: query target)',
    )
    return parser.parse_args()


def main():
    cli = parse_cli()

    visualize_training(
        cli.config,
        valid_metric=cli.valid_metric,
        gpu=cli.gpu,
        output_dir=resolve_path(cli.output_dir) if cli.output_dir else None,
        show=not cli.no_show,
        uniform_sets=cli.uniform_sets,
    )


if __name__ == '__main__':
    main()
