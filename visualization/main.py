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
from model import KGEModel
from strategy import get_strategy, resolve_strategy_name

DEFAULT_DISPLAY_EPOCHS = 200
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
    batch_size):
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
    
    time_per_epoch = format_duration(timing['train_time'] / max(num_epochs, 1))
    # Prefer train-only peak when available (excludes valid/test).
    peak_value = timing.get('train_peak_gpu_memory_gb')
    if peak_value is None:
        peak_value = timing.get('peak_gpu_memory_gb')
    if peak_value is None:
        peak_gpu_memory = None
    else:
        peak_gpu_memory = '{:.2f} GB'.format(float(peak_value))

    lines = [
        'Model: {}'.format(model_name),
        'Loss: {}'.format(loss_name),
        'Strategy: {}'.format(strategy_name),
        'Dim: {}'.format(dim),
        'Batch Size: {}'.format(batch_size),
        ''
    ]
    
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
        '',
        'Training: {}'.format(train_time),
        'Valid: {}'.format(valid_time),
        'Test: {}'.format(test_time),
        'Total: {}'.format(total_time),
        ''
    ]

    if time_per_epoch is not None:
        lines += [
            'Time per epoch: {}'.format(time_per_epoch)
        ]
    if peak_gpu_memory is not None:
        lines += [
            'Peak GPU memory: {}'.format(peak_gpu_memory)
        ]
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


def build_eff_report(args, timing, num_epochs, epoch_steps):
    '''Train-only efficiency metrics for fair strategy comparison.'''
    train_epoch_times = timing.get('train_epoch_times') or []
    if len(train_epoch_times) > 1:
        # Skip first epoch as warmup (allocator / cudnn).
        steady = train_epoch_times[1:]
    else:
        steady = train_epoch_times
    time_per_epoch = float(np.mean(steady)) if steady else 0.0

    peak = timing.get('train_peak_gpu_memory_gb')
    peak_str = '{:.4f}'.format(peak) if peak is not None else 'N/A'
    reserved = timing.get('train_peak_gpu_memory_reserved_gb')
    reserved_str = '{:.4f}'.format(reserved) if reserved is not None else 'N/A'

    lines = [
        'Efficiency report (train-only)',
        '==============================',
        'Model: {}'.format(getattr(args, 'model', None)),
        'Loss: {}'.format(getattr(args, 'loss', None)),
        'Strategy: {}'.format(getattr(args, 'strategy', None)),
        'Dim: {}'.format(getattr(args, 'dim', None)),
        'Batch size: {}'.format(getattr(args, 'batch_size', None)),
        'Negative sample size: {}'.format(
            getattr(args, 'negative_sample_size', None)
        ),
        'Negative chunk size: {}'.format(
            getattr(args, 'negative_chunk_size', 0) or 'auto'
        ),
        'Uniform pair chunk size: {}'.format(
            getattr(args, 'uniform_pair_chunk_size', 0) or 'auto'
        ),
        'Nentity: {}'.format(getattr(args, 'nentity', None)),
        'Candidates per positive: {}'.format(candidates_per_positive(args)),
        'Steps per epoch: {}'.format(epoch_steps),
        'Num epochs measured: {}'.format(num_epochs),
        'Warmup epochs excluded from mean time: {}'.format(
            1 if len(train_epoch_times) > 1 else 0
        ),
        '',
        'Train time total (s): {:.4f}'.format(timing.get('train_time', 0.0)),
        'Train time per epoch (s): {:.4f}'.format(time_per_epoch),
        'Train time per epoch (formatted): {}'.format(format_duration(time_per_epoch)),
        'Valid time total (s): {:.4f}'.format(timing.get('valid_time', 0.0)),
        'Test time total (s): {:.4f}'.format(timing.get('test_time', 0.0)),
        '',
        'Train peak GPU memory allocated (GB): {}'.format(peak_str),
        'Train peak GPU memory reserved (GB): {}'.format(reserved_str),
        '',
        'Notes:',
        '- Peak memory is max_memory_allocated during train steps only',
        '  (reset each epoch before train; excludes valid/test).',
        '- Time per epoch averages train-only wall time with CUDA sync;',
        '  first epoch excluded when num_epochs > 1.',
    ]
    return '\n'.join(lines) + '\n'


def write_eff_report(output_dir, report_text):
    path = os.path.join(output_dir, 'eff-report.txt')
    with open(path, 'w') as fout:
        fout.write(report_text)
    print('Efficiency report saved to {}'.format(path))
    return path


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
    get_strategy(args)

    return args


def resolve_num_epochs(config, display_epochs):
    return min(int(config['epochs']), int(display_epochs))


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
    model = KGEModel(
        model_name=args.model,
        nentity=args.nentity,
        nrelation=args.nrelation,
        dim=args.dim,
        margin_gamma=args.margin_gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
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
    uniform_margin = getattr(args, 'uniform_margin', 2.0)
    uniform_t = DEFAULT_UNIFORM_T

    if loss_name in ('kgmau', 'kgmamu'):
        align_loss = loss_fn.margin_alignment(
            query_e, target_e, align_margin=align_margin,
        ).item()
    else:
        align_loss = loss_fn.alignment(query_e, target_e).item()

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

        postfix = format_training_postfix(
            valid_metric,
            metrics[valid_metric],
            loss_name,
            history['loss'][-1],
            history['align_loss'][-1],
            history['uniform_loss'][-1],
        )
        epoch_bar.set_postfix_str(postfix, refresh=True)

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
    }

    datasets = {
        'valid_triples': valid_triples,
        'test_triples': test_triples,
        'all_true_triples': all_true_triples,
    }
    return history, model, args, datasets, timing


def _truncate_history(history, display_epochs):
    n = min(display_epochs, len(history['epochs']))
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


def plot_alignment_uniformity(history, display_epochs, uniform_sets, output_path=None):
    history = _truncate_history(history, display_epochs)
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


def plot_loss_and_metric(history, valid_metric, display_epochs, loss_label='loss', output_path=None):
    history = _truncate_history(history, display_epochs)
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


def visualize_training(
    config_path,
    valid_metric='MRR',
    display_epochs=DEFAULT_DISPLAY_EPOCHS,
    gpu=1,
    output_dir=None,
    show=True,
    uniform_sets=None):
    config, config_path = load_config(resolve_path(config_path))
    num_epochs = resolve_num_epochs(config, display_epochs)
    args = build_args(config)
    uniform_sets = validate_uniform_sets(uniform_sets or DEFAULT_UNIFORM_KEYS)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    model_name = config.get('model')
    loss_name = getattr(args, 'loss', 'NoneLoss')
    strategy_name = getattr(args, 'strategy', 'NoneStrategy')
    dim = getattr(args, 'dim', 'NoneDim')
    batch_size = getattr(args, 'batch_size', 'NoneBatchSize')

    print('Config: {}'.format(config_path))
    print('Model: {}  Loss: {}  Strategy: {}  Dataset: {}'.format(
        model_name, loss_name, getattr(args, 'strategy', 'uniform'), config.get('data_path')
    ))
    print('Training for {} epochs (displaying first {})'.format(num_epochs, display_epochs))
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
        display_epochs,
        uniform_sets,
        output_path=os.path.join(output_dir, 'alignment_uniformity.png'),
    )
    fig_curve = plot_loss_and_metric(
        history,
        valid_metric,
        display_epochs,
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
    )
    write_results_report(output_dir, report_text)

    eff_text = build_eff_report(
        args,
        timing,
        num_epochs=num_epochs,
        epoch_steps=timing.get('epoch_steps', 0),
    )
    write_eff_report(output_dir, eff_text)

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
    parser.add_argument(
        '--display-epochs', type=int, default=DEFAULT_DISPLAY_EPOCHS,
        help='First N epochs to train and plot',
    )
    parser.add_argument('--gpu', type=int, default=1, help='GPU device id')
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
        display_epochs=cli.display_epochs,
        gpu=cli.gpu,
        output_dir=resolve_path(cli.output_dir) if cli.output_dir else None,
        show=not cli.no_show,
        uniform_sets=cli.uniform_sets,
    )


if __name__ == '__main__':
    main()
