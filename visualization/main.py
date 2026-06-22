from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'codes'))

import run as train_run
from dataloader import BidirectionalOneShotIterator, TrainDataset
from loss import AlignmentUniformityLoss, compute_kge_loss, AUGammaController, build_training_optimizer, is_learnable_au_gammas, update_au_gamma_schedule
from model import KGEModel

UNIFORM_SET_KEYS = ('query', 'target', 'head', 'tail', 'entity', 'relation')
DEFAULT_UNIFORM_SETS = ['query', 'target', 'head', 'tail', 'entity', 'relation']

UNIFORM_COLORS = {
    'query': '#6aa84f',
    'target': '#38761d',
    'head': '#93c47d',
    'tail': '#b6d7a8',
    'entity': '#274e13',
    'relation': '#d9ead3',
}

LOSS_DISPLAY_NAMES = {
    'ce': 'CE',
    'mr': 'MR',
    'bce': 'BCE',
    'mse': 'MSE',
    'bpr': 'BPR',
    'infonce': 'InfoNCE',
    'self_adv': 'SA',
    'au': 'AU',
}


def get_loss_display_name(args):
    loss_name = getattr(args, 'loss', 'self_adv')
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
    try:
        config_rel = os.path.relpath(config_path, ROOT)
    except ValueError:
        config_rel = config_path
    config_slug = config_rel.replace(os.sep, '_')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_name = '{}_{}'.format(config_slug, timestamp)
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
    args.cpu_num = config.get('cpu_num', 10)

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
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
    )
    if args.cuda:
        model = model.cuda()

    if is_learnable_au_gammas(args):
        AUGammaController(args).ensure_model_params(model)

    train_dataloader_head = DataLoader(
        TrainDataset(
            train_triples, args.nentity, args.nrelation,
            args.negative_sample_size, 'head-batch'
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=TrainDataset.collate_fn,
    )
    train_dataloader_tail = DataLoader(
        TrainDataset(
            train_triples, args.nentity, args.nrelation,
            args.negative_sample_size, 'tail-batch'
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=TrainDataset.collate_fn,
    )
    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
    optimizer = build_training_optimizer(model, args)
    return model, train_iterator, optimizer


def validate_uniform_sets(uniform_sets):
    if not uniform_sets:
        raise ValueError('uniform_sets must be non-empty')
    allowed = set(UNIFORM_SET_KEYS)
    unknown = [key for key in uniform_sets if key not in allowed]
    if unknown:
        raise ValueError(
            'Unknown uniform_sets: {}. Allowed: {}'.format(
                unknown, ', '.join(UNIFORM_SET_KEYS)
            )
        )
    return list(uniform_sets)


def compute_au_metrics(model, positive_sample, mode, args, uniform_sets):
    au = AlignmentUniformityLoss(args)
    head = model.entity_embedding[positive_sample[:, 0]]
    relation = model.relation_embedding[positive_sample[:, 1]]
    tail = model.entity_embedding[positive_sample[:, 2]]

    query_e = model.query_encoder(head, relation, tail, mode=mode)
    target_e = model.target_encoder(tail, head=head, relation=relation, mode=mode)

    align_loss = au.alignment(query_e, target_e).item()
    
    tuni = getattr(args, 'tuni', 2)
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
        uniform_components[key] = au.uniformity(embeddings[key], tuni=tuni).item()

    uniform_loss = float(np.mean(list(uniform_components.values())))
    return align_loss, uniform_components, uniform_loss


def train_step_with_metrics(model, optimizer, train_iterator, args, uniform_sets):
    model.train()
    optimizer.zero_grad()

    positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
    if args.cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

    negative_score = model((positive_sample, negative_sample), mode=mode)
    positive_score = model(positive_sample)
    loss, log = compute_kge_loss(
        positive_score, negative_score, subsampling_weight, model, args,
        positive_sample=positive_sample, mode=mode,
    )
    loss.backward()
    optimizer.step()

    if is_learnable_au_gammas(args):
        AUGammaController(args).clamp_log_gammas(model)

    align_loss, uniform_components, uniform_loss = compute_au_metrics(
        model, positive_sample, mode, args, uniform_sets,
    )
    log['align_loss'] = align_loss
    log['uniform_loss'] = uniform_loss
    log['uniform'] = uniform_components
    return log


def train_and_collect_history(args, num_epochs, valid_metric='MRR', uniform_sets=None):
    uniform_sets = validate_uniform_sets(uniform_sets or DEFAULT_UNIFORM_SETS)
    train_triples, valid_triples, _, all_true_triples = load_dataset(args)
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

    epoch_bar = tqdm(
        range(1, num_epochs + 1),
        desc='Training',
        unit='epoch',
        dynamic_ncols=True,
    )
    for epoch in epoch_bar:
        if is_learnable_au_gammas(args):
            args.current_epoch = epoch
            update_au_gamma_schedule(args)

        batch_logs = []
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

        history['epochs'].append(epoch)
        history['align_loss'].append(np.mean([log['align_loss'] for log in batch_logs]))
        history['uniform_loss'].append(np.mean([log['uniform_loss'] for log in batch_logs]))
        for key in uniform_sets:
            history['uniform'][key].append(
                np.mean([log['uniform'][key] for log in batch_logs])
            )
        history['loss'].append(np.mean([log['loss'] for log in batch_logs]))

        metrics = KGEModel.test_step(model, valid_triples, all_true_triples, args)
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

    return history


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
        color = UNIFORM_COLORS.get(key, '#6aa84f')
        line, = ax_right.plot(
            epochs, history['uniform'][key],
            color=color, linewidth=2,
            label=r'$l_{uniform}^{' + key + '}$',
        )
        uniform_lines.append(line)

    ax_left.set_xlabel('training epochs')
    ax_left.set_ylabel('alignment', color='#e69138')
    ax_right.set_ylabel('uniformity', color='#6aa84f')
    ax_left.tick_params(axis='y', labelcolor='#e69138')
    ax_right.tick_params(axis='y', labelcolor='#6aa84f')
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
    display_epochs=100,
    gpu=0,
    output_dir=None,
    show=True,
    uniform_sets=None,
):
    config, config_path = load_config(resolve_path(config_path))
    num_epochs = resolve_num_epochs(config, display_epochs)
    args = build_args(config)
    uniform_sets = validate_uniform_sets(uniform_sets or DEFAULT_UNIFORM_SETS)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    print('Config: {}'.format(config_path))
    print('Model: {}  Loss: {}  Dataset: {}'.format(
        config.get('model'), getattr(args, 'loss', 'self_adv'), config.get('data_path')
    ))
    print('Training for {} epochs (displaying first {})'.format(num_epochs, display_epochs))
    print('Uniform sets: {}'.format(', '.join(uniform_sets)))

    if output_dir is None:
        output_dir = build_output_dir(config_path)
    else:
        output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print('Output dir: {}'.format(output_dir))

    history = train_and_collect_history(
        args, num_epochs, valid_metric=valid_metric, uniform_sets=uniform_sets,
    )

    loss_label = {
        'ce': 'CE loss',
        'mr': 'MR loss',
        'bce': 'BCE loss',
        'mse': 'MSE loss',
        'bpr': 'BPR loss',
        'infonce': 'InfoNCE loss',
        'self_adv': 'SA loss',
        'au': 'AU loss',
    }.get(getattr(args, 'loss', 'self_adv'), get_loss_display_name(args) + ' loss')

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
    parser.add_argument('--display-epochs', type=int, default=100, help='First N epochs to train and plot')
    parser.add_argument('--gpu', type=int, default=1, help='GPU device id')
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory to save PNG figures (default: visualization/outputs/<config_path>_<timestamp>)',
    )
    parser.add_argument('--no-show', action='store_true', help='Save figures without opening a window')
    parser.add_argument(
        '--uniform-sets', nargs='+', default=None,
        choices=list(UNIFORM_SET_KEYS),
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
