import argparse
import copy
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import T5Config

from data.sr_dataset import SeqRecDataset
from data.utils import ndcg_at_k, recall_at_k
from model.id.rqvae import RQVAE
from model.rec.t5 import T5Rec
from model.rec.manual import t5_manual_forward


def setup_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Generative Recommendation Training")

    # General
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--gpu_id", type=str, default="1")
    parser.add_argument("--optimizer", type=str, default="adamw")

    # Rec model hyperparameters
    parser.add_argument("--lr_rec", type=float, default=5e-4)
    parser.add_argument("--wd_rec", type=float, default=1e-2)

    # ID model hyperparameters
    parser.add_argument("--lr_id_load", type=float, default=1e-3)
    parser.add_argument("--wd_id_load", type=float, default=1e-4)
    parser.add_argument("--lr_id", type=float, default=1e-4)
    parser.add_argument("--wd_id", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=2e-2)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument(
        "--id_mode", type=str, choices=["tiger", "letter"], default="tiger"
    )
    parser.add_argument(
        "--id_ckpt_position", type=str, choices=["", "-last"], default="-last"
    )
    parser.add_argument("--load_path", type=str, default="")
    parser.add_argument("--in_dim", type=int, default=4096)
    parser.add_argument("--codebook_size_list", type=list, default=[256] * 4)

    # Training control
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--k_values", type=list, default=[5, 10, 20])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--pcgrad", type=bool, default=True)
    parser.add_argument("--freq_out", type=int, default=1)

    # Dataset parameters
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--max_his_len", type=int, default=20)

    return parser.parse_args()


def create_dataloader(args):
    """Create train, validation, and test dataloaders."""
    datasets = {
        "train": SeqRecDataset(args.data_path, args.dataset, args.max_his_len, "train"),
        "valid": SeqRecDataset(args.data_path, args.dataset, args.max_his_len, "valid"),
        "test": SeqRecDataset(args.data_path, args.dataset, args.max_his_len, "test"),
    }

    dataloaders = {
        key: DataLoader(
            dataset=dataset,
            batch_size=args.batch_size if key != "test" else args.batch_size // 4,
            shuffle=(key == "train"),
            num_workers=4,
            pin_memory=True,
        )
        for key, dataset in datasets.items()
    }

    return dataloaders


def initialize_models(args, device):
    """Initialize both RQVAE and T5 models."""
    if args.id_mode == "tiger":
        model_id_path = f"./ckpt/rqvae-{args.dataset}-tiger-lr_{args.lr_id_load}-wd_{args.wd_id_load}{args.id_ckpt_position}.pth"
    elif args.id_mode == "letter":
        model_id_path = f"./ckpt/rqvae-{args.dataset}-letter-alpha_{args.alpha}-beta_{args.beta}-lr_{args.lr_id_load}-wd_{args.wd_id_load}{args.id_ckpt_position}.pth"
    else:
        raise NotImplementedError(f"id type {args.id_mode} is not implemented.")

    model_id = torch.load(
        model_id_path, weights_only=False, map_location=torch.device("cpu")
    )
    model_id = model_id.to(device)

    model_id.build_indices()
    if args.id_mode == "letter":
        model_id.update_cluster_labels()

    config = T5Config(
        # Token-related
        vocab_size=256 * (len(args.codebook_size_list)) + 2,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        bos_token_id=0,
        # Model dimensions
        d_model=128,
        d_ff=1024,
        d_kv=64,
        # Architecture
        num_layers=4,
        num_decoder_layers=4,
        num_heads=6,
        is_encoder_decoder=True,
        # Attention
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        # Initialization & normalization
        initializer_factor=1.0,
        layer_norm_epsilon=1e-6,
        # Dropout
        dropout_rate=0.1,
        # Others
        feed_forward_proj="relu",
        n_positions=512,
        use_cache=True,
    )

    model_rec = T5Rec(config=config).to(device)

    model_rec.config.bos_token_id = 0
    model_rec.config.pad_token_id = 0
    model_rec.config.eos_token_id = 1
    model_rec.config.decoder_start_token_id = 0

    return model_id, model_rec


def initialize_optimizer(model_id, model_rec, args):
    """Initialize optimizers for the models."""
    if args.optimizer == "adamw":
        optim_id = AdamW(model_id.parameters(), lr=args.lr_id, weight_decay=args.wd_id)
        optim_rec = AdamW(
            model_rec.parameters(), lr=args.lr_rec, weight_decay=args.wd_rec
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} is not implemented.")

    return optim_id, optim_rec


def shift_to_token(hard_indices, soft_probs):
    """Shift hard indices and soft probabilities into token space."""
    batch_size, max_his_len_plus, codebook_size, codebook_num = soft_probs.shape
    device = hard_indices.device
    soft_probs = soft_probs.permute(0, 1, 3, 2)

    shift = (
        torch.tensor([2 + i * codebook_size for i in range(codebook_num)])
        .reshape(1, 1, codebook_num)
        .to(device)
    )
    hard_indices = (hard_indices + shift).reshape(batch_size, -1)

    output_tensor = torch.zeros(
        batch_size, max_his_len_plus, codebook_num, codebook_num * codebook_size + 2
    ).to(device)
    for i in range(codebook_num):
        start_idx, end_idx = i * codebook_size + 2, (i + 1) * codebook_size + 2
        output_tensor[:, :, i, start_idx:end_idx] += soft_probs[:, :, i, :].reshape(
            batch_size, -1, codebook_size
        )

    soft_probs = output_tensor.reshape(batch_size, -1, codebook_num * codebook_size + 2)
    return hard_indices, soft_probs, shift


def prepare_inputs(his, tgt, model_id):
    """Prepare inputs for the rec model."""
    device = his.device

    hard_indices, soft_probs, loss_id = model_id(torch.cat([his, tgt.unsqueeze(1)], 1))
    batch_size, _, codebook_size, codebook_num = soft_probs.shape

    hard_indices, soft_probs, shift = shift_to_token(hard_indices, soft_probs)

    # split his and tgt
    input_ids = hard_indices[:, :-codebook_num]
    inputs_embeds = soft_probs[:, :-codebook_num, :]
    decoder_input_ids = hard_indices[:, -codebook_num:]
    decoder_inputs_embeds = soft_probs[:, -codebook_num:, :]

    # prepare padding and eos
    padding = torch.zeros(codebook_num * codebook_size + 2).to(device)
    padding[0] = 1.0
    eos = torch.zeros_like(padding)
    eos[1] = 1.0

    # add padding to his
    his_ = his.unsqueeze(-1).repeat(1, 1, codebook_num).flatten(start_dim=1, end_dim=2)
    pad_indices = torch.nonzero(his_ == 0, as_tuple=True)
    input_ids[pad_indices[0], pad_indices[1]] = 0
    inputs_embeds[pad_indices[0], pad_indices[1], :] = padding

    # add eos to his
    input_ids = torch.cat(
        [input_ids, torch.zeros((batch_size, 1), dtype=torch.int64).to(device)],
        dim=1,
    )
    inputs_embeds = torch.cat([inputs_embeds, padding.expand(batch_size, 1, -1)], dim=1)
    pos = torch.sum((input_ids > 0), dim=1)
    input_ids[torch.arange(batch_size), pos] = 1
    inputs_embeds[torch.arange(batch_size), pos, :] = eos

    # add eos to tgt
    labels = torch.cat(
        [
            decoder_input_ids,
            torch.ones((batch_size, 1), dtype=torch.int64).to(device),
        ],
        dim=1,
    )

    # add bos to tgt
    decoder_input_ids = torch.cat(
        [
            torch.zeros((batch_size, 1), dtype=torch.int64).to(device),
            decoder_input_ids,
        ],
        dim=1,
    )
    decoder_inputs_embeds = torch.cat(
        [padding.expand(batch_size, 1, -1), decoder_inputs_embeds], dim=1
    )

    # encoder padding mask: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
    key_padding_mask = (input_ids != 0).int()

    return (
        input_ids,
        inputs_embeds,
        decoder_input_ids,
        decoder_inputs_embeds,
        labels,
        key_padding_mask,
        shift,
        loss_id,
    )


def train_epoch(
    epoch,
    model_id,
    model_rec,
    dataloader,
    optim_id,
    optim_rec,
    freq_out,
    lr_rec,
    gamma,
    pcgrad,
    device,
    id_mode,
):
    """Performs one training epoch."""
    model_id.train()
    model_rec.train()

    total_loss_rec_in = []
    total_loss_rec_out = []
    total_loss_id_out = []
    total_loss_out = []

    for step, batch_data in enumerate(dataloader):

        # Important: Prevents CUDA Out Of Memory (OOM) errors during training
        torch.cuda.empty_cache()

        # his: [batch_size, max_his_len], tgt: [batch_size]
        his, tgt = batch_data["his"].to(device), batch_data["tgt"].to(device)

        # model_rec down
        optim_rec.zero_grad()

        (
            input_ids,
            inputs_embeds,
            decoder_input_ids,
            decoder_inputs_embeds,
            labels,
            key_padding_mask,
            _,
            loss_id,
        ) = prepare_inputs(his, tgt, model_id)

        loss_rec = model_rec._forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=key_padding_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
        )

        loss_rec.backward()
        optim_rec.step()
        total_loss_rec_in.append(loss_rec.item())

        # model_id update
        if (step + 1) % (len(dataloader) // freq_out) == 0:

            # Stage 1: Update model_rec_copy
            model_rec_copy = copy.deepcopy(model_rec).to(device)
            model_rec_copy.zero_grad()
            (
                input_ids,
                inputs_embeds,
                decoder_input_ids,
                decoder_inputs_embeds,
                labels,
                key_padding_mask,
                _,
                loss_id,
            ) = prepare_inputs(his, tgt, model_id)

            loss_rec = model_rec_copy._forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=key_padding_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
            )
            grads = torch.autograd.grad(
                loss_rec,
                model_rec_copy.parameters(),
                retain_graph=True,
                create_graph=True,
            )
            params_update = {
                name: param - lr_rec * grad
                for (name, param), grad in zip(model_rec_copy.named_parameters(), grads)
            }

            # Stage 2: Update model_id
            optim_id.zero_grad()
            loss_rec = t5_manual_forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=key_padding_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                model_rec=model_rec_copy,
                params=params_update,
            )

            if pcgrad:
                params_with_grad = [
                    (name, p)
                    for name, p in model_id.named_parameters()
                    if p.requires_grad
                ]
                grads_id = torch.autograd.grad(
                    loss_id,
                    [p for _, p in params_with_grad],
                    retain_graph=True,
                )
                grads_rec = torch.autograd.grad(
                    loss_rec,
                    [p for _, p in params_with_grad],
                    retain_graph=True,
                    allow_unused=True,
                )
                for (name, param), grad_id, grad_rec in zip(
                    params_with_grad, grads_id, grads_rec
                ):
                    if grad_rec is not None:
                        grad_rec_proj = grad_rec
                        dot_product = torch.sum(grad_id * grad_rec)
                        if dot_product < 0:
                            # x = x - <x,y>/<y,y> y
                            negtive_term = torch.sum(
                                grad_rec * F.normalize(grad_id, p=2, dim=-1)
                            ) * F.normalize(grad_id, p=2, dim=-1)
                            grad_rec_proj -= negtive_term
                        param.grad = gamma * grad_id + grad_rec_proj
                    else:
                        param.grad = gamma * grad_id

                loss = gamma * loss_id + loss_rec
                optim_id.step()
            else:
                loss = gamma * loss_id + loss_rec
                loss.backward()
                optim_id.step()

            total_loss_rec_out.append(loss_rec.item())
            total_loss_id_out.append(loss_id.item())
            total_loss_out.append(loss.item())

            model_id.build_indices()
            model_id.update_cluster_labels()

    avg_loss_rec_in = np.mean(total_loss_rec_in)
    avg_loss_rec_out = np.mean(total_loss_rec_out)
    avg_loss_id_out = np.mean(total_loss_id_out)
    avg_loss_out = np.mean(total_loss_out)

    print(
        f"Epoch {epoch} - "
        f"Avg Rec In Loss: {avg_loss_rec_in:.4f}, "
        f"Avg Rec Out Loss: {avg_loss_rec_out:.4f}, "
        f"Avg Id Out Loss: {avg_loss_id_out:.4f}, "
        f"Avg Total Out Loss: {avg_loss_out:.4f}, "
    )


@torch.no_grad()
def eval_epoch(
    model_id,
    model_rec,
    dataloader,
    device,
):
    """Performs one eval epoch."""
    model_id.eval()
    model_rec.eval()

    loss = 0.0
    num = 0

    for step, batch_data in enumerate(dataloader):
        # his: [batch_size, max_his_len], tgt: [batch_size]
        his, tgt = batch_data["his"].to(device), batch_data["tgt"].to(device)
        (
            input_ids,
            inputs_embeds,
            decoder_input_ids,
            decoder_inputs_embeds,
            labels,
            key_padding_mask,
            _,
            _,
        ) = prepare_inputs(his, tgt, model_id)

        loss_rec = model_rec._forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=key_padding_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
        )
        loss += loss_rec.item() * tgt.shape[0]
        num += tgt.shape[0]

    metric = {"loss": loss / num}
    return metric


@torch.no_grad()
def test(model_id, model_rec, dataloader, device, beam_size=10, k_values=[5, 10]):
    """Perform evaluation on the test set."""
    model_id.build_indices()

    model_id.eval()
    model_rec.eval()

    tgts = []
    preds = []
    for step, batch_data in enumerate(dataloader):
        his, tgt = batch_data["his"].to(device), batch_data["tgt"].to(device)
        (input_ids, _, decoder_input_ids, _, _, key_padding_mask, shift, _) = (
            prepare_inputs(his, tgt, model_id)
        )

        pred_ids = model_rec._inference(
            input_ids=input_ids,
            attention_mask=key_padding_mask,
            all_indices=model_id.all_indices + shift[0].cpu().numpy(),
            beam_size=beam_size,
        )
        tgts.append(decoder_input_ids[:, 1:].cpu().numpy())
        preds.append(pred_ids.cpu().numpy())
    tgts = np.concatenate(tgts, axis=0)  # [N, codebook_num]
    preds = np.concatenate(preds, axis=0)  # [N, beam_size, codebook_num]
    tgts = np.array([",".join(map(str, tgt)) for tgt in tgts])  # [N]
    preds = np.array(
        [[",".join(map(str, pred)) for pred in sample] for sample in preds]
    )  # [N, beam_size]

    metric = {}
    for k in k_values:
        metric["recall@{}".format(k)] = recall_at_k(preds, tgts, k)
        metric["ndcg@{}".format(k)] = ndcg_at_k(preds, tgts, k)

    return metric


def run():
    args = parse_args()
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda")

    # Initialize dataloaders
    dataloaders = create_dataloader(args)
    train_dataloader, valid_dataloader, test_dataloader = (
        dataloaders["train"],
        dataloaders["valid"],
        dataloaders["test"],
    )

    # Initialize models
    model_id, model_rec = initialize_models(args, device)

    # Initialize optimizers
    optim_id, optim_rec = initialize_optimizer(model_id, model_rec, args)

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(model_id)
    print(model_rec)

    # Training and Evaluation
    best_epoch = -1
    best_valid_metric = {"loss": 1e8}
    patience_counter = 0
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_path = f"./ckpt/{args.id_mode}_t5_blo-{args.dataset}-lr_id_{args.lr_id}-wd_id_{args.wd_id}-gamma_{args.gamma}-freq_out_{args.freq_out}.pth"
    refer_metric = "loss"

    if args.load_path == "":

        for epoch in range(args.max_epochs):
            # Train
            train_epoch(
                epoch,
                model_id,
                model_rec,
                train_dataloader,
                optim_id,
                optim_rec,
                args.freq_out,
                args.lr_rec,
                args.gamma,
                args.pcgrad,
                device,
                args.id_mode,
            )

            # Validation
            valid_metric = eval_epoch(model_id, model_rec, valid_dataloader, device)
            print(
                f"Epoch {epoch}: Validation Metrics: {', '.join([f'{k} = {v:.4f}' for k, v in valid_metric.items()])}"
            )

            if valid_metric[refer_metric] < best_valid_metric[refer_metric]:
                best_epoch = epoch
                best_valid_metric = valid_metric
                patience_counter = 0
                torch.save(
                    {
                        "model_id_weight": model_id,
                        "model_rec_weight": model_rec,
                    },
                    ckpt_path,
                )
                print("Save model weights successfully.")
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}.")
            if patience_counter >= args.patience:
                print(
                    f"Early stopping triggered. Validation metric did not improve for {args.patience} epochs."
                )
                break

        weights = torch.load(ckpt_path, weights_only=False, map_location=device)
        model_id = weights["model_id_weight"]
        model_rec = weights["model_rec_weight"]

        # Validation for best epoch
        valid_metric = eval_epoch(model_id, model_rec, valid_dataloader, device)
        print(
            f"Best Epoch {best_epoch}: Validation Metrics: {', '.join([f'{k} = {v:.4f}' for k, v in valid_metric.items()])}"
        )

    else:
        weights = torch.load(args.load_path, weights_only=False, map_location=device)
        model_id = weights["model_id_weight"]
        model_rec = weights["model_rec_weight"]

    # Test for best epoch
    test_metric = test(
        model_id,
        model_rec,
        test_dataloader,
        device,
        args.beam_size,
        args.k_values,
    )
    print(
        f"Best Epoch {best_epoch}: Test Metrics: {', '.join([f'{k} = {v:.4f}' for k, v in test_metric.items()])}"
    )


if __name__ == "__main__":
    run()
