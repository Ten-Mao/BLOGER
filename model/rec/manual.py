import torch.nn.functional as F
import torch
import math
from torch.nn import CrossEntropyLoss


def t5_manual_forward(
    input_ids=None,
    inputs_embeds=None,
    attention_mask=None,
    decoder_input_ids=None,
    decoder_inputs_embeds=None,
    labels=None,
    model_rec=None,
    params=None,
):
    params = model_rec.state_dict() if params == None else params
    config = model_rec.config
    num_layers = config.num_layers
    d_model = config.d_model
    d_kv = config.d_kv
    relative_attention_num_buckets = config.relative_attention_num_buckets
    relative_attention_max_distance = config.relative_attention_max_distance
    eps = config.layer_norm_epsilon
    n_heads = config.num_heads
    dropout = config.dropout_rate

    embed_tokens = params["shared.weight"]

    def get_inputs_embeds(input_ids, inputs_embeds):
        assert input_ids is not None and inputs_embeds is not None
        hard_embeds = F.embedding(input_ids, embed_tokens)
        soft_embeds = torch.matmul(inputs_embeds, embed_tokens)
        inputs_embeds = (hard_embeds - soft_embeds).detach() + soft_embeds
        return inputs_embeds

    inputs_embeds = get_inputs_embeds(input_ids, inputs_embeds)
    decoder_inputs_embeds = get_inputs_embeds(decoder_input_ids, decoder_inputs_embeds)

    encoder_hidden = F.dropout(inputs_embeds, p=dropout, training=True)
    w_rab = None  # used for all attention layers
    for i in range(num_layers):
        prefix = f"encoder.block.{i}.layer.0"
        ln_weight = params[f"{prefix}.layer_norm.weight"]
        encoder_hidden_normed = t5_manual_layer_norm(ln_weight, eps, encoder_hidden)

        w_q, w_k, w_v, w_o = (
            params[f"{prefix}.SelfAttention.q.weight"],
            params[f"{prefix}.SelfAttention.k.weight"],
            params[f"{prefix}.SelfAttention.v.weight"],
            params[f"{prefix}.SelfAttention.o.weight"],
        )
        if w_rab is None:
            w_rab = params[f"{prefix}.SelfAttention.relative_attention_bias.weight"]
        encoder_attn_output = t5_manual_self_attention(
            w_q,
            w_k,
            w_v,
            w_o,
            w_rab,
            False,
            relative_attention_num_buckets,
            relative_attention_max_distance,
            n_heads,
            d_kv,
            dropout,
            encoder_hidden_normed,
            attention_mask,
        )
        encoder_hidden = encoder_hidden + F.dropout(
            encoder_attn_output, p=dropout, training=True
        )
        encoder_hidden = clamp_for_fp16(encoder_hidden)

        prefix = f"encoder.block.{i}.layer.1"
        ln_weight = params[f"{prefix}.layer_norm.weight"]
        encoder_hidden_normed = t5_manual_layer_norm(ln_weight, eps, encoder_hidden)

        w_wi = params[f"{prefix}.DenseReluDense.wi.weight"]
        w_wo = params[f"{prefix}.DenseReluDense.wo.weight"]
        encoder_ff_output = t5_manual_ff(w_wi, w_wo, dropout, encoder_hidden_normed)
        encoder_hidden = encoder_hidden + F.dropout(
            encoder_ff_output, p=dropout, training=True
        )

    ln_weight = params["encoder.final_layer_norm.weight"]
    encoder_hidden = t5_manual_layer_norm(ln_weight, eps, encoder_hidden)
    encoder_hidden = F.dropout(encoder_hidden, p=dropout, training=True)

    decoder_hidden = F.dropout(decoder_inputs_embeds, p=dropout, training=True)
    w_rab = None  # used for all attention layers
    for i in range(num_layers):
        prefix = f"decoder.block.{i}.layer.0"
        ln_weight = params[f"{prefix}.layer_norm.weight"]
        decoder_hidden_normed = t5_manual_layer_norm(ln_weight, eps, decoder_hidden)

        w_q, w_k, w_v, w_o = (
            params[f"{prefix}.SelfAttention.q.weight"],
            params[f"{prefix}.SelfAttention.k.weight"],
            params[f"{prefix}.SelfAttention.v.weight"],
            params[f"{prefix}.SelfAttention.o.weight"],
        )
        if w_rab is None:
            w_rab = params[f"{prefix}.SelfAttention.relative_attention_bias.weight"]

        decoder_attn_output = t5_manual_self_attention(
            w_q,
            w_k,
            w_v,
            w_o,
            w_rab,
            True,
            relative_attention_num_buckets,
            relative_attention_max_distance,
            n_heads,
            d_kv,
            dropout,
            decoder_hidden_normed,
            None,
        )
        decoder_hidden = decoder_hidden + F.dropout(
            decoder_attn_output, p=dropout, training=True
        )
        decoder_hidden = clamp_for_fp16(decoder_hidden)

        prefix = f"decoder.block.{i}.layer.1"
        ln_weight = params[f"{prefix}.layer_norm.weight"]
        decoder_hidden_normed = t5_manual_layer_norm(ln_weight, eps, decoder_hidden)

        w_q, w_k, w_v, w_o = (
            params[f"{prefix}.EncDecAttention.q.weight"],
            params[f"{prefix}.EncDecAttention.k.weight"],
            params[f"{prefix}.EncDecAttention.v.weight"],
            params[f"{prefix}.EncDecAttention.o.weight"],
        )
        decoder_attn_output = t5_manual_cross_attention(
            w_q,
            w_k,
            w_v,
            w_o,
            n_heads,
            d_kv,
            dropout,
            decoder_hidden_normed,
            encoder_hidden,
            attention_mask,
        )
        decoder_hidden = decoder_hidden + F.dropout(
            decoder_attn_output, p=dropout, training=True
        )
        decoder_hidden = clamp_for_fp16(decoder_hidden)

        prefix = f"decoder.block.{i}.layer.2"
        ln_weight = params[f"{prefix}.layer_norm.weight"]
        decoder_hidden_normed = t5_manual_layer_norm(ln_weight, eps, decoder_hidden)

        w_wi = params[f"{prefix}.DenseReluDense.wi.weight"]
        w_wo = params[f"{prefix}.DenseReluDense.wo.weight"]
        decoder_ff_output = t5_manual_ff(w_wi, w_wo, dropout, decoder_hidden_normed)
        decoder_hidden = decoder_hidden + F.dropout(
            decoder_ff_output, p=dropout, training=True
        )
        decoder_hidden = clamp_for_fp16(decoder_hidden)

    ln_weight = params["decoder.final_layer_norm.weight"]
    decoder_hidden = t5_manual_layer_norm(ln_weight, eps, decoder_hidden)
    decoder_hidden = F.dropout(decoder_hidden, p=dropout, training=True)

    sequence_output = decoder_hidden
    if config.tie_word_embeddings:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (d_model**-0.5)

    lm_logits = F.linear(sequence_output, embed_tokens)

    def ranking_loss(lm_logits, labels):
        assert labels is not None
        t_logits = lm_logits / config.temperature
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="mean")
        labels = labels.to(lm_logits.device)
        loss = loss_fct(t_logits.reshape(-1, t_logits.size(-1)), labels.reshape(-1))
        return loss

    loss = ranking_loss(lm_logits, labels)

    return loss


def t5_manual_layer_norm(w_ln, eps, hidden_states):
    hidden_states = hidden_states.to(torch.float32)
    variance = (hidden_states).pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)

    # convert into half-precision if necessary
    if w_ln.dtype in [torch.float16, torch.bfloat16]:
        hidden_states = hidden_states.to(w_ln.dtype)

    return w_ln * hidden_states


def t5_manual_self_attention(
    w_q,
    w_k,
    w_v,
    w_o,
    w_rab,
    is_decoder,
    relative_attention_num_buckets,
    relative_attention_max_distance,
    n_heads,
    d_kv,
    dropout,
    hidden_states,
    attention_mask,
):
    batch_size, seq_length = hidden_states.shape[:2]

    query_states = F.linear(hidden_states, w_q)
    query_states = query_states.view(batch_size, -1, n_heads, d_kv).transpose(1, 2)
    key_states = F.linear(hidden_states, w_k)
    key_states = key_states.view(batch_size, -1, n_heads, d_kv).transpose(1, 2)
    value_states = F.linear(hidden_states, w_v)
    value_states = value_states.view(batch_size, -1, n_heads, d_kv).transpose(1, 2)

    scores = torch.matmul(query_states, key_states.transpose(3, 2))

    assert w_rab is not None
    position_bias = compute_bias(
        w_rab,
        seq_length,
        seq_length,
        device=scores.device,
        cache_position=None,
        is_decoder=is_decoder,
        relative_attention_num_buckets=relative_attention_num_buckets,
        relative_attention_max_distance=relative_attention_max_distance,
    )

    if is_decoder:
        attention_mask = torch.tril(
            torch.ones([seq_length, seq_length], device=scores.device)
        )
        mask = attention_mask[None, None, :, :]
        mask = mask.to(dtype=hidden_states.dtype)
        mask = (1.0 - mask) * torch.finfo(hidden_states.dtype).min
    else:
        assert attention_mask is not None
        mask = attention_mask[:, None, None, :]
        mask = mask.to(dtype=hidden_states.dtype)
        mask = (1.0 - mask) * torch.finfo(hidden_states.dtype).min

    position_bias_masked = position_bias + mask
    scores += position_bias_masked

    attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn_weights = F.dropout(attn_weights, p=dropout, training=True)

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_length, -1)
    attn_output = F.linear(attn_output, w_o)

    return attn_output


def t5_manual_ff(w_wi, w_wo, dropout, hidden_states):
    forwarded_states = F.linear(hidden_states, w_wi)
    forwarded_states = F.relu(forwarded_states)
    forwarded_states = F.dropout(forwarded_states, p=dropout, training=True)
    if (
        isinstance(w_wo, torch.Tensor)
        and forwarded_states.dtype != w_wo.dtype
        and w_wo.dtype != torch.int8
    ):
        forwarded_states = forwarded_states.to(w_wo.dtype)
    forwarded_states = F.linear(forwarded_states, w_wo)

    return forwarded_states


def t5_manual_cross_attention(
    w_q,
    w_k,
    w_v,
    w_o,
    n_heads,
    d_kv,
    dropout,
    hidden_states,
    key_value_states,
    attention_mask,
):
    batch_size, seq_length = hidden_states.shape[:2]
    seq_length_encoder = key_value_states.shape[1]
    query_states = F.linear(hidden_states, w_q)
    query_states = query_states.view(batch_size, -1, n_heads, d_kv).transpose(1, 2)
    key_states = F.linear(key_value_states, w_k)
    key_states = key_states.view(batch_size, -1, n_heads, d_kv).transpose(1, 2)
    value_states = F.linear(key_value_states, w_v)
    value_states = value_states.view(batch_size, -1, n_heads, d_kv).transpose(1, 2)

    scores = torch.matmul(query_states, key_states.transpose(3, 2))

    position_bias = torch.zeros(
        (1, n_heads, seq_length, seq_length_encoder),
        device=scores.device,
        dtype=scores.dtype,
    )
    position_bias.requires_grad = True

    assert attention_mask is not None
    mask = attention_mask[:, None, None, :]
    mask = mask.to(dtype=hidden_states.dtype)
    mask = (1.0 - mask) * torch.finfo(hidden_states.dtype).min

    scores += mask

    attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn_weights = F.dropout(attn_weights, p=dropout, training=True)

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_length, -1)
    attn_output = F.linear(attn_output, w_o)

    return attn_output


def _relative_position_bucket(
    relative_position, bidirectional=True, num_buckets=32, max_distance=128
):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(
            relative_position, torch.zeros_like(relative_position)
        )
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets


def compute_bias(
    relative_attention_bias,
    query_length,
    key_length,
    device=None,
    cache_position=None,
    is_decoder=False,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
):
    """Compute binned relative position bias"""
    if device is None:
        device = relative_attention_bias.device
    if cache_position is None:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
    else:
        context_position = cache_position[:, None].to(device)
    memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    relative_position = (
        memory_position - context_position
    )  # shape (query_length, key_length)
    relative_position_bucket = _relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not is_decoder),
        num_buckets=relative_attention_num_buckets,
        max_distance=relative_attention_max_distance,
    )
    values = F.embedding(
        relative_position_bucket, relative_attention_bias
    )  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(
        0
    )  # shape (1, num_heads, query_length, key_length)
    return values


def clamp_for_fp16(hidden_states):
    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16:
        clamp_value = torch.where(
            torch.isinf(hidden_states).any(),
            torch.finfo(hidden_states.dtype).max - 1000,
            torch.finfo(hidden_states.dtype).max,
        )
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    return hidden_states
