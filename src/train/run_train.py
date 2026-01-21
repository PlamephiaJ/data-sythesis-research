import datetime
import os
import os.path
from itertools import chain

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ema import ExponentialMovingAverage
from losses import OptimizationManager, SEDDInfoNCELoss, StepFn
from optimizers import OptimizerRegistry
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import model.noise_lib as noise_lib
from data_process import data
from model import SEDD, graph_lib
from sample import sampling
from utils import utils
from utils.eval_factory import get_alignment_metric, get_eval_lm
from utils.tokenizer_factory import get_caption_tokenizer, get_text_tokenizer


torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


def _run(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
        # Initialize TensorBoard writer
        tb_dir = os.path.join(work_dir, "tensorboard")
        utils.makedirs(tb_dir)
        writer = SummaryWriter(log_dir=tb_dir)
        mprint(f"TensorBoard logging to: {tb_dir}")

    mprint(work_dir)
    mprint(cfg)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024**3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)

    # build score model
    score_model = SEDD(cfg).to(device)
    score_model = DDP(
        score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True
    )

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    # Log model architecture to TensorBoard
    if rank == 0:
        writer.add_text("model/num_parameters", str(num_parameters), 0)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)
    sampling_eps = 1e-5

    # build optimization state
    # optimizer = losses.get_optimizer(
    #     cfg, chain(score_model.parameters(), noise.parameters())
    # )
    optimizer = OptimizerRegistry.build(
        cfg, chain(score_model.parameters(), noise.parameters())
    )
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.amp.GradScaler()
    mprint(f"Scaler: {scaler}")
    state = dict(
        optimizer=optimizer,
        scaler=scaler,
        model=score_model,
        noise=noise,
        ema=ema,
        step=0,
    )

    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state["step"])

    # load in tokenizer
    tokenizer_text = get_text_tokenizer("gpt2")
    tokenizer_caption = get_caption_tokenizer("bert-base-uncased")
    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg)

    # mprint(f"Length of datasets: {len(train_ds)}, {len(eval_ds)}")

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = OptimizationManager(cfg)
    train_step_fn = StepFn(
        loss_fn=SEDDInfoNCELoss(
            cfg, noise, graph, True, p_uncond=cfg.training.p_uncond
        ),
        train=True,
        optimize_fn=optimize_fn,
        accum=cfg.training.accum,
    )

    eval_step_fn = StepFn(
        loss_fn=SEDDInfoNCELoss(
            cfg, noise, graph, False, p_uncond=cfg.training.p_uncond
        ),
        train=False,
        optimize_fn=optimize_fn,
        accum=cfg.training.accum,
    )

    if cfg.training.snapshot_sampling:
        sampling_shape = (
            cfg.training.batch_size // (cfg.ngpus * cfg.training.accum),
            cfg.model.length,
        )
        sampling_fn = sampling.PCSampler(
            graph,
            noise,
            sampling_shape,
            cfg.sampling.predictor,
            cfg.sampling.steps,
            cfg.sampling.noise_removal,
            sampling_eps,
            device,
        )

    metric = get_alignment_metric(
        model_name="intfloat/e5-base-v2",
        use_sentence_transformers=True,
        device=str(device),
    )

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")

    while state["step"] < num_train_steps + 1:
        step = state["step"]

        if cfg.data.trainset.name != "text8":
            # batch = next(train_iter)["input_ids"].to(device)
            # batch = next(train_iter)["text_input_ids"].to(device)
            batch_data = next(train_iter)
            text = batch_data["text_input_ids"].to(device)
            text_mask = batch_data["text_attention_mask"].to(device)
            style_caption = batch_data["style_caption_input_ids"].to(device)
            style_caption_mask = batch_data["style_caption_attention_mask"].to(device)
        else:
            pass
            # batch = next(train_iter).to(device)
        loss = train_step_fn(state, text, text_mask, style_caption, style_caption_mask)

        # flag to see if there was movement i.e. a full batch got computed
        if step != state["step"]:
            if step % cfg.training.log_freq == 0:
                dist.all_reduce(loss)
                loss /= world_size

                mprint("step: %d, training_loss: %.5e" % (step, loss.item()))

                # Log training loss to TensorBoard
                if rank == 0:
                    writer.add_scalar("loss/train", loss.item(), step)
                    # Log learning rate
                    current_lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar("training/learning_rate", current_lr, step)
                    # Log gradient norm if available
                    total_norm = 0
                    for p in score_model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm**0.5
                    writer.add_scalar("training/gradient_norm", total_norm, step)

            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0:
                if cfg.data.validset.name != "text8":
                    # eval_batch = next(eval_iter)["text_input_ids"].to(device)
                    eval_batch_data = next(eval_iter)
                    eval_text = eval_batch_data["text_input_ids"].to(device)
                    eval_text_mask = eval_batch_data["text_attention_mask"].to(device)
                    eval_style_caption = eval_batch_data["style_caption_input_ids"].to(
                        device
                    )
                    eval_style_caption_mask = eval_batch_data[
                        "style_caption_attention_mask"
                    ].to(device)
                else:
                    pass
                    # eval_batch = next(train_iter).to(device)
                eval_loss = eval_step_fn(
                    state,
                    eval_text,
                    eval_text_mask,
                    eval_style_caption,
                    eval_style_caption_mask,
                )

                dist.all_reduce(eval_loss)
                eval_loss /= world_size

                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))

                # Log evaluation loss to TensorBoard
                if rank == 0:
                    writer.add_scalar("loss/eval", eval_loss.item(), step)

            if (
                step > 0
                and step % cfg.training.snapshot_freq == 0
                or step == num_train_steps
            ):
                # Save the checkpoint.
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(
                        os.path.join(checkpoint_dir, f"checkpoint_{save_step}.pth"),
                        state,
                    )

                # Generate and save samples
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    if cfg.data.validset.name != "text8":
                        # eval_batch = next(eval_iter)["text_input_ids"].to(device)
                        eval_batch_data = next(eval_iter)
                        eval_text = eval_batch_data["text_input_ids"][
                            : sampling_shape[0]
                        ].to(device)
                        eval_text_mask = eval_batch_data["text_attention_mask"][
                            : sampling_shape[0]
                        ].to(device)
                        eval_style_caption = eval_batch_data["style_caption_input_ids"][
                            : sampling_shape[0]
                        ].to(device)
                        eval_style_caption_mask = eval_batch_data[
                            "style_caption_attention_mask"
                        ][: sampling_shape[0]].to(device)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample = sampling_fn(
                        score_model,
                        eval_text_mask,
                        eval_style_caption,
                        eval_style_caption_mask,
                    )
                    ema.restore(score_model.parameters())

                    def truncate_at_eos(batch_ids, eos_id):
                        # batch_ids: torch.Tensor [B, T]
                        output = []
                        for row in batch_ids.tolist():
                            if eos_id in row:
                                k = row.index(eos_id)
                                output.append(row[:k])
                            else:
                                output.append(row)
                        return output

                    sample_trunc = truncate_at_eos(sample, tokenizer_text.eos_token_id)
                    sentences = tokenizer_text.batch_decode(sample_trunc)
                    captions = tokenizer_caption.batch_decode(
                        eval_style_caption, skip_special_tokens=True
                    )

                    file_name = os.path.join(this_sample_dir, f"sample_{rank}.json")
                    import json

                    results = [
                        {"caption": cap, "text": txt}
                        for cap, txt in zip(captions, sentences)
                    ]
                    with open(file_name, "w", encoding="utf-8") as file:
                        json.dump(results, file, indent=2, ensure_ascii=False)

                    def extract_body(sentences):
                        result = []
                        for s in sentences:
                            if "Body:" in s:
                                result.append(s.split("Body:", 1)[1].lstrip())
                            else:
                                result.append(s)
                        return result

                    similarity_scores = metric.score_batch(
                        captions, extract_body(sentences)
                    )
                    avg_similarity = sum(similarity_scores) / len(similarity_scores)
                    mprint(
                        f"Step {step}: Average Similarity Score of generated samples: {avg_similarity:.4f}"
                    )
                    # Log average similarity score to TensorBoard
                    if rank == 0:
                        writer.add_scalar(
                            "eval/avg_similarity_score", avg_similarity, step
                        )

                    if cfg.eval.perplexity:
                        with torch.inference_mode():
                            eval_model = get_eval_lm("gpt2-large", device)

                            batch_size = cfg.eval.perplexity_batch_size
                            num_samples = sample.size(0)

                            total_loss = torch.zeros(1, device=device)
                            total_tokens = torch.zeros(1, device=device)

                            eos_token_id = tokenizer_text.eos_token_id

                            for start in range(0, num_samples, batch_size):
                                end = min(start + batch_size, num_samples)
                                s = sample[start:end]  # (b, T)

                                if s.size(0) == 0:
                                    continue

                                outputs = eval_model(s)
                                logits = outputs.logits  # (b, T, V)

                                logits = logits[:, :-1, :].contiguous()  # (b, T-1, V)
                                targets = s[:, 1:]  # (b, T-1)

                                vocab_size = logits.size(-1)
                                logits_flat = logits.view(-1, vocab_size)
                                targets_flat = targets.reshape(-1)

                                token_losses = F.cross_entropy(
                                    logits_flat,
                                    targets_flat,
                                    reduction="none",
                                ).view_as(
                                    targets
                                )  # (b, T-1)

                                eos_mask = (targets == eos_token_id).int()  # (b, T-1)
                                cumsum = eos_mask.cumsum(dim=-1)
                                valid_mask = (cumsum == 0).to(token_losses.dtype)

                                masked_losses = token_losses * valid_mask

                                batch_loss_sum = masked_losses.sum()
                                batch_token_sum = valid_mask.sum()

                                total_loss += batch_loss_sum
                                total_tokens += batch_token_sum

                            total_tokens = total_tokens.clamp_min(1)

                            if dist.is_available() and dist.is_initialized():
                                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                                dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

                            mean_loss = total_loss / total_tokens
                            total_perplexity = mean_loss.exp()

                            mprint(
                                f"Generative Perplexity at step: {step}. Perplexity: {total_perplexity.item():.3f}."
                            )

                            if rank == 0:
                                writer.add_scalar(
                                    "eval/perplexity", total_perplexity.item(), step
                                )

                            del logits, token_losses, masked_losses, outputs

                    dist.barrier()

    # Close TensorBoard writer
    if rank == 0:
        writer.close()
        mprint("TensorBoard writer closed.")
