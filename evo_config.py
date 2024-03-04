JOB_NAME = "dit"
DO_ALERT = False

SEQ_LEN = 2048
HIDDEN_SIZE = 1152
NUM_ATTENTION_HEAD = 16
MLP_RATIO = 4
NUM_LAYER = 28
VOCAB_SIZE = 103168

parallel = dict(
    zero1=dict(size=1),
    tensor=dict(size=1, mode='mtp'),
    pipeline=dict(size=1, interleaved_overlap=True),
    sequence_parallel=False,
)

model = dict(
    dtype="torch.bfloat16"
)

data = dict(
    seq_len=SEQ_LEN,
    # micro_num means the number of micro_batch contained in one gradient update
    micro_num=1,
    # packed_length = micro_bsz * SEQ_LEN
    micro_bsz=96,
    # defaults to the value of micro_num
    valid_micro_num=4,
    # defaults to 0, means disable evaluate
    valid_every=0,
    pack_sample_into_one=False,
    total_steps=50000,
    skip_batches="",
    rampup_batch_size="",
    # Datasets with less than 50 rows will be discarded
    min_length=50,
    # train_folder=TRAIN_FOLDER,
    # valid_folder=VALID_FOLDER,
    empty_cache_and_diag_interval=10,
    diag_outlier_ratio=1.1,
)

ckpt = dict(
    enable_save_ckpt=False,  # enable ckpt save.
    save_ckpt_folder='./',  # Path to save training ckpt.
    # load_ckpt_folder= dict(path=MODEL_ONLY_FOLDER, content=["model"], ckpt_type="normal"),
    load_ckpt_folder="local:llm_ckpts/",
    # 'load_ckpt_info' setting guide:
    # 1. the 'path' indicate ckpt path,
    # 2. the 'content‘ means what states will be loaded, support: "model", "sampler", "optimizer", "scheduler", "all"
    # 3. the ’ckpt_type‘ means the type of checkpoint to be loaded, now only 'normal' type is supported.
    load_ckpt_info=dict(path='./'),
    checkpoint_every=10000,
    async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    oss_snapshot_freq=int(10000 / 2),  # snapshot ckpt save frequency.
)

hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=False,
    overlap_sync_param=False,
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
)

dam = dict(
    lr=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=0,  # optimizer_warmup_step
    warmup_ratio=0.01,
    eta_min=1e-5,
    last_epoch=-1,
)
loss = dict(
    label_smoothing=0,
)

adam = dict(
    lr=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
)
beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**16,
        # the minimum loss scale, defaults to None
        min_scale=1,
        # the number of steps to increase loss scale when no overflow occurs
        growth_interval=1000,
    ),
    # the multiplication factor for increasing loss scale, defaults to 2
    growth_factor=2,
    # the multiplication factor for decreasing loss scale, defaults to 0.5
    backoff_factor=0.5,
    # the maximum loss scale, defaults to None
    max_scale=2**24,
    # the number of overflows before decreasing loss scale, defaults to 2
    hysteresis=2,
)