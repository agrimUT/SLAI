""" File to store names for different metrics captured """

import enum


class OperationMetrics(enum.Enum):
    MLP_UP_PROJ = "mlp_up_proj"
    MLP_UP_PROJ_ALL_GATHER = "mlp_up_proj_all_gather"
    MLP_ACTIVATION = "mlp_activation"
    MLP_DOWN_PROJ = "mlp_down_proj"
    MLP_DOWN_PROJ_ALL_REDUCE = "mlp_down_proj_all_reduce"
    ATTN_PRE_PROJ = "attn_pre_proj"
    ATTN_PRE_PROJ_ALL_GATHER = "attn_pre_proj_all_gather"
    ATTN_POST_PROJ = "attn_post_proj"
    ATTN_POST_PROJ_ALL_REDUCE = "attn_post_proj_all_reduce"
    ATTN_KV_CACHE_SAVE = "attn_kv_cache_save"
    ATTN = "attn"
    ATTN_PREFILL = "attn_prefill"
    ATTN_DECODE = "attn_decode"
    ATTN_ROPE = "attn_rope"
    ATTN_INPUT_RESHAPE = "attn_input_reshape"
    ATTN_OUTPUT_RESHAPE = "attn_output_reshape"
    EMBED_LINEAR = "embed_linear"
    EMBED_ALL_REDUCE = "embed_all_reduce"
    LM_HEAD_LINEAR = "lm_head_linear"
    LM_HEAD_ALL_GATHER = "lm_head_all_gather"
    INPUT_LAYERNORM = "input_layernorm"
    POST_ATTENTION_LAYERNORM = "post_attention_layernorm"
    NORM = "norm"
    ADD = "add"
    NCCL_SEND = "nccl_send"
    NCCL_RECV = "nccl_recv"


class CpuOperationMetrics(enum.Enum):
    SCHEDULE = "schedule"
    SAMPLER_E2E = "sample_e2e"
    PREPARE_INPUTS_E2E = "prepare_inputs_e2e"
    MODEL_EXECUTION_E2E = "model_execution_e2e"
    PROCESS_MODEL_OUTPUTS = "process_model_outputs"


class SequenceMetricsTimeDistributions(enum.Enum):
    REQUEST_E2E_TIME = "request_e2e_time"
    REQUEST_E2E_TIME_NORMALIZED = "request_e2e_time_normalized"
    REQUEST_E2E_TIME_PIECEWISE_NORMALIZED = "request_e2e_time_piecewise_normalized"
    REQUEST_EXECUTION_TIME = "request_execution_time"
    REQUEST_EXECUTION_TIME_NORMALIZED = "request_execution_time_normalized"
    REQUEST_PREEMPTION_TIME = "request_preemption_time"
    REQUEST_SCHEDULING_DELAY = "request_scheduling_delay"
    REQUEST_EXECUTION_PLUS_PREEMPTION_TIME = "request_execution_plus_preemption_time"
    REQUEST_EXECUTION_PLUS_PREEMPTION_TIME_NORMALIZED = (
        "request_execution_plus_preemption_time_normalized"
    )
    PREFILL_TIME_E2E = "prefill_e2e_time"
    PREFILL_TIME_E2E_NORMALIZED = "prefill_e2e_time_normalized"
    PREFILL_TIME_E2E_PIECEWISE_NORMALIZED = "prefill_e2e_time_piecewise_normalized"
    PREFILL_TIME_EXECUTION_PLUS_PREEMPTION = "prefill_time_execution_plus_preemption"
    PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED = (
        "prefill_time_execution_plus_preemption_normalized"
    )
    DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED = (
        "decode_time_execution_plus_preemption_normalized"
    )
    REQUEST_ARRIVAL_TIME = "request_arrival_time"



class TokenMetricsTimeDistribution(enum.Enum):
    DECODE_TOKEN_EXECUTION_PLUS_PREEMPTION_TIME = (
        "decode_token_execution_plus_preemption_time"
    )
    DECODE_TOKEN_EXECUTION_PLUS_PREEMPTION_TIME_STRICT = (
        "decode_token_execution_plus_preemption_time_strict"
    )
    DECODE_TOKEN_EXECUTION_PLUS_PREEMPTION_TIME_RELAXED = (
        "decode_token_execution_plus_preemption_time_relaxed"
    )


class TokenMetricsTimeList(enum.Enum):
    DECODE_TOKEN_EXECUTION_PLUS_PREEMPTION_TIME_LIST = (
        "decode_token_execution_plus_preemption_time_list"
    )
    


class SequenceMetricsHistogram(enum.Enum):
    REQUEST_INTER_ARRIVAL_DELAY = "request_inter_arrival_delay"
    REQUEST_NUM_TOKENS = "request_num_tokens"
    REQUEST_PREFILL_TOKENS = "request_num_prefill_tokens"
    REQUEST_DECODE_TOKENS = "request_num_decode_tokens"
    REQUEST_PD_RATIO = "request_pd_ratio"
    REQUEST_NUM_RESTARTS = "request_num_restarts"
    REQUEST_NUM_PAUSES = "request_num_pauses"
    REQUEST_NUM_IGNORED = "request_num_ignored"
    REQUEST_TIME_BETWEEN_TOKENS = "request_time_between_tokens"  
    REQUEST_PREFILL_E2E_DEADLINE = "prefill_e2e_deadline"
    REQUEST_IS_STRICT_PREFILL = "is_strict_prefill"


class BatchMetricsCountDistribution(enum.Enum):
    BATCH_NUM_TOKENS = "batch_num_tokens"
    BATCH_NUM_PREFILL_TOKENS = "batch_num_prefill_tokens"
    BATCH_NUM_DECODE_TOKENS = "batch_num_decode_tokens"
    BATCH_SIZE = "batch_size"
    BATCH_NUM_TIME_CRITICAL_DECODE_TOKENS = "batch_num_time_critical_decode_tokens"
    BATCH_NUM_NONCRITICAL_DECODE_TOKENS = "batch_num_noncritical_decode_tokens"
    BATCH_NUM_PREEMPTED_SEQ_PREFILL = "batch_num_preempted_seq_prefill"
    BATCH_NUM_PREEMPTED_SEQ_DECODE = "batch_num_preempted_seq_decode"
    BATCH_NUM_GPU_BLOCKS_USED = "batch_num_gpu_blocks_used"


class BatchMetricsTimeDistribution(enum.Enum):
    BATCH_EXECUTION_TIME = "batch_execution_time"
    INTER_BATCH_DELAY = "inter_batch_delay"


class CompletionMetricsTimeSeries(enum.Enum):
    REQUEST_ARRIVAL = "request_arrival"
    REQUEST_COMPLETION = "request_completion"
    PREFILL_COMPLETIONS = "prefill_completion"
    DECODE_COMPLETIONS = "decode_completion"
    DECODE_REQUEST_COMPLETIONS = "decode_request_completion"
    WAITING_PREFILL_QUEUE_SIZE = "waiting_prefill_queue_size"
    WAITING_DECODE_QUEUE_SIZE  = "waiting_decode_queue_size"
    GPU_BLOCK_UTIL_PERCENT = "gpu_block_util_percent" 
    GPU_SM_UTIL_PERCENT    = "gpu_sm_util_percent"
    DECODE_TOKENS_IN_BATCH = "num_of_decode_tokens_in_batch_utilizing_gpu_compute"
    PREFILL_TOKENS_IN_BATCH = "num_of_prefill_tokens_in_batch_utilizing_gpu_compute"
    GPU_ACTIVE_PREFILL_SEQ  = "num_of_active_prefill_seq_on_gpu_memory"
    GPU_ACTIVE_DECODE_SEQ   = "num_of_active_decode_seq_on_gpu_memory"



