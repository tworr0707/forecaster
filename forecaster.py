import sys
import torch
from ensemble import Ensemble
from analysis import ForecasterAnalysis
from logger import configure_root_logger
from config import VLLM_CONFIG, validate_config
from oom_utils import ForecastingOOMError


configure_root_logger()


def main() -> None:
    # Basic environment/config validation
    try:
        validate_config()
    except Exception as e:
        print(f"Config validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA is not available; a GPU is required for vLLM runs.", file=sys.stderr)
        sys.exit(1)
    gpu_count = torch.cuda.device_count()
    tp = VLLM_CONFIG.get("tensor_parallel_size") or gpu_count
    if tp > gpu_count:
        print(f"Configured tensor_parallel_size={tp} exceeds available GPUs={gpu_count}.", file=sys.stderr)
        sys.exit(1)
    # compatibility info for operators
    torch_cuda = torch.version.cuda
    try:
        import vllm
        vllm_ver = getattr(vllm, "__version__", "unknown")
    except Exception:
        vllm_ver = "unknown"
    print(f"Environment: CUDA {torch_cuda}, GPUs detected {gpu_count}, vLLM {vllm_ver}")
    queries = [
        'Will the US attack Iran in 2025?',
        'Will India and Pakistan experience a direct military conflict before 2030?',
        'Will the UK experience a direct military conflict with Russia before 2030?',
        'Will the UK experience a direct military conflict with China before 2030?',
        'Will the UK experience a direct military conflict with North Korea before 2030?',
        'Will the UK experience a direct military conflict with Iran before 2030?',
        'Will China invade Taiwan before 2030?',
        'Will there be a partial ceasefire in Ukraine in 2025?',
        'Will NATO Article V be triggered before 2030?',
        'Will a major cyber attack disrupt or destroy UK critical national infrastructure before 2030?',
        'Will the NHS still exist in 2030?',
        'Will any European nation develop an independent nuclear deterrent separate from NATO before 2030?',
        'Will Vladimir Putin still be Russian President in 2030?',
        'Will Xi Jinping still be Chinese President in 2030?',
    ]

    ensemble = Ensemble()
    analysis = ForecasterAnalysis()

    for query in queries:
        try:
            ensemble.forecast(query)
        except ForecastingOOMError as e:
            print(
                f"GPU OOM while forecasting query '{query}'. Reduce model size/parallelism or adjust VLLM_CONFIG. {e}",
                file=sys.stderr,
            )
            sys.exit(42)
        except Exception as e:
            print(f"Error for query '{query}': {e}")

        try:
            analysis.plot_charts()
        except Exception as e:
            print(f"Error plotting charts: {e}")


if __name__ == '__main__':
    main()
