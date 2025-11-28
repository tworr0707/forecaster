import datetime as dt
import re
import os
import traceback
from typing import List, Optional, Any

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch

from logger import setup_logger
from agents_vllm import Agent
from config import DEFAULT_MODEL_VLLM
from database import Database
from semanticretriever import SemanticRetriever
from utils import calculate_expected_value, calculate_entropy, infer_likelihood, infer_confidence

logger = setup_logger(__name__)
plt.style.use('dark_background')

class Ensemble:
    """
    Ensemble combines forecasts from multiple Agents and produces a final probability distribution.
    """
    def __init__(self) -> None:
        default_models = [
            DEFAULT_MODEL_VLLM,
            "phi4",
            "llama-8B",
        ]
        self.agents: List[Any] = [Agent(model=m) for m in default_models]

        if not self.agents:
            raise RuntimeError("No agents loaded.")

        self.context_limit = min(
            agent.context_limit for agent in self.agents if hasattr(agent, "context_limit")
        )
        self.db = Database()
        self.retriever = SemanticRetriever(get_new_articles=True)
        self.ensemble_df: Optional[pd.DataFrame] = None

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.plot_path = os.path.join(base_dir, "plots")

    def forecast(self, query: str, image_files: Optional[List[str]] = None) -> None:
        """Generate and persist an ensemble forecast for the supplied query."""
        try:
            logger.info("User query: '%s'", query)
            logger.info("Generating ensemble forecast…")
            image_files = image_files or []
            ensemble_df = pd.DataFrame(index=range(101))

            full_context: Optional[str] = None
            try:
                full_context = self.retriever.get_context(query)
            except Exception as e:
                logger.error(
                    "Failed to retrieve context for query '%s': %s\n%s",
                    query,
                    e,
                    traceback.format_exc(),
                )
                full_context = None

            context_chunks: List[Optional[str]] = [None]
            if full_context:
                context_chunks = self.chunk_context(full_context)

            agent_idx = 1
            time_now = dt.datetime.now()

            for agent in self.agents:
                try:
                    agent.start_forecast()
                    forecasts: List[pd.Series] = []
                    context_idx = 1
                    for context in (context_chunks if context_chunks[0] is not None else [None]):
                        logger.info(
                            "Running agent %d for context chunk %d of %d…",
                            agent_idx,
                            context_idx,
                            len(context_chunks),
                        )
                        forecast_df: Optional[pd.DataFrame] = None
                        try:
                            forecast_df = agent.forecast(query=query, context=context)
                        except Exception as e:
                            logger.error(
                                "Forecast failed for agent %d chunk %d: %s\n%s",
                                agent_idx,
                                context_idx,
                                e,
                                traceback.format_exc(),
                            )
                            context_idx += 1
                            continue

                        if forecast_df is None or "probs" not in forecast_df.columns:
                            logger.warning(
                                "Agent %d returned invalid forecast for chunk %d; skipping.",
                                agent_idx,
                                context_idx,
                            )
                            context_idx += 1
                            continue

                        forecasts.append(forecast_df["probs"])
                        try:
                            probs_tensor = torch.tensor(
                                forecast_df["probs"].to_numpy(), dtype=torch.float32
                            )
                            expected_value = calculate_expected_value(probs_tensor)
                            entropy = calculate_entropy(probs_tensor)
                            logger.info(
                                "Agent %d (%s) => Expected likelihood %.0f%%, Entropy %.3f",
                                agent_idx,
                                getattr(agent, "forecast_model_path", "unknown_model"),
                                expected_value,
                                entropy,
                            )
                            try:
                                self.db.store_forecast(
                                    date_time=time_now,
                                    query=query,
                                    model=getattr(agent, "forecast_model_path", "unknown_model"),
                                    chunk=context,
                                    expected_value=expected_value,
                                    entropy=entropy,
                                )
                            except Exception as db_err:
                                logger.error(
                                    "Failed to store forecast for agent %d: %s\n%s",
                                    agent_idx,
                                    db_err,
                                    traceback.format_exc(),
                                )
                        except Exception as metric_err:
                            logger.error(
                                "Failed to calculate metrics for agent %d: %s\n%s",
                                agent_idx,
                                metric_err,
                                traceback.format_exc(),
                            )
                        context_idx += 1

                    if forecasts:
                        avg_forecast = pd.concat(forecasts, axis=1).mean(axis=1)
                        if avg_forecast.sum() > 0:
                            avg_forecast = avg_forecast / avg_forecast.sum()
                        ensemble_df[f"agent_{agent_idx}"] = avg_forecast
                    else:
                        logger.warning("No valid forecasts obtained from agent %d.", agent_idx)
                    agent_idx += 1
                except Exception as e:
                    model_path = getattr(agent, "forecast_model_path", "unknown_model")
                    logger.error(
                        "Agent %d (%s) failed: %s\n%s",
                        agent_idx,
                        model_path,
                        e,
                        traceback.format_exc(),
                    )
                    agent_idx += 1
                finally:
                    try:
                        agent.stop_forecast()
                    except Exception as stop_err:
                        logger.error(
                            "Failed to stop forecast for agent %d: %s\n%s",
                            agent_idx,
                            stop_err,
                            traceback.format_exc(),
                        )

            logger.info("Forecasts complete!")

            try:
                ensemble_df["ensemble_forecast"] = self.ensemble_method(ensemble_df)
            except Exception as e:
                logger.error("Failed to compute ensemble forecast: %s\n%s", e, traceback.format_exc())

            self.ensemble_df = ensemble_df

            try:
                probs_tensor = torch.tensor(
                    ensemble_df["ensemble_forecast"].to_numpy(), dtype=torch.float32
                )
                ensemble_expected = calculate_expected_value(probs_tensor)
                ensemble_entropy = calculate_entropy(probs_tensor)
                try:
                    self.db.store_ensemble(
                        date_time=time_now,
                        query=query,
                        expected_value=ensemble_expected,
                        entropy=ensemble_entropy,
                    )
                except Exception as db_err:
                    logger.error(
                        "Failed to store ensemble forecast: %s\n%s",
                        db_err,
                        traceback.format_exc(),
                    )
            except Exception as metric_err:
                logger.error(
                    "Failed to calculate ensemble metrics: %s\n%s",
                    metric_err,
                    traceback.format_exc(),
                )

            self.plot_output(ensemble_df, query, time_now)

            for agent in self.agents:
                try:
                    agent.stop_forecast()
                except Exception as e:
                    logger.error("Cleanup: failed to stop forecast for agent: %s\n%s", e, traceback.format_exc())
                try:
                    if hasattr(agent, "stop_logic"):
                        agent.stop_logic()
                except Exception as e:
                    logger.error("Cleanup: failed to stop logic for agent: %s\n%s", e, traceback.format_exc())
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
        except Exception as e:
            logger.error("Unexpected error during forecast execution: %s\n%s", e, traceback.format_exc())

    def chunk_context(self, context: str) -> List[str]:
        """
        Split context into chunks so that each chunk does not exceed the context limit.
        """
        chunks = []
        start = 0
        while start < len(context):
            end = min(start + self.context_limit, len(context))
            chunks.append(context[start:end])
            start = end
        logger.info(f'Splitting context into {len(chunks)} chunks.')
        return chunks

    def ensemble_method(self, ensemble_df: pd.DataFrame, method: str = 'entropy_weighted') -> pd.Series:
        """
        Combine agent forecasts. By default, forecasts are weighted inversely to their entropy.
        """
        try:
            forecast_columns = [col for col in ensemble_df.columns if col.startswith('agent_')]
            if not forecast_columns:
                logger.warning("No agent forecasts were available for ensembling.")
                return pd.Series(np.zeros(ensemble_df.shape[0]), index=ensemble_df.index)

            if method == 'entropy_weighted':
                entropies = []
                for col in forecast_columns:
                    try:
                        tensor = torch.tensor(ensemble_df[col].to_numpy(), dtype=torch.float32)
                        entropies.append(calculate_entropy(tensor))
                    except Exception as e:
                        logger.error(
                            "Failed to calculate entropy for column %s: %s\n%s",
                            col,
                            e,
                            traceback.format_exc(),
                        )
                        entropies.append(float('inf'))
                weights = np.array([np.exp(-e) if e != float('inf') else 0.0 for e in entropies])
                if weights.sum() != 0:
                    weights /= weights.sum()
                else:
                    weights = np.ones_like(weights) / len(weights)
                weighted_forecast = np.zeros(ensemble_df.shape[0])
                for idx, col in enumerate(forecast_columns):
                    weighted_forecast += weights[idx] * ensemble_df[col].to_numpy()
                if weighted_forecast.sum() != 0:
                    weighted_forecast /= weighted_forecast.sum()
                return pd.Series(weighted_forecast, index=ensemble_df.index)

            ensemble_result = ensemble_df[forecast_columns].mean(axis=1)
            if ensemble_result.sum() != 0:
                ensemble_result /= ensemble_result.sum()
            return ensemble_result
        except Exception as e:
            logger.error("Failed to combine ensemble forecasts: %s\n%s", e, traceback.format_exc())
            return pd.Series(np.zeros(ensemble_df.shape[0]), index=ensemble_df.index)

    def plot_output(self, ensemble_df: pd.DataFrame, query: str, prediction_time: dt.datetime) -> None:
        """
        Plot the final ensemble forecast along with a yardstick visualisation.
        """
        try:
            ensemble_df.index = ensemble_df.index.astype(int)
            ensemble_forecast = torch.tensor(
                ensemble_df['ensemble_forecast'].to_numpy(), dtype=torch.float32
            )
            expected_value = calculate_expected_value(ensemble_forecast)
            pred_time_str = prediction_time.strftime('%Y%m%d_%H%M%S')

            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                gridspec_kw={'height_ratios': [3, 1]},
                figsize=(25, 10)
            )
            sns.barplot(
                x=list(ensemble_df.index),
                y=100 * ensemble_df['ensemble_forecast'],
                errorbar=None,
                ax=ax1
            )
            ax1.set_title(
                f"{query}\n"
                f"{infer_likelihood(ensemble_forecast)} with {infer_confidence(ensemble_forecast)}\n"
                f"Entropy: {calculate_entropy(ensemble_forecast):.4f}. Forecast on {pred_time_str}"
            )
            ax1.set_xlabel('Probability of query (%)')
            ax1.set_ylabel('Probability (%)')
            ax1.set_xlim(0, 100)
            ax1.set_xticks(range(0, 101, 10))
            if ensemble_df['ensemble_forecast'].sum() > 0:
                ax1_twin = ax1.twinx()
                sns.kdeplot(
                    x=ensemble_df.index,
                    weights=ensemble_df['ensemble_forecast'],
                    clip=(0, 100),
                    ax=ax1_twin,
                    color='orange'
                )
                ax1_twin.set_ylabel('Density')

            categories = [
                "Remote chance\n(0-5%)",
                "Highly unlikely\n(10-20%)",
                "Unlikely\n(25-35%)",
                "Realistic possibility\n(40-50%)",
                "Likely or probably\n(55-75%)",
                "Highly likely\n(80-90%)",
                "Almost certain\n(95-100%)",
            ]
            ranges = [(0, 5), (10, 20), (25, 35), (40, 50), (55, 75), (80, 90), (95, 100)]
            colors = ["#003366", "#0B5394", "#3C78D8", "#6FA8DC", "#93C47D", "#6AA84F", "#38761D"]

            ax2.set_xlim(0, 100)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            for (start, end), category, color in zip(ranges, categories, colors):
                width = end - start
                rect = Rectangle((start, 0.3), width, 0.7, color=color, ec="black", lw=1.5)
                ax2.add_patch(rect)
                ax2.text(
                    start + width / 2,
                    0.65,
                    category,
                    color='white',
                    ha='center',
                    va='center',
                    fontsize=10,
                    fontweight='bold'
                )
            ax2.plot([expected_value, expected_value], [0.3, 1], color='red', lw=3, linestyle='--')
            ax2.text(
                expected_value,
                0.15,
                f"{expected_value:.1f}%",
                color='red',
                ha='center',
                fontsize=12,
                fontweight='bold'
            )
            plt.tight_layout()

            safe_query = re.sub(r'[^a-zA-Z0-9 _-]', '', query).replace(' ', '_')
            try:
                os.makedirs(self.plot_path, exist_ok=True)
            except Exception as e:
                logger.error("Failed to create plot directory '%s': %s\n%s", self.plot_path, e, traceback.format_exc())

            plot_file = os.path.join(self.plot_path, f"{safe_query}_{pred_time_str}.png")
            try:
                plt.savefig(plot_file)
                logger.info("Plot saved to %s", plot_file)
            except (PermissionError, OSError) as e:
                logger.error("Failed to save plot to '%s': %s", plot_file, e)
            finally:
                plt.close(fig)
        except Exception as e:
            logger.error("Failed to generate plot output: %s\n%s", e, traceback.format_exc())
