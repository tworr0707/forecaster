from ensemble_torch import Ensemble
from analysis_torch import ForecasterAnalysis
from logger_torch import configure_root_logger


configure_root_logger()


def main() -> None:
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
        except Exception as e:
            print(f"Error for query '{query}': {e}")

        try:
            analysis.plot_charts()
        except Exception as e:
            print(f"Error plotting charts: {e}")


if __name__ == '__main__':
    main()
