from pathlib import Path
import argparse

from pypoprf.config.settings import Settings
from pypoprf.core.feature_extraction import FeatureExtractor
from pypoprf.core.model_grf import ModelGRF
from pypoprf.core.dasymetric import DasymetricMapper
from pypoprf.utils.logger import get_logger

logger = get_logger()


def run_grf(config_file: str, model_path: str | None = None, no_viz: bool = True) -> None:
    settings = Settings.from_file(config_file)

    output_dir = Path(settings.work_dir) / 'output'
    output_dir.mkdir(exist_ok=True)

    feature_extractor = FeatureExtractor(settings)
    model = ModelGRF(settings)

    if model_path:
        features = feature_extractor.get_dummy()
        model.train(
            features,
            model_path=model_path,
            scaler_path=model_path.replace('model', 'scaler'),
            log_scale=settings.log_scale,
            save_model=False,
        )
    else:
        features = feature_extractor.extract()
        model.train(features, log_scale=settings.log_scale)

    prediction_path = model.predict(log_scale=settings.log_scale)
    mapper = DasymetricMapper(settings)
    mapper.map(prediction_path)

    if not no_viz:
        from pypoprf.utils.visualization import Visualizer

        visualizer = Visualizer(settings)
        viz_output = str(output_dir / 'visualization.png')
        visualizer.map_redistribute(
            mastergrid_path=settings.mastergrid,
            probability_path=str(output_dir / 'prediction.tif'),
            normalize_path=str(output_dir / 'normalized_census.tif'),
            population_path=str(output_dir / 'dasymetric.tif'),
            output_path=viz_output,
            vis_params={
                'vmin': [0, 0, 0, 0],
                'vmax': [1300, 250, 1, 250],
                'cmap': 'viridis',
                'titles': ['Zones', 'Probability', 'Normalized Zones', 'Redistributed'],
            },
            dpi=300,
            figsize=(15, 5),
            nodata=-99,
        )

    logger.info('GRF workflow finished successfully')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pypoprf workflow with GRF model in separate runner.')
    parser.add_argument('-c', '--config', required=True, help='Path to configuration file')
    parser.add_argument('-m', '--model', default=None, help='Optional model pickle path')
    parser.add_argument('--viz', action='store_true', help='Enable visualization output')
    args = parser.parse_args()

    run_grf(config_file=args.config, model_path=args.model, no_viz=not args.viz)
