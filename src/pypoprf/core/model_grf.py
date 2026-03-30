import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import rasterio
import threading
from typing import Tuple, Optional
from rasterio.windows import Window
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ..config.settings import Settings
from ..utils.joblib_manager import joblib_resources
from ..utils.logger import get_logger
from ..utils.matplotlib_utils import with_non_interactive_matplotlib
from ..utils.raster_processing import progress_bar
from ..utils.raster import get_windows
from .grf import PyGRFBuilder
from concurrent.futures import ThreadPoolExecutor

logger = get_logger()


class ModelGRF:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.selected_features = None
        self.target_mean = None
        self.local_weight = 0.5
        self.grf_bandwidth = None
        self.grf_kernel = 'adaptive'
        self.grf_train_weighted = True
        self.grf_predict_weighted = True
        self.grf_resampled = True
        self.grf_n_estimators = 80
        self.grf_max_features = 1.0
        self.grf_random_state = 0
        self.grf_perm_repeats = 8
        self.grf_perm_n_jobs = 1
        self.grf_progress_step = 50
        self.grf_neighbor_cache = None
        self.grf_predict_n_jobs = 1
        self.grf_rf_n_jobs = 1
        self.coord_mode = 'pixel'

        self.output_dir = Path(settings.work_dir) / 'output'
        self.output_dir.mkdir(exist_ok=True)

        self._load_grf_params_from_settings()

    def _load_grf_params_from_settings(self) -> None:
        grf_cfg = getattr(self.settings, 'grf', None)
        if grf_cfg is None or not isinstance(grf_cfg, dict):
            grf_cfg = {}

        self.local_weight = float(grf_cfg.get('local_weight', 0.5))
        self.local_weight = min(max(self.local_weight, 0.0), 1.0)

        cfg_bw = grf_cfg.get('band_width', None)
        self.grf_bandwidth = int(cfg_bw) if cfg_bw is not None else None

        self.grf_kernel = str(grf_cfg.get('kernel', 'adaptive'))
        if self.grf_kernel not in ['adaptive', 'fixed']:
            self.grf_kernel = 'adaptive'

        self.grf_train_weighted = bool(grf_cfg.get('train_weighted', True))
        self.grf_predict_weighted = bool(grf_cfg.get('predict_weighted', True))
        self.grf_resampled = bool(grf_cfg.get('resampled', True))
        self.grf_n_estimators = max(10, int(grf_cfg.get('n_estimators', 80)))
        self.grf_max_features = grf_cfg.get('max_features', 1.0)
        self.grf_random_state = int(grf_cfg.get('random_state', 0))
        self.grf_perm_repeats = max(1, int(grf_cfg.get('perm_repeats', 8)))
        self.grf_perm_n_jobs = int(grf_cfg.get('perm_n_jobs', 1))
        if self.grf_perm_n_jobs == 0:
            self.grf_perm_n_jobs = 1
        self.grf_progress_step = max(1, int(grf_cfg.get('progress_step', 50)))
        self.grf_predict_n_jobs = max(1, int(grf_cfg.get('predict_n_jobs', 1)))
        self.grf_rf_n_jobs = max(1, int(grf_cfg.get('rf_n_jobs', 1)))

        cache_setting = grf_cfg.get('neighbor_cache', None)
        if cache_setting:
            cache_path = Path(cache_setting)
            if not cache_path.is_absolute():
                cache_path = Path(self.settings.work_dir) / cache_path
            self.grf_neighbor_cache = str(cache_path)

    def _resolve_grf_bandwidth(self, sample_count: int) -> int:
        if sample_count <= 2:
            return 1
        if self.grf_bandwidth is not None:
            return max(1, min(int(self.grf_bandwidth), sample_count - 1))
        lower = min(20, sample_count - 1)
        upper = max(lower, int(round(sample_count * 0.2)))
        return min(max(lower, upper), sample_count - 1)

    def _allocate_prediction_threads(self) -> tuple[int, int]:
        total_workers = max(1, int(self.settings.max_workers))
        requested_inner = max(1, int(self.grf_predict_n_jobs))

        if total_workers == 1:
            return 1, 1

        if requested_inner <= 1:
            return total_workers, 1

        return 1, min(requested_inner, total_workers)

    def _window_pixel_coordinates(self, window: Window, transform=None, coord_mode: str = 'pixel') -> pd.DataFrame:
        row_start = int(window.row_off)
        col_start = int(window.col_off)
        row_stop = row_start + int(window.height)
        col_stop = col_start + int(window.width)

        rows = np.arange(row_start, row_stop)
        cols = np.arange(col_start, col_stop)
        yy, xx = np.meshgrid(rows, cols, indexing='ij')

        if coord_mode == 'geospatial' and transform is not None:
            x_geo = transform.c + (xx + 0.5) * transform.a + (yy + 0.5) * transform.b
            y_geo = transform.f + (xx + 0.5) * transform.d + (yy + 0.5) * transform.e
            return pd.DataFrame({'x': x_geo.ravel().astype(float), 'y': y_geo.ravel().astype(float)})

        return pd.DataFrame({'x': xx.ravel().astype(float), 'y': yy.ravel().astype(float)})

    def _get_zone_coordinates(self, zone_ids: pd.Series) -> pd.DataFrame:
        coord_acc = {}

        with rasterio.open(self.settings.mastergrid, 'r') as mst:
            nodata = mst.nodata
            windows = get_windows(mst, self.settings.block_size if self.settings.block_size else (512, 512))

            for window in windows:
                zone_arr = mst.read(1, window=window)
                valid = zone_arr != nodata
                if not np.any(valid):
                    continue

                local_rows, local_cols = np.where(valid)
                zone_vals = zone_arr[valid]
                global_rows = local_rows + int(window.row_off)
                global_cols = local_cols + int(window.col_off)

                tmp = pd.DataFrame({
                    'zone': zone_vals,
                    'x_sum': global_cols.astype(float),
                    'y_sum': global_rows.astype(float),
                })
                grouped = tmp.groupby('zone').agg(
                    x_sum=('x_sum', 'sum'),
                    y_sum=('y_sum', 'sum'),
                    n=('zone', 'size'),
                )

                for zone, row in grouped.iterrows():
                    prev = coord_acc.get(zone, [0.0, 0.0, 0])
                    coord_acc[zone] = [
                        prev[0] + float(row['x_sum']),
                        prev[1] + float(row['y_sum']),
                        prev[2] + int(row['n']),
                    ]

        coord_rows = []
        for zone, vals in coord_acc.items():
            x_sum, y_sum, count = vals
            if count > 0:
                coord_rows.append({'zone': zone, 'x': x_sum / count, 'y': y_sum / count})

        coord_df = pd.DataFrame(coord_rows)
        zone_df = pd.DataFrame({'zone': zone_ids.values})
        merged = zone_df.merge(coord_df, on='zone', how='left')

        if merged['x'].isna().any() or merged['y'].isna().any():
            merged['x'] = merged['x'].fillna(merged['x'].mean())
            merged['y'] = merged['y'].fillna(merged['y'].mean())

        return merged[['x', 'y']]

    def _resolve_training_coordinates(self, data: pd.DataFrame) -> pd.DataFrame:
        lower_map = {c.lower(): c for c in data.columns}

        if 'lon' in lower_map and 'lat' in lower_map:
            lon_col = lower_map['lon']
            lat_col = lower_map['lat']
            logger.info("Using lon/lat columns from training data for GRF coordinates")
            self.coord_mode = 'geospatial'
            return data[[lon_col, lat_col]].rename(columns={lon_col: 'x', lat_col: 'y'})

        if 'x' in lower_map and 'y' in lower_map:
            x_col = lower_map['x']
            y_col = lower_map['y']
            logger.info("Using x/y columns from training data for GRF coordinates")
            self.coord_mode = 'pixel'
            return data[[x_col, y_col]].rename(columns={x_col: 'x', y_col: 'y'})

        if 'id' in data.columns:
            census_path = self.settings.census.get('path') if hasattr(self.settings, 'census') else None
            if census_path is not None and Path(census_path).exists():
                census_df = pd.read_csv(census_path)
                census_cols = {c.lower(): c for c in census_df.columns}

                if 'id' in census_df.columns and 'lon' in census_cols and 'lat' in census_cols:
                    lon_col = census_cols['lon']
                    lat_col = census_cols['lat']
                    logger.info("Using lon/lat columns from census CSV for GRF coordinates")
                    coord_df = census_df[['id', lon_col, lat_col]].copy()
                    coord_df = coord_df.rename(columns={lon_col: 'x', lat_col: 'y'})
                    merged = pd.DataFrame({'id': data['id'].values}).merge(coord_df, on='id', how='left')
                    if merged['x'].notna().all() and merged['y'].notna().all():
                        self.coord_mode = 'geospatial'
                        return merged[['x', 'y']]

            logger.info("No lon/lat in input table; deriving GRF coordinates from mastergrid by id")
            self.coord_mode = 'pixel'
            return self._get_zone_coordinates(data['id'])

        raise ValueError("GRF requires coordinates: provide lon/lat (or x/y), or include id to derive coordinates from mastergrid")

    def train(
        self,
        data: pd.DataFrame,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        log_scale: bool = False,
        save_model: bool = True,
    ) -> None:
        logger.info("Phase 1/4: preparing training data")
        data = data.dropna()
        drop_cols = np.intersect1d(data.columns.values, ['id', 'pop', 'dens'])
        X = data.drop(columns=drop_cols).copy()
        y = data['dens'].values
        coords = self._resolve_training_coordinates(data)

        if log_scale:
            y = np.log(np.maximum(y, 0.1))

        self.target_mean = y.mean()
        self.feature_names = X.columns.values

        if scaler_path is None:
            self.scaler = RobustScaler()
            self.scaler.fit(X)
        else:
            with joblib_resources():
                self.scaler = joblib.load(scaler_path)

        if model_path is None:
            logger.info("Phase 2/4: feature selection started (0%)")
            X_scaled = self.scaler.transform(X)
            baseline_rf = RandomForestRegressor(
                n_estimators=self.grf_n_estimators,
                random_state=self.grf_random_state,
                n_jobs=self.settings.max_workers,
            )
            baseline_rf.fit(X_scaled, y)
            logger.info("Phase 2/4: feature selection baseline model fitted (50%)")

            importances, selected = self._select_features(baseline_rf, X_scaled, y)
            _ = importances
            logger.info("Phase 2/4: feature selection finished (100%)")

            X = X[selected]
            self.selected_features = selected
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.grf_bandwidth = self._resolve_grf_bandwidth(len(X))

            if self.grf_neighbor_cache is None and self.grf_kernel == 'adaptive':
                self.grf_neighbor_cache = str(Path(self.settings.output_dir) / f'grf_neighbors_bw{self.grf_bandwidth}.npz')

            logger.info(
                f"Phase 3/4: GRF training started with n_estimators={self.grf_n_estimators}, "
                f"bandwidth={self.grf_bandwidth}, kernel={self.grf_kernel}"
            )

            grf = PyGRFBuilder(
                band_width=self.grf_bandwidth,
                n_estimators=self.grf_n_estimators,
                max_features=self.grf_max_features,
                kernel=self.grf_kernel,
                train_weighted=self.grf_train_weighted,
                predict_weighted=self.grf_predict_weighted,
                resampled=self.grf_resampled,
                bootstrap=False,
                n_jobs=self.grf_rf_n_jobs,
                random_state=self.grf_random_state,
                neighbor_cache_path=self.grf_neighbor_cache,
                predict_n_jobs=self.grf_predict_n_jobs,
            )
            X_scaled_df = pd.DataFrame(X_scaled, columns=selected)
            y_series = pd.Series(y)

            def _progress(done: int, total: int) -> None:
                if done == 1 or done == total or done % self.grf_progress_step == 0:
                    logger.info(f"Phase 3/4: GRF local model {done}/{total}")

            grf.fit(X_scaled_df, y_series, coords, progress_callback=_progress)
            self.model = grf
            self._evaluate_and_save_metrics(
                X_scaled_df,
                y_series,
                coords,
                log_scale=log_scale,
            )
            logger.info("Phase 3/4: GRF training finished")
        else:
            with joblib_resources():
                self.model = joblib.load(model_path)
            if self.scaler is not None and hasattr(self.scaler, 'feature_names_in_'):
                self.selected_features = self.scaler.feature_names_in_

        if save_model:
            logger.info("Phase 4/4: saving model artifacts")
            with joblib_resources():
                self._save_model()
            logger.info("Phase 4/4: save completed")

    def _select_features(
        self,
        baseline_model: RandomForestRegressor,
        X: np.ndarray,
        y: np.ndarray,
        limit: float = 0.0,
        plot: bool = True,
        save: bool = True,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        names = self.feature_names
        ymean = self.target_mean

        result = permutation_importance(
            baseline_model,
            X,
            y,
            n_repeats=self.grf_perm_repeats,
            n_jobs=self.grf_perm_n_jobs,
            random_state=0,
            scoring='neg_root_mean_squared_error',
        )

        sorted_idx = result.importances_mean.argsort()
        importances = pd.DataFrame(result.importances[sorted_idx].T / ymean, columns=names[sorted_idx])
        selected = importances.columns.values[np.median(importances, axis=0) > limit]

        if plot:
            self._plot_feature_importance(importances, limit)

        if save:
            save_path = Path(self.settings.work_dir) / 'output' / 'feature_importance.csv'
            importances.to_csv(save_path, index=False)

        return importances, selected

    @with_non_interactive_matplotlib
    def _plot_feature_importance(self, importance_df: pd.DataFrame, limit: float) -> None:
        sy = importance_df.shape[1] * 0.25 + 0.5
        fig, ax = plt.subplots(1, 1, figsize=(4, sy), dpi=90)

        importance_df.plot.box(vert=False, whis=5, ax=ax, color='k', sym='.k')
        ax.axvline(x=limit, color='k', linestyle='--', lw=0.5)
        ax.set_xlabel('Decrease in nRMSE')

        plt.tight_layout()
        save_path = Path(self.settings.work_dir) / 'output' / 'feature_selection.png'
        plt.savefig(save_path)
        plt.close()

    @with_non_interactive_matplotlib
    def predict(self, log_scale: bool = False) -> str:
        if self.model is None or self.scaler is None:
            raise RuntimeError('Model not trained. Call train() first.')

        with joblib_resources():
            src = {}
            try:
                for k in self.settings.covariate:
                    src[k] = rasterio.open(self.settings.covariate[k], 'r')

                mst = rasterio.open(self.settings.mastergrid, 'r')
                profile = mst.profile.copy()
                profile.update({
                    'dtype': 'float32',
                    'blockxsize': self.settings.block_size[0],
                    'blockysize': self.settings.block_size[1],
                })

                reading_lock = threading.Lock()
                writing_lock = threading.Lock()
                names = self.selected_features if self.selected_features is not None else self.feature_names
                outfile = Path(self.settings.output_dir) / 'prediction.tif'

                with rasterio.open(outfile, 'w', **profile) as dst:
                    prediction_counter = {'done': 0}
                    prediction_step = max(10, self.grf_progress_step)

                    def process(window):
                        df = pd.DataFrame()
                        with reading_lock:
                            zone_arr = mst.read(1, window=window)
                            for s in src:
                                arr = src[s].read(window=window)[0, :, :]
                                df[s + '_avg'] = arr.flatten()

                        nodata = mst.nodata
                        if nodata is None:
                            valid_mask = np.isfinite(zone_arr)
                        else:
                            valid_mask = zone_arr != nodata

                        pred_nodata = profile.get('nodata', mst.nodata if mst.nodata is not None else -99.0)
                        out_arr = np.full(zone_arr.shape, pred_nodata, dtype=np.float32)
                        valid_idx = np.where(valid_mask.ravel())[0]

                        if len(valid_idx) == 0:
                            with writing_lock:
                                dst.write(out_arr, window=window, indexes=1)
                                prediction_counter['done'] += 1
                            return

                        df = df[names]
                        if len(valid_idx) != len(df):
                            df = df.iloc[valid_idx].reset_index(drop=True)
                        coords_df = self._window_pixel_coordinates(
                            window,
                            transform=mst.transform,
                            coord_mode=self.coord_mode,
                        )
                        if len(valid_idx) != len(coords_df):
                            coords_df = coords_df.iloc[valid_idx].reset_index(drop=True)
                        sx = self.scaler.transform(df)
                        sx_df = pd.DataFrame(sx, columns=names)
                        yp, _, _ = self.model.predict(
                            sx_df,
                            coords_df,
                            local_weight=self.local_weight,
                            predict_n_jobs=effective_inner_jobs,
                        )
                        yp = np.array(yp)

                        if log_scale:
                            yp = np.exp(yp)

                        out_arr.reshape(-1)[valid_idx] = yp.astype(np.float32)
                        with writing_lock:
                            dst.write(out_arr, window=window, indexes=1)
                            prediction_counter['done'] += 1

                    if self.settings.by_block:
                        block_windows = list(dst.block_windows())
                        windows = []
                        for _, window in block_windows:
                            windows.append(window)

                        total_windows = len(windows)
                        window_workers, effective_inner_jobs = self._allocate_prediction_threads()
                        logger.info(
                            f"Phase 5/5: prediction parallelism window_workers={window_workers}, "
                            f"grf_predict_n_jobs={effective_inner_jobs}, max_workers={self.settings.max_workers}"
                        )

                        def process_with_status(window):
                            process(window)
                            done = prediction_counter['done']
                            if done == 1 or done == total_windows or done % prediction_step == 0:
                                logger.info(f"Phase 5/5: Prediction window {done}/{total_windows}")

                        with ThreadPoolExecutor(max_workers=window_workers) as executor:
                            list(progress_bar(
                                executor.map(process_with_status, windows),
                                self.settings.show_progress,
                                len(windows),
                                desc='Prediction',
                            ))
                    else:
                        _, effective_inner_jobs = self._allocate_prediction_threads()
                        full_window = Window(0, 0, dst.width, dst.height)
                        process(full_window)
            finally:
                for k in src:
                    try:
                        src[k].close()
                    except Exception:
                        pass

                if mst is not None:
                    try:
                        mst.close()
                    except Exception:
                        pass

        return str(outfile)

    def _save_model(self) -> None:
        model_path = self.output_dir / 'model.pkl.gz'
        scaler_path = self.output_dir / 'scaler.pkl.gz'

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self, model_path: str, scaler_path: str) -> None:
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = self.scaler.get_feature_names_out()

    def _evaluate_and_save_metrics(
        self,
        X_scaled_df: pd.DataFrame,
        y_true: pd.Series,
        coords: pd.DataFrame,
        log_scale: bool = False,
    ) -> None:
        y_true_arr = np.asarray(y_true, dtype=float)

        y_pred, _, _ = self.model.predict(
            X_scaled_df,
            coords,
            local_weight=self.local_weight,
            predict_n_jobs=self.grf_predict_n_jobs,
        )
        y_pred_arr = np.asarray(y_pred, dtype=float)

        if log_scale:
            y_true_eval = np.exp(y_true_arr)
            y_pred_eval = np.exp(y_pred_arr)
        else:
            y_true_eval = y_true_arr
            y_pred_eval = y_pred_arr

        rmse = float(mean_squared_error(y_true_eval, y_pred_eval) ** 0.5)
        mae = float(mean_absolute_error(y_true_eval, y_pred_eval))
        r2 = float(r2_score(y_true_eval, y_pred_eval))
        y_mean = float(np.mean(y_true_eval)) if len(y_true_eval) > 0 else 0.0
        nrmse = float(rmse / y_mean) if y_mean != 0 else float('nan')

        metrics_df = pd.DataFrame([
            {
                'split': 'train',
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'nrmse': nrmse,
                'target_mean': y_mean,
                'log_scale': bool(log_scale),
            }
        ])

        metrics_path = Path(self.settings.work_dir) / 'output' / 'model_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)

        logger.info(f"Model metrics saved to: {metrics_path}")
        logger.info(
            f"Train metrics - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, nRMSE: {nrmse:.4f}"
        )
