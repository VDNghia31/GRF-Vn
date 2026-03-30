import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import distance
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample


class PyGRFBuilder:
    def __init__(
        self,
        band_width,
        n_estimators=200,
        max_features=1.0,
        kernel="adaptive",
        train_weighted=True,
        predict_weighted=True,
        resampled=True,
        n_jobs=None,
        bootstrap=True,
        random_state=None,
        neighbor_cache_path=None,
        predict_n_jobs=4,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.band_width = band_width
        self.kernel = kernel
        self.train_weighted = train_weighted
        self.predict_weighted = predict_weighted
        self.n_jobs = n_jobs
        self.bootstrap = bootstrap
        self.resampled = resampled
        self.random_state = random_state
        self.global_model = None
        self.local_models = None
        self.train_data_coords = None
        self.distance_matrix = None
        self.train_tree = None
        self.train_data_columns = None
        self.rf_add_params = kwargs
        self.neighbor_cache_path = neighbor_cache_path
        self.predict_n_jobs = max(1, int(predict_n_jobs))

    def _resolve_bandwidth(self, sample_count: int) -> int:
        if sample_count <= 2:
            return 1
        if self.kernel == "adaptive":
            return max(1, min(int(self.band_width), sample_count - 1))
        return self.band_width

    def fit(self, X_train, y_train, coords, progress_callback=None):
        self.train_data_columns = X_train.columns.tolist()
        sample_count = len(X_train)
        effective_bw = self._resolve_bandwidth(sample_count)

        if self.bootstrap:
            rf_global = RandomForestRegressor(
                bootstrap=self.bootstrap,
                oob_score=True,
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                **self.rf_add_params,
            )
        else:
            rf_global = RandomForestRegressor(
                bootstrap=self.bootstrap,
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                **self.rf_add_params,
            )

        rf_global.fit(X_train, y_train)
        self.global_model = rf_global
        global_oob_prediction = rf_global.oob_prediction_ if self.bootstrap else None

        self.local_models = []

        coords_array = np.array(coords, dtype=np.float64)
        self.train_data_coords = coords_array
        self.train_tree = cKDTree(coords_array)
        self.distance_matrix = distance.cdist(coords_array, coords_array, "euclidean")

        if self.train_weighted:
            if self.kernel == "adaptive":
                bandwidth_array = np.partition(self.distance_matrix, effective_bw, axis=1)[:, effective_bw] * 1.0000001
                weight_matrix = (1 - (self.distance_matrix / bandwidth_array[:, np.newaxis]) ** 2) ** 2
            else:
                weight_matrix = (1 - (self.distance_matrix / self.band_width) ** 2) ** 2

        local_oob_prediction = []

        adaptive_idx_matrix = None
        if self.kernel == "adaptive":
            adaptive_idx_matrix = self._load_or_build_adaptive_neighbors(effective_bw)

        for i in range(sample_count):
            distance_array = self.distance_matrix[i]
            if self.kernel == "adaptive":
                idx = adaptive_idx_matrix[i]
            else:
                idx = np.where(distance_array < self.band_width)[0]
                idx = idx[np.argsort(distance_array[idx])]

            if len(idx) == 0:
                idx = np.array([i])

            if self.train_weighted:
                sample_weights = weight_matrix[i][idx]

            local_X_train = X_train.iloc[idx]
            local_y_train = y_train.iloc[idx]

            if self.bootstrap:
                rf_local = RandomForestRegressor(
                    bootstrap=self.bootstrap,
                    oob_score=True,
                    n_estimators=self.n_estimators,
                    max_features=self.max_features,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    **self.rf_add_params,
                )
            else:
                rf_local = RandomForestRegressor(
                    bootstrap=self.bootstrap,
                    n_estimators=self.n_estimators,
                    max_features=self.max_features,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    **self.rf_add_params,
                )

            if self.train_weighted:
                if self.resampled and len(local_X_train) < 2 * self.n_estimators:
                    resampled_length = min(2 * self.n_estimators, 2 * len(local_X_train)) - len(local_X_train)
                    more_X, more_y, more_w = resample(
                        local_X_train,
                        local_y_train,
                        sample_weights,
                        replace=True,
                        n_samples=resampled_length,
                        random_state=self.random_state,
                    )
                    local_X = pd.concat([local_X_train, more_X], ignore_index=True)
                    local_y = pd.concat([local_y_train, more_y], ignore_index=True)
                    local_w = np.concatenate((sample_weights, more_w))
                    rf_local.fit(local_X, local_y, local_w)
                else:
                    rf_local.fit(local_X_train, local_y_train, sample_weights)
            else:
                if self.resampled and len(local_X_train) < 2 * self.n_estimators:
                    resampled_length = min(2 * self.n_estimators, 2 * len(local_X_train)) - len(local_X_train)
                    more_X, more_y = resample(
                        local_X_train,
                        local_y_train,
                        replace=True,
                        n_samples=resampled_length,
                        random_state=self.random_state,
                    )
                    local_X = pd.concat([local_X_train, more_X], ignore_index=True)
                    local_y = pd.concat([local_y_train, more_y], ignore_index=True)
                    rf_local.fit(local_X, local_y)
                else:
                    rf_local.fit(local_X_train, local_y_train)

            if self.bootstrap:
                local_oob_prediction.append(rf_local.oob_prediction_[0])

            self.local_models.append(rf_local)

            if progress_callback is not None:
                progress_callback(i + 1, sample_count)

        if self.bootstrap:
            return global_oob_prediction, local_oob_prediction

    def _load_or_build_adaptive_neighbors(self, effective_bw: int) -> np.ndarray:
        if self.neighbor_cache_path:
            cache_path = Path(self.neighbor_cache_path)
            if cache_path.exists():
                try:
                    loaded = np.load(cache_path)
                    cached_idx = loaded["idx"]
                    cached_bw = int(loaded["bw"])
                    cached_n = int(loaded["n"])
                    if cached_n == self.distance_matrix.shape[0] and cached_bw == effective_bw:
                        return cached_idx
                except Exception:
                    pass

        idx_matrix = np.argpartition(self.distance_matrix, effective_bw, axis=1)[:, : effective_bw + 1]
        selected_dist = np.take_along_axis(self.distance_matrix, idx_matrix, axis=1)
        row_order = np.argsort(selected_dist, axis=1)
        idx_matrix = np.take_along_axis(idx_matrix, row_order, axis=1)

        if self.neighbor_cache_path:
            cache_path = Path(self.neighbor_cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, idx=idx_matrix, bw=effective_bw, n=self.distance_matrix.shape[0])

        return idx_matrix

    def predict(self, X_test, coords_test, local_weight, predict_n_jobs=None):
        predict_global = self.global_model.predict(X_test).flatten()

        effective_predict_jobs = self.predict_n_jobs if predict_n_jobs is None else max(1, int(predict_n_jobs))

        coords_test_array = np.array(coords_test, dtype=np.float64)

        sample_count = self.train_data_coords.shape[0]
        effective_bw = self._resolve_bandwidth(sample_count)

        if self.kernel == "adaptive" and self.train_tree is not None:
            neighbor_dist, adaptive_predict_idx = self.train_tree.query(coords_test_array, k=effective_bw + 1)
            if effective_bw + 1 == 1:
                neighbor_dist = neighbor_dist[:, np.newaxis]
                adaptive_predict_idx = adaptive_predict_idx[:, np.newaxis]
            bandwidth_array = neighbor_dist[:, -1] * 1.0000001
            if self.predict_weighted:
                weight_matrix = (1 - (neighbor_dist / bandwidth_array[:, np.newaxis]) ** 2) ** 2
            else:
                weight_matrix = None

            if self.predict_weighted:
                n_test, k_neighbors = adaptive_predict_idx.shape
                local_prediction_neighbors = np.empty((n_test, k_neighbors), dtype=np.float32)

                assignments = {}
                for row_idx in range(n_test):
                    for col_idx in range(k_neighbors):
                        model_idx = int(adaptive_predict_idx[row_idx, col_idx])
                        if model_idx not in assignments:
                            assignments[model_idx] = [[], []]
                        assignments[model_idx][0].append(row_idx)
                        assignments[model_idx][1].append(col_idx)

                items = []
                for model_idx, (rows_list, cols_list) in assignments.items():
                    rows_arr = np.array(rows_list, dtype=np.int32)
                    cols_arr = np.array(cols_list, dtype=np.int32)
                    items.append((model_idx, rows_arr, cols_arr))

                def _predict_model_neighbors(item):
                    model_idx, rows_arr, cols_arr = item
                    unique_rows, inverse = np.unique(rows_arr, return_inverse=True)
                    preds = self.local_models[model_idx].predict(X_test.iloc[unique_rows])
                    return rows_arr, cols_arr, preds[inverse]

                if effective_predict_jobs > 1 and len(items) > 1:
                    with ThreadPoolExecutor(max_workers=effective_predict_jobs) as executor:
                        for rows_arr, cols_arr, preds in executor.map(_predict_model_neighbors, items):
                            local_prediction_neighbors[rows_arr, cols_arr] = preds
                else:
                    for item in items:
                        rows_arr, cols_arr, preds = _predict_model_neighbors(item)
                        local_prediction_neighbors[rows_arr, cols_arr] = preds

                denominator = np.sum(weight_matrix, axis=1)
                numerator = np.sum(local_prediction_neighbors * weight_matrix, axis=1)
                first_pred = local_prediction_neighbors[:, 0]
                predict_local_array = np.where(denominator <= 0, first_pred, numerator / denominator)
                predict_local = predict_local_array.tolist()
            else:
                nearest_idx = adaptive_predict_idx[:, 0]
                predict_local_array = np.array([
                    self.local_models[int(model_idx)].predict(X_test.iloc[[i]])[0]
                    for i, model_idx in enumerate(nearest_idx)
                ])
                predict_local = predict_local_array.tolist()
        else:
            distance_matrix_test_to_train = distance.cdist(coords_test_array, self.train_data_coords, "euclidean")

            if effective_predict_jobs > 1 and len(self.local_models) > 1:
                def _predict_one(local_model):
                    local_predict_one = local_model.predict(X_test)
                    return local_predict_one[:, np.newaxis]

                with ThreadPoolExecutor(max_workers=effective_predict_jobs) as executor:
                    local_predict_list = list(executor.map(_predict_one, self.local_models))
            else:
                local_predict_list = []
                for local_model in self.local_models:
                    local_predict_one = local_model.predict(X_test)
                    local_predict_list.append(local_predict_one[:, np.newaxis])
            local_predict_matrix = np.concatenate(local_predict_list, axis=1)

            if self.predict_weighted:
                if self.kernel == "adaptive":
                    bandwidth_array = np.partition(distance_matrix_test_to_train, effective_bw, axis=1)[:, effective_bw] * 1.0000001
                    weight_matrix = (1 - (distance_matrix_test_to_train / bandwidth_array[:, np.newaxis]) ** 2) ** 2
                else:
                    weight_matrix = (1 - (distance_matrix_test_to_train / self.band_width) ** 2) ** 2

            adaptive_predict_idx = None
            if self.predict_weighted and self.kernel == "adaptive":
                adaptive_predict_idx = np.argpartition(distance_matrix_test_to_train, effective_bw, axis=1)[:, : effective_bw + 1]
                selected_dist = np.take_along_axis(distance_matrix_test_to_train, adaptive_predict_idx, axis=1)
                row_order = np.argsort(selected_dist, axis=1)
                adaptive_predict_idx = np.take_along_axis(adaptive_predict_idx, row_order, axis=1)

            predict_local = []
            for i in range(len(X_test)):
                distance_array = distance_matrix_test_to_train[i]
                local_predict_array = local_predict_matrix[i]

                if self.predict_weighted:
                    if self.kernel == "adaptive":
                        idx = adaptive_predict_idx[i]
                    else:
                        idx = np.where(distance_array < self.band_width)[0]
                        idx = idx[np.argsort(distance_array[idx])]

                    if len(idx) == 0:
                        idx = np.array([np.argmin(distance_array)])

                    sample_weights = weight_matrix[i][idx]
                    local_prediction_bandwidth = local_predict_array[idx]
                    denominator = np.sum(sample_weights)
                    if denominator <= 0:
                        this_local_prediction = local_prediction_bandwidth[0]
                    else:
                        this_local_prediction = np.sum(local_prediction_bandwidth * sample_weights) / denominator
                else:
                    this_idx = np.argmin(distance_array)
                    this_local_prediction = local_predict_array[this_idx]

                predict_local.append(this_local_prediction)

            predict_local_array = np.array(predict_local)
        predict_global_array = np.array(predict_global)
        predict_combined = (predict_local_array * local_weight + predict_global_array * (1 - local_weight)).tolist()

        return predict_combined, predict_global, predict_local

    def get_local_feature_importance(self):
        if self.local_models is None:
            return None

        column_list = ["model_index"] + self.train_data_columns
        feature_importance_df = pd.DataFrame(columns=column_list)

        for i in range(len(self.local_models)):
            this_row = [i]
            this_row.extend(self.local_models[i].feature_importances_)
            feature_importance_df = pd.concat(
                [feature_importance_df, pd.DataFrame([this_row], columns=column_list)],
                ignore_index=True,
            )

        return feature_importance_df