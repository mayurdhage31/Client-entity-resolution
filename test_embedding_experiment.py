"""Offline unit fixtures only. Real MiniLM is checked by the separate CLI flag."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import embedding_features as ef
import pair_features as pf
import run_embedding_experiment as experiment
from test_matcher import normalized, training_features


def context():
    return ef.CacheContext("a" * 40, "unit-preprocessing-v1", "unit-fixture-only", dimension=2)


def extended_training():
    frame, labels = training_features()
    frame[ef.EMBEDDING_NAMES[0]] = [1.0, 0.9, -1.0, -0.8]
    frame[ef.EMBEDDING_NAMES[1]] = [1.0, 0.7, -0.8, -1.0]
    return frame, labels


class EmbeddingFeatureChecks(unittest.TestCase):
    def test_missing_text_is_not_encoded_and_has_zero_similarity(self):
        full = {"a": normalized(), "b": normalized(company_name="alpha")}
        text_records = ef.project_text_records(full)
        pairs = [{"record_id_left": "a", "record_id_right": "b"}]
        base = pf.build_feature_frame(pairs, full)
        frame = ef.extend_pair_features(base, pairs, text_records, {})
        self.assertEqual(frame[ef.EMBEDDING_NAMES[0]].iloc[0], 0)
        self.assertEqual(frame[ef.EMBEDDING_NAMES[1]].iloc[0], 0)
        self.assertEqual(frame["company_name_one_missing"].iloc[0], 1)
        self.assertEqual(frame["address_both_missing"].iloc[0], 1)
        pd.testing.assert_frame_equal(frame.loc[:, list(pf.FEATURE_NAMES)], base)
        with tempfile.TemporaryDirectory() as folder:
            def never_encode(texts, batch_size):
                raise AssertionError("Empty text reached encoder")
            vectors, stats = ef.EmbeddingCache(Path(folder), context()).get_or_encode(["", None, "  "], never_encode)
            self.assertEqual(vectors, {})
            self.assertEqual(stats["cache_state"], "empty")

    def test_symmetry_and_negative_cosine_are_preserved(self):
        records = {"a": normalized(company_name="alpha", address="street one"),
                   "b": normalized(company_name="beta", address="street two")}
        vectors = {"alpha": np.array([1., 0.]), "beta": np.array([-1., 0.]),
                   "street one": np.array([0., 1.]), "street two": np.array([0., -1.])}
        forward = [{"record_id_left": "a", "record_id_right": "b"}]
        reverse = [{"record_id_left": "b", "record_id_right": "a"}]
        text_records = ef.project_text_records(records)
        first = ef.extend_pair_features(pf.build_feature_frame(forward, records), forward, text_records, vectors)
        second = ef.extend_pair_features(pf.build_feature_frame(reverse, records), reverse, text_records, vectors)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first[ef.EMBEDDING_NAMES[0]].iloc[0], -1)
        self.assertEqual(first[ef.EMBEDDING_NAMES[1]].iloc[0], -1)
        self.assertEqual(tuple(first.columns), ef.FEATURE_NAMES)

    def test_only_observed_name_address_text_enters_encoder_pool(self):
        rows = {"opaque-id": {**normalized(company_name="alpha", address="street", registration_id="00123"),
                               "true_entity_id": "SECRET", "split_group_id": "FAMILY", "split": "test"}}
        projected = ef.project_text_records(rows)
        self.assertEqual(ef.unique_texts(projected), ["alpha", "street"])
        with self.assertRaisesRegex(ValueError, "Only observed names/addresses"):
            ef.unique_texts(rows)

    def test_experimental_schema_rejects_extra_metadata_and_out_of_range_cosines(self):
        frame, _ = extended_training()
        for invalid in (frame.assign(true_entity_id="SECRET"), frame.assign(company_name_embedding_cosine=1.01),
                        frame.assign(address_embedding_cosine=np.nan)):
            with self.assertRaises(ValueError):
                ef.validate_feature_frame(invalid)


class CacheChecks(unittest.TestCase):
    def test_unique_nonempty_texts_batching_and_warm_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = ef.EmbeddingCache(Path(folder), context())
            calls = []
            def fixture_encoder(texts, batch_size):
                calls.append((texts, batch_size))
                return np.array([[1., 0.] if text == "alpha" else [0., 1.] for text in texts])
            vectors, stats = cache.get_or_encode(["beta", "alpha", "beta", "", " "], fixture_encoder, batch_size=2)
            self.assertEqual(calls, [(["alpha", "beta"], 2)])
            self.assertEqual((stats["cache_hits"], stats["cache_misses"], stats["cache_state"]), (0, 2, "cold"))
            restored, warm = cache.get_or_encode(["alpha", "beta"], fixture_encoder)
            self.assertEqual(len(calls), 1)
            self.assertEqual((warm["cache_hits"], warm["cache_misses"], warm["embedding_encode_seconds"]), (2, 0, 0))
            for text in vectors:
                np.testing.assert_array_equal(vectors[text], restored[text])

    def test_text_model_revision_preprocessing_and_runtime_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            base = ef.EmbeddingCache(Path(folder), context())
            contexts = [replace(context(), model_revision="b" * 40),
                        replace(context(), preprocessing_version="unit-preprocessing-v2"),
                        replace(context(), runtime_version="different-runtime"),
                        replace(context(), model_id="another-model")]
            keys = [base.path_for("alpha"), base.path_for("beta")]
            keys += [ef.EmbeddingCache(Path(folder), item).path_for("alpha") for item in contexts]
            self.assertEqual(len(keys), len(set(keys)))
            base.get_or_encode(["alpha"], lambda texts, batch: np.array([[1., 0.]]))
            _, changed = ef.EmbeddingCache(Path(folder), contexts[0]).get_or_encode(
                ["alpha"], lambda texts, batch: np.array([[0., 1.]]))
            self.assertEqual(changed["cache_misses"], 1)

    def test_cache_identity_corruption_and_invalid_vectors_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = ef.EmbeddingCache(Path(folder), context())
            cache.get_or_encode(["alpha"], lambda texts, batch: np.array([[1., 0.]]))
            with cache.path_for("alpha").open("wb") as handle:
                np.savez_compressed(handle, vector=np.array([1., 0.]), identity=json.dumps(cache.identity("different")))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                cache.get_or_encode(["alpha"], lambda texts, batch: np.array([[1., 0.]]))
            with self.assertRaisesRegex(ValueError, "unit norm"):
                cache.get_or_encode(["zero-vector"], lambda texts, batch: np.array([[0., 0.]]))


class ExperimentalModelChecks(unittest.TestCase):
    def test_fit_is_training_only_with_requested_configuration(self):
        frame, labels = extended_training()
        model = experiment.fit_embedding_matcher(frame, labels)
        scaler = model.named_steps["scaler"]
        self.assertEqual(int(scaler.n_samples_seen_), 4)
        np.testing.assert_allclose(scaler.mean_, frame.mean().to_numpy())
        before = scaler.mean_.copy()
        experiment.predict_embedding(model, frame.iloc[:1])
        np.testing.assert_array_equal(before, scaler.mean_)
        self.assertEqual(model.named_steps["logistic_regression"].random_state, 42)
        self.assertEqual(model.named_steps["logistic_regression"].max_iter, 1000)

    def test_experimental_save_load_predictions_and_feature_schema(self):
        frame, labels = extended_training()
        model = experiment.fit_embedding_matcher(frame, labels)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            experiment.save_embedding_matcher(model, directory, context(), {"unit_fixture": True})
            restored, threshold = experiment.load_embedding_matcher(directory)
            scores, predictions = experiment.predict_embedding(model, frame)
            saved_scores, saved_predictions = experiment.predict_embedding(restored, frame, threshold)
            np.testing.assert_array_equal(scores, saved_scores)
            np.testing.assert_array_equal(predictions, saved_predictions)
            self.assertEqual(threshold, 0.5)
            self.assertEqual(tuple(json.loads((directory / "feature_names.json").read_text())), ef.FEATURE_NAMES)

    def test_validation_f1_selection_keeps_simpler_model_on_tie(self):
        self.assertEqual(experiment.select_next_model({"f1": 1.0}, {"f1": 1.0})[0], "stage3_logistic_regression")
        self.assertEqual(experiment.select_next_model({"f1": 1.0}, {"f1": 0.9})[0], "stage3_logistic_regression")
        self.assertEqual(experiment.select_next_model({"f1": 0.9}, {"f1": 1.0})[0], "embedding_experiment")


if __name__ == "__main__":
    unittest.main()
