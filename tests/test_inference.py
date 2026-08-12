import pandas as pd

from src.inference import predict_feature_frame, predict_feature_mapping


class CapturingModel:
    feature_names_ = [
        "payer",
        "primary_diagnosis_code",
        "payer_dx_interaction",
        "num_diagnoses",
    ]

    def predict_proba(self, features):
        self.features = features.copy()
        return {(0, 1): 0.75}


def test_mapping_and_frame_paths_share_feature_preparation():
    interactive_model = CapturingModel()
    frame_model = CapturingModel()
    raw_features = {
        "payer": "Medicare",
        "primary_diagnosis_code": "J18.9",
        "num_diagnoses": 2,
    }

    interactive_result = predict_feature_mapping(
        interactive_model,
        0.7,
        raw_features,
    )
    frame_result = predict_feature_frame(
        frame_model,
        0.7,
        pd.DataFrame(
            [
                {
                    **raw_features,
                    "payer_dx_interaction": "Medicare_J18.9",
                }
            ]
        ),
    )

    assert (
        interactive_result
        == frame_result
        == {
            "readmission_probability": 0.75,
            "prediction": 1,
            "threshold": 0.7,
        }
    )
    pd.testing.assert_frame_equal(
        interactive_model.features,
        frame_model.features,
    )
    assert (
        interactive_model.features.columns.tolist() == interactive_model.feature_names_
    )
    assert str(interactive_model.features["payer"].dtype) == "category"
    assert interactive_model.features.loc[0, "payer_dx_interaction"] == "Medicare_J18.9"
