import torch
import numpy as np


class UncertaintyEstimator:

    def __init__(
        self,
        model,
        device="cpu",
        num_samples=50
    ):

        self.model = model
        self.device = device
        self.num_samples = num_samples

    def predict_with_uncertainty(
        self,
        graph
    ):

        self.model.train()

        predictions = []

        with torch.no_grad():

            for _ in range(
                self.num_samples
            ):

                output = self.model(
                    graph.x.to(self.device),
                    graph.edge_index.to(self.device),
                    graph.batch.to(self.device)
                )

                prob = torch.softmax(
                    output,
                    dim=1
                )

                predictions.append(
                    prob.cpu().numpy()
                )

        predictions = np.array(
            predictions
        )

        mean_prediction = np.mean(
            predictions,
            axis=0
        )

        uncertainty = np.std(
            predictions,
            axis=0
        )

        predicted_class = np.argmax(
            mean_prediction
        )

        confidence = np.max(
            mean_prediction
        )

        uncertainty_score = np.mean(
            uncertainty
        )

        return {
            "predicted_class": int(
                predicted_class
            ),
            "confidence": float(
                confidence
            ),
            "uncertainty": float(
                uncertainty_score
            )
        }