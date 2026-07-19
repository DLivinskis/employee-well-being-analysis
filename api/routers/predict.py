"""
`POST /predict`.
"""

from fastapi import APIRouter, Depends

from api.endpoint_helpers.dependencies import get_inference_service
from api.endpoint_helpers.inference import InferenceService
from api.endpoint_helpers.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    inference: InferenceService = Depends(get_inference_service),
) -> PredictResponse:
    """Predict satisfaction class + probabilities for one feature set."""
    features = request.model_dump(exclude={"model_name"})
    result = inference.predict(request.model_name, features)
    return PredictResponse(
        model_used=request.model_name,
        label=result["label"],
        class_probabilities=result["class_probabilities"],
    )
