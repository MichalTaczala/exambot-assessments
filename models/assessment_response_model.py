from pydantic import BaseModel, Field


class AssessmentResponseModel(BaseModel):
    score: int = Field(description="The score for the assessment", ge=0, le=10)
    feedback: str = Field(
        description="The feedback for the assessment. Up to 3 sentences explaining the score. Focus on the aspects which caused the student lose points. Address the feedback directly to the student."
    )
