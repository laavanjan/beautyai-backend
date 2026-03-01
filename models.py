from pydantic import BaseModel, Field, model_validator
from typing import Literal, Dict, List, Any, Optional, Union

Intent = Literal["skincare_input", "question", "out_of_scope", "greeting", "restart", "mixed"]

class PendingConfirmation(BaseModel):
    field: str
    candidates: List[str]
    original_message: str
    resolved_value: Optional[str] = None

class ConversationTurn(BaseModel):
    """One exchange stored for LLM context window."""
    user: str
    assistant: str

class DialogState(BaseModel):
    session_id: str
    current_node: str = "conversation"
    slots: Dict[str, Any] = {}
    asked_fields: List[str] = []
    pending_confirmation: Optional[PendingConfirmation] = None
    products: List[Dict] = []
    trigger_recommender: bool = False
    recommender_context: Optional[Dict] = None
    updated_at: Optional[str] = None
    conversation_history: List[ConversationTurn] = Field(default_factory=list)

class ChatResponse(BaseModel):
    reply_text: str
    suggested_options: List[Dict] = []
    current_node: str
    products: List[Dict] = []
    trigger_recommender: bool = False
    recommender_context: Optional[Dict] = None

class ExtractedSlots(BaseModel):
    main_category: Optional[Literal["Face", "Hair", "Body", "Baby"]] = None
    skin_type: Optional[Literal["oily", "dry", "combination", "normal", "sensitive"]] = None
    skin_concern: Optional[Union[str, List[str]]] = Field(None, description="Skin concerns")
    hair_type: Optional[Literal["straight", "wavy", "curly", "coily"]] = None
    hair_concern: Optional[Union[str, List[str]]] = None
    age_range: Optional[Literal["teen", "20s", "30s", "40+", "50+"]] = None
    sensitivity: Optional[Literal["very sensitive", "sensitive", "normal"]] = None
    goal: Optional[str] = None
    baby_section: Optional[Literal["Baby Bath & Shampoo", "Baby Lotions & creams", "Baby Milk Powder"]] = Field(
        None, description="Specific section within Baby Products"
    )

    @model_validator(mode='after')
    def normalize_concerns(self):
        if isinstance(self.skin_concern, str):
            self.skin_concern = [self.skin_concern.strip()] if self.skin_concern else None
        if isinstance(self.hair_concern, str):
            self.hair_concern = [self.hair_concern.strip()] if self.hair_concern else None
        return self