"""
Pydantic models for the LaundryEnv OpenEnv interface.
Defines typed Observation, Action, and Reward structures.
"""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─── Item & Order Models ──────────────────────────────────────────────────────

class ClothingItem(BaseModel):
    item_id: str
    name: str
    category: Literal["shirt", "pants", "dress", "jacket", "other"]
    condition: Literal["new", "like_new", "good", "fair", "poor"]
    listing_type: Literal["sell", "rent", "buy_request"]
    price: float
    owner_id: str
    available: bool = True
    tags: List[str] = Field(default_factory=list)


class Order(BaseModel):
    order_id: str
    customer_id: str
    item_id: str
    order_type: Literal["purchase", "rental", "return", "complaint"]
    status: Literal["pending", "processing", "completed", "cancelled", "disputed"]
    created_at: str
    notes: Optional[str] = None
    priority: Literal["low", "medium", "high", "urgent"] = "medium"


class CustomerMessage(BaseModel):
    message_id: str
    customer_id: str
    subject: str
    body: str
    category: Optional[Literal["refund", "lost_item", "damage", "general", "dispute"]] = None
    resolved: bool = False
    sentiment: Optional[Literal["positive", "neutral", "negative", "angry"]] = None


# ─── Observation Model ────────────────────────────────────────────────────────

class MarketplaceState(BaseModel):
    total_listings: int
    active_rentals: int
    pending_orders_count: int
    revenue_today: float
    disputes_open: int
    satisfaction_score: float  # 0.0–1.0


class Observation(BaseModel):
    marketplace_state: MarketplaceState
    pending_orders: List[Order] = Field(default_factory=list)
    inventory: List[ClothingItem] = Field(default_factory=list)
    customer_messages: List[CustomerMessage] = Field(default_factory=list)
    current_task: str
    task_context: Dict[str, Any] = Field(default_factory=dict)
    step_number: int = 0
    max_steps: int = 20


# ─── Action Model ─────────────────────────────────────────────────────────────

class Action(BaseModel):
    action_type: Literal[
        # Order actions
        "route_order",
        "approve_order",
        "cancel_order",
        "escalate_order",
        # Inventory actions
        "restock_item",
        "update_price",
        "remove_listing",
        "add_listing",
        # Customer support actions
        "send_message",
        "issue_refund",
        "mark_resolved",
        "assign_agent",
        # General
        "no_op",
    ]
    payload: Dict[str, Any] = Field(default_factory=dict)


# ─── Reward Model ─────────────────────────────────────────────────────────────

class StepReward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    reason: str
    incremental: float = 0.0   # reward from this step alone
    cumulative: float = 0.0    # total so far


# ─── Step Return ─────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: StepReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
