"""
LaundryEnv – OpenEnv-compliant environment for laundry marketplace simulation.

Implements:
  reset() -> Observation
  step(action) -> (Observation, reward, done, info)
  state() -> dict
"""
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from env.models import (
    Action, ClothingItem, CustomerMessage, MarketplaceState,
    Observation, Order, StepResult, StepReward,
)
from tasks.task_definitions import (
    EASY_ORDERS, INVENTORY_SCENARIO, SUPPORT_SCENARIO,
    grade_easy_task, grade_medium_task, grade_hard_task,
)


SAMPLE_INVENTORY = [
    ClothingItem(item_id="ITM001", name="Blue Jeans",      category="pants",  condition="good",     listing_type="sell", price=25.0, owner_id="SHOP_1"),
    ClothingItem(item_id="ITM002", name="White Shirt",     category="shirt",  condition="like_new", listing_type="sell", price=40.0, owner_id="SHOP_1"),
    ClothingItem(item_id="ITM003", name="Summer Dress",    category="dress",  condition="new",      listing_type="rent", price=20.0, owner_id="SHOP_2"),
    ClothingItem(item_id="ITM004", name="Winter Jacket",   category="jacket", condition="good",     listing_type="sell", price=80.0, owner_id="SHOP_1"),
    ClothingItem(item_id="ITM005", name="Sports Leggings", category="pants",  condition="new",      listing_type="sell", price=15.0, owner_id="SHOP_3"),
]

TASK_IDS = ["task_easy", "task_medium", "task_hard"]


class LaundryEnv:
    """
    OpenEnv-compliant laundry marketplace environment.
    """

    def __init__(self, task_id: str = "task_easy"):
        if task_id not in TASK_IDS:
            raise ValueError(f"task_id must be one of {TASK_IDS}")
        self.task_id = task_id
        self._step_count = 0
        self._max_steps = 20
        self._actions_log: list = []
        self._cumulative_reward = 0.0
        self._done = False
        self._state: Dict[str, Any] = {}
        self.reset()

    # ── OpenEnv Interface ────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._step_count = 0
        self._actions_log = []
        self._cumulative_reward = 0.0
        self._done = False
        self._state = self._build_initial_state()
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, StepReward, bool, Dict[str, Any]]:
        """Apply action, return (observation, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        self._actions_log.append(action.dict())

        # Apply action to state
        action_reward, feedback = self._apply_action(action)

        # Check termination
        done = self._check_done()

        # Compute full reward
        partial_score = self._compute_partial_score()
        incremental = action_reward
        self._cumulative_reward = min(1.0, self._cumulative_reward + incremental)

        reward = StepReward(
            value=round(partial_score, 3),
            reason=feedback,
            incremental=round(incremental, 3),
            cumulative=round(self._cumulative_reward, 3),
        )

        # Penalise no-ops (lazy agent)
        if action.action_type == "no_op":
            reward.value = max(0.0, reward.value - 0.05)
            reward.reason = "No-op penalised: -0.05"

        observation = self._build_observation()
        info = {
            "step": self._step_count,
            "actions_taken": len(self._actions_log),
            "task_id": self.task_id,
        }

        self._done = done
        return observation, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return the full current state dict."""
        return {
            "task_id": self.task_id,
            "step": self._step_count,
            "done": self._done,
            "cumulative_reward": self._cumulative_reward,
            "actions_log": self._actions_log,
            **self._state,
        }

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _build_initial_state(self) -> Dict[str, Any]:
        if self.task_id == "task_easy":
            return {
                "pending_orders": [
                    Order(
                        order_id=o["order_id"],
                        customer_id=f"CUST_{i}",
                        item_id=f"ITM00{i+1}",
                        order_type=o["order_type"],
                        status="pending",
                        created_at=datetime.now().isoformat(),
                        priority=o["priority"],
                    )
                    for i, o in enumerate(EASY_ORDERS)
                ],
                "inventory": list(SAMPLE_INVENTORY),
                "customer_messages": [],
                "routed_orders": {},
            }
        elif self.task_id == "task_medium":
            return {
                "pending_orders": [],
                "inventory": list(SAMPLE_INVENTORY),
                "customer_messages": [],
                "budget_remaining": INVENTORY_SCENARIO["budget"],
                "restock_actions": [],
                "price_updates": {},
                "removed_items": [],
            }
        else:  # task_hard
            return {
                "pending_orders": [],
                "inventory": list(SAMPLE_INVENTORY),
                "customer_messages": [
                    CustomerMessage(
                        message_id=t["message_id"],
                        customer_id=t["customer_id"],
                        subject=t["subject"],
                        body=t["body"],
                        category=t["category"],
                        sentiment=t["sentiment"],
                    )
                    for t in SUPPORT_SCENARIO["tickets"]
                ],
                "resolved_tickets": [],
            }

    def _build_observation(self) -> Observation:
        mstate = MarketplaceState(
            total_listings=len(self._state.get("inventory", [])),
            active_rentals=2,
            pending_orders_count=len(self._state.get("pending_orders", [])),
            revenue_today=320.50,
            disputes_open=len([m for m in self._state.get("customer_messages", []) if not m.resolved]),
            satisfaction_score=0.72,
        )
        task_ctx: Dict[str, Any] = {}
        if self.task_id == "task_easy":
            task_ctx = {
                "instruction": "Route each pending order to the correct department: sales, support, rentals, or returns.",
                "routing_map_hint": "purchase→sales, complaint→support, rental→rentals, return→returns",
                "tip": "Handle urgent priority orders first for a bonus.",
            }
        elif self.task_id == "task_medium":
            task_ctx = {
                "instruction": "Optimise inventory: restock out-of-stock items, fix over/under-priced listings, remove excess stock.",
                "budget": self._state.get("budget_remaining", 500),
                "market_prices_hint": "Check market_price vs current_price for each item.",
            }
        else:
            task_ctx = {
                "instruction": "Resolve all customer support tickets: issue refunds, send messages, cancel disputes, mark resolved.",
                "sla_note": "Angry/urgent customers should be handled first.",
            }

        return Observation(
            marketplace_state=mstate,
            pending_orders=self._state.get("pending_orders", []),
            inventory=self._state.get("inventory", []),
            customer_messages=self._state.get("customer_messages", []),
            current_task=self.task_id,
            task_context=task_ctx,
            step_number=self._step_count,
            max_steps=self._max_steps,
        )

    def _apply_action(self, action: Action) -> Tuple[float, str]:
        """Apply action to state, return (incremental_reward, feedback_string)."""
        p = action.payload
        atype = action.action_type

        if self.task_id == "task_easy":
            if atype == "route_order":
                oid = p.get("order_id")
                dept = p.get("department")
                self._state["routed_orders"][oid] = dept
                # Find correct dept
                for o in EASY_ORDERS:
                    if o["order_id"] == oid:
                        correct = o["correct_dept"]
                        if dept == correct:
                            reward = 0.15 if o["priority"] == "urgent" else 0.1
                            return reward, f"Correctly routed {oid} to {dept}."
                        else:
                            return -0.05, f"Wrong routing for {oid}: got {dept}, expected {correct}."
                return 0.0, f"Unknown order {oid}."

        elif self.task_id == "task_medium":
            if atype == "restock_item":
                qty = p.get("quantity", 0)
                cost = qty * INVENTORY_SCENARIO["restock_cost_per_unit"]
                if cost > self._state["budget_remaining"]:
                    return -0.05, "Over budget! Restock rejected."
                self._state["budget_remaining"] -= cost
                self._state["restock_actions"].append(p)
                return 0.05, f"Restocked {p.get('item_id')} ×{qty} (cost ${cost:.0f})."
            elif atype == "update_price":
                self._state["price_updates"][p.get("item_id")] = p.get("new_price")
                return 0.05, f"Price updated for {p.get('item_id')} → ${p.get('new_price')}."
            elif atype == "remove_listing":
                self._state["removed_items"].append(p.get("item_id"))
                return 0.05, f"Removed listing {p.get('item_id')}."

        elif self.task_id == "task_hard":
            if atype == "issue_refund":
                return 0.1, f"Refund of ${p.get('amount', 0):.0f} issued to {p.get('customer_id')}."
            elif atype == "send_message":
                return 0.05, f"Message sent to customer {p.get('customer_id')}."
            elif atype == "mark_resolved":
                mid = p.get("message_id")
                for msg in self._state["customer_messages"]:
                    if msg.message_id == mid:
                        msg.resolved = True
                self._state["resolved_tickets"].append(mid)
                return 0.1, f"Ticket {mid} marked resolved."
            elif atype == "cancel_order":
                return 0.05, f"Order {p.get('order_id')} cancelled."

        return 0.0, f"Action {atype} applied."

    def _compute_partial_score(self) -> float:
        """Compute current progress score without finalising."""
        if self.task_id == "task_easy":
            result = grade_easy_task(self._actions_log)
        elif self.task_id == "task_medium":
            result = grade_medium_task(self._actions_log, self._state)
        else:
            result = grade_hard_task(self._actions_log)
        return result.score

    def _check_done(self) -> bool:
        if self._step_count >= self._max_steps:
            return True
        if self.task_id == "task_easy":
            routed = self._state.get("routed_orders", {})
            return len(routed) >= len(EASY_ORDERS)
        elif self.task_id == "task_medium":
            actions = [a["action_type"] for a in self._actions_log]
            return len(self._actions_log) >= 10 and "update_price" in actions
        elif self.task_id == "task_hard":
            resolved = self._state.get("resolved_tickets", [])
            return len(resolved) >= len(SUPPORT_SCENARIO["tickets"])
        return False

    def final_grade(self) -> Dict[str, Any]:
        """Return final graded result for the completed episode."""
        if self.task_id == "task_easy":
            result = grade_easy_task(self._actions_log)
        elif self.task_id == "task_medium":
            result = grade_medium_task(self._actions_log, self._state)
        else:
            result = grade_hard_task(self._actions_log)
        return {
            "task_id": self.task_id,
            "final_score": result.score,
            "breakdown": result.breakdown,
            "feedback": result.feedback,
            "steps_taken": self._step_count,
        }
