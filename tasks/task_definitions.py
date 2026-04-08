"""
Task definitions for LaundryEnv.
Three tasks: Easy (Order Triage), Medium (Inventory Mgmt), Hard (Customer Support).
Each task has an objective, initial state generator, and a grader returning 0.0–1.0.
"""
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class TaskResult:
    score: float          # 0.0 – 1.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""


# ═══════════════════════════════════════════════════════════
# TASK 1 – EASY: Order Triage
# ═══════════════════════════════════════════════════════════

EASY_ORDERS = [
    {"order_id": "ORD001", "order_type": "purchase",  "priority": "low",    "correct_dept": "sales"},
    {"order_id": "ORD002", "order_type": "complaint", "priority": "urgent", "correct_dept": "support"},
    {"order_id": "ORD003", "order_type": "rental",    "priority": "medium", "correct_dept": "rentals"},
    {"order_id": "ORD004", "order_type": "return",    "priority": "high",   "correct_dept": "returns"},
    {"order_id": "ORD005", "order_type": "complaint", "priority": "high",   "correct_dept": "support"},
]

ROUTING_MAP = {
    "purchase":  "sales",
    "complaint": "support",
    "rental":    "rentals",
    "return":    "returns",
}


def grade_easy_task(agent_actions: List[Dict[str, Any]]) -> TaskResult:
    """
    Grade order triage task.
    Score = (correctly routed orders) / total_orders
    Bonus for handling urgent orders first.
    """
    routed = {}
    urgent_handled_early = False
    urgent_position = None

    for i, action in enumerate(agent_actions):
        if action.get("action_type") == "route_order":
            order_id = action["payload"].get("order_id")
            dept = action["payload"].get("department")
            routed[order_id] = dept
            if order_id == "ORD002" and i <= 1:
                urgent_handled_early = True
                urgent_position = i

    correct = sum(
        1 for o in EASY_ORDERS
        if routed.get(o["order_id"]) == o["correct_dept"]
    )
    accuracy = correct / len(EASY_ORDERS)
    urgency_bonus = 0.1 if urgent_handled_early else 0.0
    score = min(1.0, accuracy + urgency_bonus)

    return TaskResult(
        score=round(score, 3),
        breakdown={
            "routing_accuracy": round(accuracy, 3),
            "urgency_bonus": urgency_bonus,
            "orders_routed": len(routed),
            "correct_routes": correct,
        },
        feedback=(
            f"Routed {correct}/{len(EASY_ORDERS)} orders correctly. "
            + ("Urgent order prioritised correctly! +0.1 bonus." if urgent_handled_early else "Tip: handle urgent orders first.")
        ),
    )


# ═══════════════════════════════════════════════════════════
# TASK 2 – MEDIUM: Inventory Management
# ═══════════════════════════════════════════════════════════

INVENTORY_SCENARIO = {
    "items": [
        {"item_id": "ITM001", "name": "Blue Jeans",      "stock": 0,  "ideal_stock": 5,  "current_price": 25.0, "market_price": 30.0},
        {"item_id": "ITM002", "name": "White Shirt",     "stock": 1,  "ideal_stock": 8,  "current_price": 40.0, "market_price": 35.0},
        {"item_id": "ITM003", "name": "Summer Dress",    "stock": 12, "ideal_stock": 4,  "current_price": 20.0, "market_price": 22.0},
        {"item_id": "ITM004", "name": "Winter Jacket",   "stock": 0,  "ideal_stock": 3,  "current_price": 80.0, "market_price": 90.0},
        {"item_id": "ITM005", "name": "Sports Leggings", "stock": 3,  "ideal_stock": 6,  "current_price": 15.0, "market_price": 18.0},
    ],
    "budget": 500.0,
    "restock_cost_per_unit": 10.0,
}


def grade_medium_task(agent_actions: List[Dict[str, Any]], final_state: Dict[str, Any]) -> TaskResult:
    """
    Grade inventory management.
    Criteria:
    - Stock normalisation (how close to ideal levels): 40%
    - Price optimisation (prices close to market): 30%
    - Budget discipline (didn't overspend): 20%
    - Removed overstocked items (summer dress): 10%
    """
    items_map = {i["item_id"]: dict(i) for i in INVENTORY_SCENARIO["items"]}

    # Parse actions
    restocked = {}
    repriced = {}
    removed = set()
    spent = 0.0

    for action in agent_actions:
        atype = action.get("action_type")
        p = action.get("payload", {})
        if atype == "restock_item":
            item_id = p.get("item_id")
            qty = p.get("quantity", 0)
            restocked[item_id] = restocked.get(item_id, 0) + qty
            spent += qty * INVENTORY_SCENARIO["restock_cost_per_unit"]
        elif atype == "update_price":
            repriced[p.get("item_id")] = p.get("new_price", 0)
        elif atype == "remove_listing":
            removed.add(p.get("item_id"))

    # 1. Stock score
    stock_scores = []
    for item in INVENTORY_SCENARIO["items"]:
        iid = item["item_id"]
        new_stock = item["stock"] + restocked.get(iid, 0)
        if iid in removed:
            new_stock = 0
        diff = abs(new_stock - item["ideal_stock"])
        stock_scores.append(max(0, 1 - diff / max(item["ideal_stock"], 1)))
    stock_score = sum(stock_scores) / len(stock_scores)

    # 2. Price score
    price_scores = []
    for item in INVENTORY_SCENARIO["items"]:
        iid = item["item_id"]
        new_price = repriced.get(iid, item["current_price"])
        market = item["market_price"]
        deviation = abs(new_price - market) / market
        price_scores.append(max(0, 1 - deviation))
    price_score = sum(price_scores) / len(price_scores)

    # 3. Budget score
    budget_score = 1.0 if spent <= INVENTORY_SCENARIO["budget"] else max(0, 1 - (spent - INVENTORY_SCENARIO["budget"]) / INVENTORY_SCENARIO["budget"])

    # 4. Overstock removal (ITM003 had 12, ideal 4)
    overstock_score = 1.0 if "ITM003" in removed or restocked.get("ITM003", 0) == 0 else 0.0

    score = (stock_score * 0.4 + price_score * 0.3 + budget_score * 0.2 + overstock_score * 0.1)

    return TaskResult(
        score=round(score, 3),
        breakdown={
            "stock_normalisation": round(stock_score, 3),
            "price_optimisation": round(price_score, 3),
            "budget_discipline": round(budget_score, 3),
            "overstock_handling": overstock_score,
            "budget_spent": round(spent, 2),
        },
        feedback=(
            f"Stock: {stock_score:.0%}, Prices: {price_score:.0%}, "
            f"Budget: {budget_score:.0%}, Overstock: {'✓' if overstock_score else '✗'}. "
            f"Spent ${spent:.0f} of ${INVENTORY_SCENARIO['budget']:.0f}."
        ),
    )


# ═══════════════════════════════════════════════════════════
# TASK 3 – HARD: Customer Support Resolution
# ═══════════════════════════════════════════════════════════

SUPPORT_SCENARIO = {
    "tickets": [
        {
            "message_id": "MSG001",
            "customer_id": "CUST_A",
            "subject": "Never received my rental jacket",
            "body": "I paid for a jacket rental 2 weeks ago and never got it. I want a full refund immediately.",
            "category": "lost_item",
            "sentiment": "angry",
            "required_actions": ["issue_refund", "mark_resolved"],
            "refund_amount": 45.0,
            "sla_hours": 4,
        },
        {
            "message_id": "MSG002",
            "customer_id": "CUST_B",
            "subject": "Returned item not processed",
            "body": "I returned my dress 5 days ago but still see it as 'active rental' and am being charged.",
            "category": "dispute",
            "sentiment": "negative",
            "required_actions": ["cancel_order", "issue_refund", "mark_resolved"],
            "refund_amount": 12.0,
            "sla_hours": 8,
        },
        {
            "message_id": "MSG003",
            "customer_id": "CUST_C",
            "subject": "Item arrived damaged",
            "body": "The shirt I bought arrived with a large stain. This is unacceptable for the price I paid.",
            "category": "damage",
            "sentiment": "negative",
            "required_actions": ["issue_refund", "send_message", "mark_resolved"],
            "refund_amount": 30.0,
            "sla_hours": 12,
        },
    ]
}


def grade_hard_task(agent_actions: List[Dict[str, Any]]) -> TaskResult:
    """
    Grade customer support resolution.
    Criteria per ticket:
    - All required actions performed: 50%
    - Correct refund amounts issued: 30%
    - Response sent to customer: 10%
    - Ticket marked resolved: 10%
    """
    ticket_scores = []

    for ticket in SUPPORT_SCENARIO["tickets"]:
        mid = ticket["message_id"]
        relevant = [a for a in agent_actions if a.get("payload", {}).get("message_id") == mid
                    or a.get("payload", {}).get("customer_id") == ticket["customer_id"]]

        action_types_done = {a["action_type"] for a in relevant}

        # Required actions
        required = set(ticket["required_actions"])
        actions_done = len(required & action_types_done) / len(required)

        # Refund amount check
        refund_correct = 0.0
        for a in relevant:
            if a["action_type"] == "issue_refund":
                amt = a.get("payload", {}).get("amount", 0)
                if abs(amt - ticket["refund_amount"]) <= ticket["refund_amount"] * 0.1:
                    refund_correct = 1.0

        # Message sent
        msg_sent = 1.0 if "send_message" in action_types_done else 0.0

        # Resolved
        resolved = 1.0 if "mark_resolved" in action_types_done else 0.0

        t_score = actions_done * 0.5 + refund_correct * 0.3 + msg_sent * 0.1 + resolved * 0.1
        ticket_scores.append(t_score)

    score = sum(ticket_scores) / len(ticket_scores)

    return TaskResult(
        score=round(score, 3),
        breakdown={
            f"ticket_{i+1}_score": round(s, 3)
            for i, s in enumerate(ticket_scores)
        },
        feedback=(
            f"Resolved {sum(1 for s in ticket_scores if s >= 0.8)}/{len(ticket_scores)} tickets fully. "
            f"Average score: {score:.0%}. "
            "Ensure correct refund amounts and always send a response message."
        ),
    )
