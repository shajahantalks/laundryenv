# 🧺 WashWear LaundryEnv

> An OpenEnv-compliant AI agent training environment built around a real-world laundry marketplace — buy, sell, and rent clothing.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://openenv.dev)
[![HF Spaces](https://img.shields.io/badge/HuggingFace-Spaces-yellow)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

---

## Overview & Motivation

WashWear LaundryEnv simulates the operations of a sustainable clothing marketplace — the kind of platform where people can buy pre-loved garments, rent outfits for occasions, or sell items they no longer wear.

Real e-commerce platforms deal with complex workflows: order routing, inventory balancing, customer dispute resolution. These make excellent RL/LLM agent training grounds because they:

- Mirror genuine human work tasks (no toy problems)
- Have clear success criteria and intermediate feedback
- Scale in difficulty naturally
- Reward sensible prioritisation and domain reasoning

---

## Action & Observation Space

### Observation Space

Each step returns an `Observation` object (typed via Pydantic):

| Field | Type | Description |
|---|---|---|
| `marketplace_state` | `MarketplaceState` | High-level metrics (revenue, disputes, listings count) |
| `pending_orders` | `List[Order]` | Orders awaiting action |
| `inventory` | `List[ClothingItem]` | Current item listings |
| `customer_messages` | `List[CustomerMessage]` | Support tickets |
| `current_task` | `str` | Active task ID |
| `task_context` | `dict` | Task-specific hints and instructions |
| `step_number` | `int` | Current step |
| `max_steps` | `int` | Episode limit (20) |

### Action Space

Each action is an `Action` object:

```python
Action(
    action_type: str,   # one of the types below
    payload: dict       # action-specific arguments
)
```

| `action_type` | Payload keys | Description |
|---|---|---|
| `route_order` | `order_id`, `department` | Route order to sales/support/rentals/returns |
| `approve_order` | `order_id` | Approve a pending order |
| `cancel_order` | `order_id` | Cancel an order |
| `escalate_order` | `order_id` | Escalate to senior team |
| `restock_item` | `item_id`, `quantity` | Restock inventory |
| `update_price` | `item_id`, `new_price` | Change listing price |
| `remove_listing` | `item_id` | Remove an item listing |
| `add_listing` | `item_id`, `name`, `price`, … | Add new listing |
| `send_message` | `customer_id`, `message` | Send customer a message |
| `issue_refund` | `customer_id`, `message_id`, `amount` | Issue a refund |
| `mark_resolved` | `message_id` | Close a support ticket |
| `no_op` | — | Do nothing (penalised) |

---

## Tasks

### Task 1 — Order Triage (Easy)

**Objective:** Route 5 incoming orders to the correct department.

**Rules:** `purchase → sales`, `complaint → support`, `rental → rentals`, `return → returns`

**Bonus:** Handling the `urgent` complaint order in the first two steps awards +0.1.

**Grader:** `score = (correctly_routed / total) + urgency_bonus`

**Expected difficulty:** Easy — a rule-based agent scores 1.0.

**Baseline score:** ~0.75 (LLM with no hints)

---

### Task 2 — Inventory Management (Medium)

**Objective:** Within a ₹500 budget, normalise stock levels and fix prices for 5 items.

**Criteria:**
- Stock normalisation (40%): bring stock close to `ideal_stock`
- Price optimisation (30%): match prices to `market_price`
- Budget discipline (20%): don't exceed ₹500
- Overstock handling (10%): deal with ITM003 (12 items, ideal: 4)

**Grader:** weighted sum of four sub-scores.

**Expected difficulty:** Medium — requires multi-step planning.

**Baseline score:** ~0.55

---

### Task 3 — Customer Support Resolution (Hard)

**Objective:** Fully resolve 3 complex customer support tickets.

**Tickets:**
1. Lost rental jacket → full refund (₹45) + resolve
2. Unprocessed return → cancel order + refund (₹12) + resolve
3. Damaged item on arrival → refund (₹30) + send message + resolve

**Grader:** per-ticket score averaging required actions, correct refund amounts, communication, and resolution status.

**Expected difficulty:** Hard — multi-action chains, correct amounts, empathetic communication.

**Baseline score:** ~0.40

---

## Setup & Usage

### Local Development

```bash
# 1. Clone this repo
git clone https://huggingface.co/spaces/your-username/laundry-env
cd laundry-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the web server
python app.py
# → Open http://localhost:7860
```

### Run the OpenEnv Interface Directly

```python
from env.laundry_env import LaundryEnv
from env.models import Action

# Pick a task
env = LaundryEnv(task_id="task_easy")  # or task_medium, task_hard
obs = env.reset()

print(obs.task_context["instruction"])

# Take a step
action = Action(
    action_type="route_order",
    payload={"order_id": "ORD001", "department": "sales"}
)
obs, reward, done, info = env.step(action)
print(f"Reward: {reward.value}  |  {reward.reason}")

# Get final grade when done
if done:
    print(env.final_grade())
```

### Run Baseline Inference (Hugging Face API)

```bash
# Get your free token at https://huggingface.co/settings/tokens
export HF_TOKEN=hf_your_token_here

python inference.py
# Evaluates all 3 tasks and saves baseline_results.json
```

### Docker

```bash
# Build
docker build -t laundryenv .

# Run
docker run -p 7860:7860 -e HF_TOKEN=hf_xxx laundryenv

# Open http://localhost:7860
```

---

## Reward Function

Rewards are designed to provide feedback throughout the trajectory:

| Behaviour | Reward |
|---|---|
| Correct order route | +0.10 |
| Correct urgent route (first 2 steps) | +0.15 |
| Wrong route | −0.05 |
| Restock within budget | +0.05 |
| Restock over budget | −0.05 |
| Correct refund issued | +0.10 |
| Support message sent | +0.05 |
| Ticket resolved | +0.10 |
| `no_op` action | −0.05 |

The `StepReward` model tracks both `incremental` (this step) and `cumulative` rewards.

---

## Baseline Performance

Evaluated using `mistralai/Mistral-7B-Instruct-v0.3` via Hugging Face Inference API:

| Task | Difficulty | Baseline Score |
|---|---|---|
| task_easy | Easy | 0.75 |
| task_medium | Medium | 0.55 |
| task_hard | Hard | 0.40 |
| **Average** | | **0.57** |

A perfect rule-based agent scores 1.0 on task_easy and ~0.85 on task_medium.

---

## File Structure

```
laundryenv/
├── app.py                    # FastAPI web server
├── inference.py              # Baseline eval script
├── openenv.yaml              # OpenEnv metadata
├── requirements.txt
├── Dockerfile
├── README.md
├── env/
│   ├── __init__.py
│   ├── laundry_env.py        # Main LaundryEnv class
│   └── models.py             # Pydantic models
├── tasks/
│   ├── __init__.py
│   └── task_definitions.py   # Tasks + graders
└── templates/
    └── index.html            # Marketplace website
```

---

## Deploying to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space
2. Choose **Docker** as the SDK
3. Push this repo to the Space:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/laundry-env
   git push hf main
   ```
4. Add `HF_TOKEN` as a Space secret in Settings → Variables and secrets
5. The space will build and be live at `https://YOUR_USERNAME-laundry-env.hf.space`

---

## License

MIT — built for the Meta OpenEnv Hackathon 🧺
