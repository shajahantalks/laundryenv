"""
Baseline inference script for LaundryEnv.
Uses OpenAI API client pointed at Hugging Face Inference API.
Reads HF_TOKEN from environment variables.

Usage:
    export HF_TOKEN=your_token_here
    python inference.py
"""
import json
import os
import sys

# OpenAI SDK works with HF inference endpoints
try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))

from env.laundry_env import LaundryEnv
from env.models import Action

# ─── Config ───────────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set.")
    print("Get your token from https://huggingface.co/settings/tokens")
    sys.exit(1)

# Hugging Face Inference API – OpenAI-compatible endpoint
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # free-tier model on HF
client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=HF_TOKEN,
)

TASKS = ["task_easy", "task_medium", "task_hard"]


# ─── Agent Logic ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI agent managing a laundry marketplace platform.
Your job is to complete tasks by taking actions in JSON format.

Each turn you receive an observation (the current state) and must respond with a
single JSON action object like:
{
  "action_type": "<action_type>",
  "payload": { ... }
}

Available action_types:
- route_order: {"order_id": "...", "department": "sales|support|rentals|returns"}
- approve_order: {"order_id": "..."}
- cancel_order: {"order_id": "..."}
- restock_item: {"item_id": "...", "quantity": N}
- update_price: {"item_id": "...", "new_price": N.N}
- remove_listing: {"item_id": "..."}
- send_message: {"customer_id": "...", "message": "..."}
- issue_refund: {"customer_id": "...", "message_id": "...", "amount": N.N}
- mark_resolved: {"message_id": "..."}
- no_op: {}

Respond ONLY with the JSON object. No explanation, no markdown, just the raw JSON.
"""


def obs_to_prompt(obs_dict: dict) -> str:
    """Convert observation to a readable prompt string."""
    ctx = obs_dict.get("task_context", {})
    return (
        f"TASK: {obs_dict['current_task']}\n"
        f"STEP: {obs_dict['step_number']}/{obs_dict['max_steps']}\n"
        f"INSTRUCTION: {ctx.get('instruction', '')}\n\n"
        f"STATE:\n{json.dumps(obs_dict, indent=2, default=str)[:3000]}\n\n"
        "What is your next action? Respond with a single JSON action object."
    )


def ask_agent(conversation: list) -> dict:
    """Call the LLM and parse its JSON action response."""
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=conversation,
        max_tokens=256,
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        print(f"  [WARN] Could not parse response: {raw[:100]}")
        return {"action_type": "no_op", "payload": {}}


# ─── Main Evaluation Loop ─────────────────────────────────────────────────────

def run_task(task_id: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Running task: {task_id.upper()}")
    print(f"{'='*60}")

    env = LaundryEnv(task_id=task_id)
    obs = env.reset()

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    step_rewards = []

    while True:
        obs_dict = obs.dict()
        user_msg = obs_to_prompt(obs_dict)
        conversation.append({"role": "user", "content": user_msg})

        action_dict = ask_agent(conversation)
        print(f"  Step {obs.step_number+1}: {action_dict.get('action_type')} {action_dict.get('payload', {})}")

        # Record assistant response
        conversation.append({"role": "assistant", "content": json.dumps(action_dict)})

        try:
            action = Action(**action_dict)
        except Exception as e:
            print(f"  [ERROR] Invalid action: {e}")
            action = Action(action_type="no_op")

        obs, reward, done, info = env.step(action)
        step_rewards.append(reward.value)
        print(f"  → Reward: {reward.value:.3f} | {reward.reason[:60]}")

        if done:
            break

    final = env.final_grade()
    print(f"\n  FINAL SCORE: {final['final_score']:.3f}")
    print(f"  Feedback: {final['feedback']}")
    return final


def main():
    print("\n🧺 LaundryEnv Baseline Evaluation")
    print(f"  Model: {HF_MODEL}")
    print(f"  Tasks: {TASKS}")

    results = {}
    for task_id in TASKS:
        try:
            results[task_id] = run_task(task_id)
        except Exception as e:
            print(f"  [ERROR] Task {task_id} failed: {e}")
            results[task_id] = {"final_score": 0.0, "error": str(e)}

    print(f"\n{'='*60}")
    print("  BASELINE SUMMARY")
    print(f"{'='*60}")
    total = 0.0
    for task_id, result in results.items():
        score = result.get("final_score", 0.0)
        total += score
        print(f"  {task_id:20s}: {score:.3f}")
    avg = total / len(TASKS)
    print(f"  {'AVERAGE':20s}: {avg:.3f}")
    print(f"{'='*60}\n")

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump({"model": HF_MODEL, "results": results, "average": avg}, f, indent=2)
    print("  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
