# Demo variant iter=0 produced by scripts/demo_flow.py ScriptedPlanner
# In production this file would be trainer source authored by the planner LLM.
def train():
    # backend=dry ignores this body and fabricates a synthetic log
    return {'iter': 0, 'source': 'demo_flow.ScriptedPlanner'}
