"""
LLM Dependency Annotation Prompt Templates

"""

# ============================================================================
# PROMPT TEMPLATE 1: Dependency Edge Annotation
# ============================================================================

DEPENDENCY_ANNOTATION_PROMPT = """
You are an expert in task planning and workflow analysis. Your task is to identify the **logical dependencies** between steps in a hierarchical plan and convert it into a Directed Acyclic Graph (DAG).

## INPUT:
**User Query**: {query}

**Hierarchical Plan** (steps are numbered like 1, 1.1, 1.2, 2, 2.1, etc.):
{hierarchical_plan}

## YOUR TASK:
1. Analyze the logical dependencies between steps
2. Identify which steps MUST be completed before others can start
3. **Assume unlimited parallel resources** - only mark dependencies that are logically necessary
4. Ignore the hierarchical numbering - focus on actual dependencies
5. **Maximize parallelization** where possible

## DEPENDENCY RULES:
- **Sequential dependency**: Step A must complete before Step B can start
- **Data dependency**: Step B requires output/results from Step A
- **Logical dependency**: Step B logically depends on Step A being done first

## OUTPUT FORMAT:
Provide your answer as a JSON object with the following structure:

```json
{{
  "dependencies": [
    {{"from": "step_id", "to": "step_id", "reason": "brief explanation"}},
    ...
  ],
  "parallel_groups": [
    ["step_id_1", "step_id_2", ...],
    ...
  ]
}}
```

- `dependencies`: List of edges where "from" must complete before "to" can start
- `parallel_groups`: Groups of steps that can run in parallel (optional)

## EXAMPLE:

**Input Plan:**
```
1. Write report
1.1 Gather data
1.2 Analyze data
1.3 Write summary
2. Review report
2.1 Check grammar
2.2 Verify facts
```

**Output:**
```json
{{
  "dependencies": [
    {{"from": "1.1", "to": "1.2", "reason": "Analysis requires gathered data"}},
    {{"from": "1.2", "to": "1.3", "reason": "Summary requires analyzed data"}},
    {{"from": "1.3", "to": "2", "reason": "Can't review incomplete report"}},
    {{"from": "2", "to": "2.1", "reason": "Review starts grammar check"}},
    {{"from": "2", "to": "2.2", "reason": "Review starts fact verification"}}
  ],
  "parallel_groups": [
    ["2.1", "2.2"]
  ]
}}
```

## IMPORTANT GUIDELINES:
- DO NOT simply copy the hierarchical structure
- Child steps (e.g., 1.1, 1.2) may NOT depend on each other unless logically necessary
- Sibling steps can often run in parallel
- Only include edges for DIRECT dependencies (no transitive edges)
- If Step A → B and B → C exist, do NOT add A → C

Now, analyze the following plan:

**Query**: {query}

**Plan**:
{hierarchical_plan}

Provide your dependency annotation in JSON format.
"""


# ============================================================================
# PROMPT TEMPLATE 2: Node Weight Annotation
# ============================================================================

NODE_WEIGHT_ANNOTATION_PROMPT = """
You are an expert in estimating task complexity and execution time. Your task is to assign a **weight** (execution time/complexity) to each step in a plan.

## INPUT:
**User Query**: {query}

**Steps**:
{steps_list}

## YOUR TASK:
Estimate the relative execution time or complexity for each step on a scale of 1-10:
- 1-2: Very quick (< 1 minute) - e.g., simple checks, lookups
- 3-4: Quick (1-5 minutes) - e.g., simple edits, basic operations
- 5-6: Moderate (5-15 minutes) - e.g., data processing, file operations
- 7-8: Long (15-60 minutes) - e.g., complex analysis, large file operations
- 9-10: Very long (> 1 hour) - e.g., training models, extensive computations

## OUTPUT FORMAT:
Provide your answer as a JSON object:

```json
{{
  "weights": [
    {{"step_id": "1", "weight": 5, "reasoning": "explanation"}},
    {{"step_id": "1.1", "weight": 3, "reasoning": "explanation"}},
    ...
  ]
}}
```

## EXAMPLE:

**Input:**
```
1. Create file
1.1 Get file creation information
1.2 Use file writing tool to create and write content
1.3 Confirm file creation success
```

**Output:**
```json
{{
  "weights": [
    {{"step_id": "1", "weight": 1, "reasoning": "Parent step, no actual work"}},
    {{"step_id": "1.1", "weight": 2, "reasoning": "Quick information gathering"}},
    {{"step_id": "1.2", "weight": 4, "reasoning": "File write operation takes moderate time"}},
    {{"step_id": "1.3", "weight": 1, "reasoning": "Simple verification check"}}
  ]
}}
```

Now, estimate the weights for the following steps:

**Query**: {query}

**Steps**:
{steps_list}

Provide your weight annotations in JSON format.
"""


# ============================================================================
# PROMPT TEMPLATE 3: Tool Parameter Completeness Check
# ============================================================================

TOOL_PARAMETER_CHECK_PROMPT = """
You are an expert in API design and tool usage validation. Your task is to check if tool parameters are complete and correctly specified.

## INPUT:
**User Query**: {query}

**Tool Call**:
- Step: {step}
- Tool: {tool_name}
- Tool Description: {tool_description}
- Expected Parameters: {tool_arguments}
- Provided Parameters: {provided_params}

## YOUR TASK:
1. Check if all required parameters are provided
2. Verify parameter values are reasonable given the query context
3. Identify any missing or incorrect parameters
4. Suggest corrections if needed

## OUTPUT FORMAT:
```json
{{
  "is_complete": true/false,
  "missing_parameters": ["param1", "param2"],
  "incorrect_parameters": [
    {{"param": "param_name", "issue": "description", "suggested_value": "value"}}
  ],
  "completeness_score": 0.0-1.0
}}
```

## EXAMPLE:

**Input:**
- Query: "Create a file called 'test.txt' with content 'Hello World'"
- Tool: file_write
- Expected: {{"file_path": "string", "content": "string"}}
- Provided: {{"file_path": "test.txt"}}

**Output:**
```json
{{
  "is_complete": false,
  "missing_parameters": ["content"],
  "incorrect_parameters": [],
  "completeness_score": 0.5,
  "suggestions": {{"content": "Hello World"}}
}}
```

Now, check the following tool call:

**Query**: {query}
**Tool**: {tool_name}
**Description**: {tool_description}
**Expected Parameters**: {tool_arguments}
**Provided Parameters**: {provided_params}

Provide your analysis in JSON format.
"""


# ============================================================================
# Helper Functions
# ============================================================================

def format_hierarchical_plan(plan_steps: list) -> str:
    """Format steps into a readable hierarchical plan string."""
    lines = []
    for step in plan_steps:
        step_id = step.get('id', '')
        content = step.get('content', '')
        lines.append(f"{step_id} {content}")
    
    return '\n'.join(lines)


def format_steps_list(steps: list) -> str:
    """Format steps as a bullet list."""
    lines = []
    for step in steps:
        step_id = step.get('id', step.get('step', ''))
        content = step.get('content', step.get('step', ''))
        lines.append(f"- {step_id}: {content}")
    
    return '\n'.join(lines)


def generate_dependency_annotation_prompt(query: str, hierarchical_plan: str) -> str:
    """Build prompt for dependency annotation."""
    return DEPENDENCY_ANNOTATION_PROMPT.format(
        query=query,
        hierarchical_plan=hierarchical_plan
    )


def generate_weight_annotation_prompt(query: str, steps: list) -> str:
    """Build prompt for weight annotation."""
    steps_list = format_steps_list(steps)
    return NODE_WEIGHT_ANNOTATION_PROMPT.format(
        query=query,
        steps_list=steps_list
    )


def generate_tool_parameter_check_prompt(query: str, step: str, tool_name: str,
                                        tool_description: str, tool_arguments: dict,
                                        provided_params: dict) -> str:
    """Build prompt for tool parameter validation."""
    return TOOL_PARAMETER_CHECK_PROMPT.format(
        query=query,
        step=step,
        tool_name=tool_name,
        tool_description=tool_description,
        tool_arguments=json.dumps(tool_arguments, indent=2),
        provided_params=json.dumps(provided_params, indent=2)
    )


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    import json
    
    # Example 1: Dependency Annotation
    query = "Create a file and write some content"
    plan = """1. Create file
1.1 Get file creation information
1.2 Use file writing tool to create and write content
1.3 Confirm file creation success"""
    
    prompt = generate_dependency_annotation_prompt(query, plan)
    print("="*80)
    print("DEPENDENCY ANNOTATION PROMPT")
    print("="*80)
    print(prompt)
    print("\n")
    
    # Example 2: Weight Annotation
    steps = [
        {"id": "1", "content": "Create file"},
        {"id": "1.1", "content": "Get file creation information"},
        {"id": "1.2", "content": "Use file writing tool to create and write content"},
        {"id": "1.3", "content": "Confirm file creation success"}
    ]
    
    prompt = generate_weight_annotation_prompt(query, steps)
    print("="*80)
    print("WEIGHT ANNOTATION PROMPT")
    print("="*80)
    print(prompt)
    
    # Save prompts to file
    prompts_doc = {
        "dependency_annotation": DEPENDENCY_ANNOTATION_PROMPT,
        "node_weight_annotation": NODE_WEIGHT_ANNOTATION_PROMPT,
        "tool_parameter_check": TOOL_PARAMETER_CHECK_PROMPT
    }
    
    with open('/mnt/user-data/outputs/annotation_prompts.json', 'w', encoding='utf-8') as f:
        json.dump(prompts_doc, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("Prompts saved to: /mnt/user-data/outputs/annotation_prompts.json")