import json
import re
import uuid
import asyncio
from datetime import datetime
from sqlalchemy.orm import Session
from app.models import workflow
from app.models.workflow import Workflow
from app.models.tool import Tool
from app.services import tool_service, conversation_session_service, knowledge_base_service, workflow_service, memory_service, geocoding_service
from app.schemas.conversation_session import ConversationSessionUpdate
from app.schemas.memory import MemoryCreate
from app.services.graph_execution_engine import GraphExecutionEngine
from app.services.llm_tool_service import LLMToolService
from app.services.workflow_intent_service import WorkflowIntentService
from app.services.input_validation_service import InputValidationService, ValidationMode, ValidationResult
from app.core.config import settings

import httpx
import numexpr


class WorkflowExecutionService:
    def __init__(self, db: Session):
        self.db = db
        self.llm_tool_service = LLMToolService(db)
        self.workflow_intent_service = WorkflowIntentService(db)
        self.input_validation_service = InputValidationService()

    async def _execute_tool(self, tool_name: str, params: dict, company_id: int = None, session_id: str = None):
        """
        Execute a tool using the unified tool execution service.
        Supports builtin, custom, and MCP tools.
        """
        # The "listen" tool is a special case that signals a pause.
        if tool_name == "listen_for_input":
            return {"status": "paused_for_input"}

        # The "prompt" tool signals a pause and sends data to the frontend.
        if tool_name == "prompt_for_input":
            return {
                "status": "paused_for_prompt",
                "prompt": {
                    "text": params.get("prompt_text", "Please provide input."),
                    "options": params.get("options", [])
                }
            }

        # Use unified tool execution for all tool types (builtin, custom, MCP)
        from app.services import tool_execution_service

        result = await tool_execution_service.execute_tool(
            db=self.db,
            tool_name=tool_name,
            parameters=params,
            session_id=session_id,
            company_id=company_id
        )

        if result is None:
            return {"error": f"Tool '{tool_name}' not found."}

        # Normalize result format for workflow engine
        if "result" in result:
            return {"output": result["result"]}
        return result

    def _resolve_placeholders(self, value: str, context: dict, results: dict):
        """Resolves placeholders like {{context.variable}}, {{context.obj.key}}, or {{step_id.output}}."""
        if not isinstance(value, str) or '{{' not in value:
            return value

        print(f"DEBUG: Resolving placeholders in: '{value}' with context: {context}")

        def drill_down(obj, keys):
            """Helper to drill down into nested objects/dicts."""
            for key in keys:
                if isinstance(obj, dict):
                    obj = obj.get(key, '')
                else:
                    return ''
            return obj

        def resolve_single_placeholder(placeholder: str):
            """Resolve a single placeholder and return the actual value (preserving type)."""
            path = placeholder.split(".")
            source = path[0]

            resolved_value = ''
            if source == "context":
                remaining_path = path[1:]
                resolved_value = drill_down(context, remaining_path)
                print(f"    - Source: context, Path: {remaining_path}, Value: '{resolved_value}'")
            else:
                step_result = results.get(source)
                print(f"    - Source: results, Step: {source}, Result: {step_result}")
                if step_result:
                    remaining_path = path[1:]
                    resolved_value = drill_down(step_result, remaining_path) if remaining_path else step_result

                    if not remaining_path:
                        output_value = step_result.get("output")
                        if isinstance(output_value, dict):
                            resolved_value = output_value.get("content", '')
                        elif output_value is None:
                            resolved_value = ''
                        else:
                            resolved_value = output_value
                print(f"    - Resolved value: '{resolved_value}'")

            return resolved_value

        # Check if the entire value is a single placeholder (e.g., "{{code-123.output.show_dict}}")
        # If so, return the actual value (dict, list, etc.) instead of converting to string
        # Use [^{}]+ to ensure we don't match strings with multiple placeholders
        single_placeholder_match = re.match(r"^\s*\{\{([^{}]+)\}\}\s*$", value)
        if single_placeholder_match:
            placeholder = single_placeholder_match.group(1).strip()
            print(f"  - Found single placeholder: {placeholder}")
            resolved = resolve_single_placeholder(placeholder)
            print(f"DEBUG: Returning actual value (type: {type(resolved).__name__}): {resolved}")
            return resolved

        # For embedded placeholders in text, convert to strings
        def replace_func(match):
            placeholder = match.group(1).strip()
            print(f"  - Found placeholder: {placeholder}")
            resolved_value = resolve_single_placeholder(placeholder)
            return str(resolved_value) if resolved_value is not None else ''

        resolved_string = re.sub(r"\{\{(.*?)\}\}", replace_func, value)
        print(f"DEBUG: Final resolved string: '{resolved_string}'")
        return resolved_string

    def _clear_validation_state(self, context: dict):
        """Clear all prompt validation-related state from context."""
        keys_to_clear = [
            "pending_prompt_options",
            "pending_allow_text_input",
            "pending_validation_mode",
            "pending_validation_llm_provider",
            "pending_validation_llm_model",
            "pending_prompt_text",
            "_validation_retry_count",
            "_validation_max_retries",
        ]
        for key in keys_to_clear:
            context.pop(key, None)

    def _clear_listen_validation_state(self, context: dict):
        """Clear all listen validation-related state from context."""
        keys_to_clear = [
            "pending_listen_validation_mode",
            "pending_listen_validation_llm_provider",
            "pending_listen_validation_llm_model",
            "pending_question_text",
            "_listen_validation_retry_count",
            "_listen_validation_max_retries",
        ]
        for key in keys_to_clear:
            context.pop(key, None)

    async def _execute_data_manipulation_node(self, node_data: dict, context: dict, results: dict):
        from types import SimpleNamespace

        expression = node_data.get("expression", "")
        output_variable = node_data.get("output_variable", "output")

        # Helper function to convert nested dicts to SimpleNamespace for dot notation access
        def dict_to_namespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [dict_to_namespace(item) for item in d]
            return d

        # Create namespace versions for dot notation access
        context_ns = dict_to_namespace(context)
        results_ns = dict_to_namespace(results)

        # Create a safe execution environment for eval
        # Allow both dict access (context['key']) and dot notation (context.key)
        safe_globals = {"__builtins__": None}
        safe_locals = {
            "context": context_ns,  # Dot notation access
            "ctx": context,         # Dict access alternative
            "results": results_ns,  # Dot notation access
            "res": results          # Dict access alternative
        }

        try:
            # Resolve placeholders in the expression before evaluation
            resolved_expression = self._resolve_placeholders(expression, context, results)

            # Run eval in thread pool to avoid blocking the event loop
            def run_eval():
                return eval(resolved_expression, safe_globals, safe_locals)

            manipulated_data = await asyncio.to_thread(run_eval)

            # Store the result in the context
            context[output_variable] = manipulated_data

            return {"output": manipulated_data}
        except Exception as e:
            return {"error": f"Error manipulating data: {e}"}

    async def _execute_code_node(self, node_data: dict, context: dict, results: dict):
        import time
        start_time = time.time()
        node_id = node_data.get("id", "unknown")
        print(f"[CODE NODE] ========== EXECUTION STARTED ==========")
        print(f"[CODE NODE] Node ID: {node_id}, Start Time: {time.strftime('%H:%M:%S')}")
        
        code = node_data.get("code", "")
        arguments = node_data.get("arguments", [])  # [{name: "arg1", value: "{{context.var}}"}]
        return_variables = node_data.get("return_variables", [])  # ["result1", "result2"]

        # Resolve argument values from placeholders
        resolved_args = {}
        arg_names_ordered = []  # Keep track of argument order for function calls
        for arg in arguments:
            arg_name = arg.get("name", "")
            arg_value = arg.get("value", "")
            if arg_name:
                resolved_value = self._resolve_placeholders(str(arg_value), context, results)
                # If resolved_value is already a dict/list, use it directly
                if isinstance(resolved_value, (dict, list)):
                    resolved_args[arg_name] = resolved_value
                # Try to parse as Python literal if it looks like a dict/list string
                elif isinstance(resolved_value, str) and (resolved_value.startswith('{') or resolved_value.startswith('[')):
                    try:
                        import ast
                        resolved_args[arg_name] = ast.literal_eval(resolved_value)
                    except (ValueError, SyntaxError):
                        resolved_args[arg_name] = resolved_value
                else:
                    resolved_args[arg_name] = resolved_value
                arg_names_ordered.append(arg_name)

        print(f"[CODE NODE] Arguments: {resolved_args}, Return vars: {return_variables}")

        # Build execution scope with arguments directly available
        # execution_scope = {
        #     "context": context,
        #     "results": results,
        #     "db": self.db,
        #     "output": None,  # Legacy support for setting output directly
        #     **resolved_args  # Spread arguments into scope so they're directly accessible
        # }

        # Define synchronous code execution function to run in thread pool
        def run_code_sync():
            try:
                # Execute the code
                exec(code, execution_scope)

                # Check if a function was defined and should be auto-called
                # Only auto-call if the function wasn't already called in the code
                import re
                func_match = re.search(r'def\s+(\w+)\s*\(', code)
                if func_match:
                    func_name = func_match.group(1)
                    # Check if function was manually called in the code (look for "func_name(" after the def block)
                    func_call_pattern = rf'{func_name}\s*\('
                    # Find all calls - if there's a call outside the def, user called it manually
                    func_def_end = code.find('def ' + func_name)
                    code_after_def = code[func_def_end:] if func_def_end >= 0 else ""
                    # Check if there's a call that's not the def line itself
                    lines_after_def = code_after_def.split('\n')[1:]  # Skip the def line
                    manual_call_exists = any(re.search(func_call_pattern, line) and not line.strip().startswith('def ') for line in lines_after_def)

                    # Also check if return variables are already set (user assigned them manually)
                    return_vars_already_set = return_variables and all(
                        var_name.strip() in execution_scope and execution_scope[var_name.strip()] is not None
                        for var_name in return_variables if var_name.strip()
                    )

                    if not manual_call_exists and not return_vars_already_set:
                        if func_name in execution_scope and callable(execution_scope[func_name]):
                            # Call the function with arguments in order
                            func = execution_scope[func_name]
                            arg_values = [resolved_args[name] for name in arg_names_ordered if name in resolved_args]
                            print(f"[CODE NODE] Auto-calling function '{func_name}' with args: {arg_values}")
                            func_result = func(*arg_values)

                            # If there's one return variable, assign the function result to it
                            if return_variables and len(return_variables) == 1:
                                var_name = return_variables[0].strip()
                                execution_scope[var_name] = func_result
                                context[var_name] = func_result
                                print(f"[CODE NODE] Output: {{{var_name}: {func_result}}}")
                                return {"output": {var_name: func_result}}
                            elif return_variables and len(return_variables) > 1 and isinstance(func_result, (tuple, list)):
                                # Multiple return values
                                output = {}
                                for i, var_name in enumerate(return_variables):
                                    var_name = var_name.strip()
                                    if i < len(func_result):
                                        output[var_name] = func_result[i]
                                        context[var_name] = func_result[i]
                                print(f"[CODE NODE] Output: {output}")
                                return {"output": output if output else func_result}
                            else:
                                # No return variables defined, just return the function result
                                print(f"[CODE NODE] Output: {func_result}")
                                return {"output": func_result}

                # Collect return variables into output (for non-function code or manually called functions)
                if return_variables:
                    output = {}
                    for var_name in return_variables:
                        var_name = var_name.strip()
                        if var_name and var_name in execution_scope:
                            output[var_name] = execution_scope[var_name]
                            # Also store in context for later use in workflow
                            context[var_name] = execution_scope[var_name]

                    print(f"[CODE NODE] Output: {output}")
                    return {"output": output if output else "Code executed successfully."}
                else:
                    # Legacy behavior: return the 'output' variable if set
                    return {"output": execution_scope.get("output", "Code executed successfully.")}

            except Exception as e:
                import traceback
                print(f"[CODE NODE] Error: {e}")
                return {"error": f"Error executing code: {e}", "traceback": traceback.format_exc()}

        # Build execution scope with arguments directly available
        execution_scope = {
            "context": context,
            "results": results,
            "db": self.db,
            "output": None,  # Legacy support for setting output directly
            **resolved_args  # Spread arguments into scope so they're directly accessible
        }

        # Run the synchronous code execution in a thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(run_code_sync)
        
        elapsed_time = time.time() - start_time
        print(f"[CODE NODE] ========== EXECUTION FINISHED ==========")
        print(f"[CODE NODE] Node ID: {node_id}, Elapsed: {elapsed_time:.3f}s, End Time: {time.strftime('%H:%M:%S')}")
        return result

    async def _execute_knowledge_retrieval_node(self, node_data: dict, context: dict, results: dict, company_id: int, workflow):
        knowledge_base_id = node_data.get("knowledge_base_id")
        query = node_data.get("query", "")
        resolved_query = self._resolve_placeholders(query, context, results)

        if not knowledge_base_id:
            return {"error": "Knowledge Base ID is required for knowledge retrieval node."}

        try:
            # Find relevant chunks from knowledge base (pass agent for correct embedding model)
            retrieved_chunks = knowledge_base_service.find_relevant_chunks(
                self.db, knowledge_base_id, company_id, resolved_query, top_k=5,
                agent=self._executing_agent
            )

            if not retrieved_chunks:
                return {"output": "I couldn't find any relevant information for your query."}

            # Format chunks as human-readable text (without LLM call)
            # Join chunks with separators for readability
            formatted_response = "\n\n---\n\n".join(retrieved_chunks)

            return {"output": formatted_response}

        except Exception as e:
            return {"error": f"Error retrieving knowledge: {e}"}

    def _evaluate_single_condition(self, variable_placeholder: str, operator: str, comparison_value: str, context: dict, results: dict) -> bool:
        """Evaluate a single condition and return True/False."""
        # Resolve the variable placeholder to get the actual value from the context or results
        actual_value = self._resolve_placeholders(variable_placeholder, context, results)

        print(f"    - Variable '{variable_placeholder}' resolved to: '{actual_value}' (type: {type(actual_value)})")
        print(f"    - Operator: '{operator}', Comparison Value: '{comparison_value}'")

        # Coerce types for comparison where possible
        try:
            if isinstance(actual_value, (int, float)):
                comparison_value = type(actual_value)(comparison_value)
        except (ValueError, TypeError):
            pass

        result = False
        if operator == "equals":
            if isinstance(actual_value, str) and isinstance(comparison_value, str):
                result = actual_value.lower().strip() == comparison_value.lower().strip()
            else:
                result = actual_value == comparison_value
        elif operator == "not_equals":
            if isinstance(actual_value, str) and isinstance(comparison_value, str):
                result = actual_value.lower().strip() != comparison_value.lower().strip()
            else:
                result = actual_value != comparison_value
        elif operator == "contains":
            result = str(comparison_value).lower() in str(actual_value).lower()
        elif operator == "greater_than":
            try:
                result = float(actual_value) > float(comparison_value)
            except (ValueError, TypeError):
                result = False
        elif operator == "less_than":
            try:
                result = float(actual_value) < float(comparison_value)
            except (ValueError, TypeError):
                result = False
        elif operator == "is_set":
            result = actual_value is not None and actual_value != ''
        elif operator == "is_not_set":
            result = actual_value is None or actual_value == ''

        print(f"    - Result: {result}")
        return result

    def _execute_conditional_node(self, node_data: dict, context: dict, results: dict):
        """
        Execute a conditional node with support for multiple conditions (if/elseif/else).

        Supports two formats:
        1. Legacy single condition: {"variable": "...", "operator": "...", "value": "..."}
           - Returns {"output": True/False} for true/false handles

        2. Multi-condition: {"conditions": [{"variable": "...", "operator": "...", "value": "..."}, ...]}
           - Returns {"output": index} for the first matching condition (handle "0", "1", "2", etc.)
           - Returns {"output": "else"} if no condition matches (handle "else")
        """
        conditions = node_data.get("conditions", [])

        # Check if using new multi-condition format
        if conditions and isinstance(conditions, list) and len(conditions) > 0:
            print(f"DEBUG: Executing multi-condition node with {len(conditions)} conditions:")

            for index, condition in enumerate(conditions):
                variable = condition.get("variable", "")
                operator = condition.get("operator", "equals")
                value = condition.get("value", "")

                print(f"  Condition {index} (handle '{index}'):")
                if self._evaluate_single_condition(variable, operator, value, context, results):
                    print(f"  ✓ Condition {index} matched! Routing to handle '{index}'")
                    return {"output": index}  # Return index for routing

            # No condition matched, return else
            print(f"  ✗ No conditions matched. Routing to 'else' handle")
            return {"output": "else"}

        else:
            # Legacy single condition format (backward compatible)
            variable_placeholder = node_data.get("variable", "")
            operator = node_data.get("operator", "equals")
            comparison_value = node_data.get("value", "")

            print(f"DEBUG: Executing single conditional node:")
            result = self._evaluate_single_condition(variable_placeholder, operator, comparison_value, context, results)
            print(f"  - Condition evaluated to: {result}")
            return {"output": result}

    def _execute_foreach_loop_node(self, node_data: dict, context: dict, results: dict, node_id: str):
        """
        Execute a For Each loop node.

        On first execution: Initialize loop state, check if array is empty.
        On subsequent executions: Increment index and check if more items.

        Returns:
            {"output": "loop"} - Continue iterating (body should execute)
            {"output": "exit"} - Loop complete or empty array
        """
        array_source = node_data.get("array_source", "")
        item_var = node_data.get("item_variable", "item")
        index_var = node_data.get("index_variable", "index")

        # Context keys for this specific loop instance
        loop_index_key = f"_loop_{node_id}_index"
        loop_array_key = f"_loop_{node_id}_array"

        # Check if this is first execution (no index in context yet)
        if loop_index_key not in context:
            # First execution - resolve array and initialize
            resolved_array = self._resolve_placeholders(array_source, context, results)

            # Ensure it's a list
            if isinstance(resolved_array, str):
                try:
                    resolved_array = json.loads(resolved_array)
                except json.JSONDecodeError:
                    resolved_array = []

            if not isinstance(resolved_array, list):
                # Try to convert dict keys to list
                if isinstance(resolved_array, dict):
                    resolved_array = list(resolved_array.items())
                else:
                    resolved_array = [resolved_array] if resolved_array else []

            # Store array in context for iteration
            context[loop_array_key] = resolved_array

            if len(resolved_array) == 0:
                # Empty array - exit immediately
                print(f"DEBUG: [ForEach] Empty array, exiting loop")
                return {"output": "exit"}

            # Initialize index to 0
            context[loop_index_key] = 0
            context[item_var] = resolved_array[0]
            context[index_var] = 0

            print(f"DEBUG: [ForEach] Starting loop with {len(resolved_array)} items")
            print(f"DEBUG: [ForEach] First item: {resolved_array[0]}")
            return {"output": "loop"}

        else:
            # Subsequent execution - increment index
            current_index = context[loop_index_key]
            array = context[loop_array_key]
            next_index = current_index + 1

            if next_index >= len(array):
                # Loop complete - clean up and exit
                print(f"DEBUG: [ForEach] Loop complete after {len(array)} iterations")
                del context[loop_index_key]
                del context[loop_array_key]
                return {"output": "exit"}

            # Continue to next item
            context[loop_index_key] = next_index
            context[item_var] = array[next_index]
            context[index_var] = next_index

            print(f"DEBUG: [ForEach] Iteration {next_index + 1}/{len(array)}, item: {array[next_index]}")
            return {"output": "loop"}

    def _execute_while_loop_node(self, node_data: dict, context: dict, results: dict, node_id: str):
        """
        Execute a While loop node.

        Evaluates condition(s) and returns 'loop' to continue or 'exit' when false.

        Returns:
            {"output": "loop"} - Condition is true, continue iterating
            {"output": "exit"} - Condition is false, exit loop
        """
        iteration_key = f"_loop_{node_id}_iteration"

        # Track iteration count (for debugging)
        if iteration_key not in context:
            context[iteration_key] = 0
        else:
            context[iteration_key] += 1

        iteration = context[iteration_key]

        conditions = node_data.get("conditions", [])

        if conditions and isinstance(conditions, list) and len(conditions) > 0:
            # Multi-condition: ALL conditions must be true (AND logic)
            print(f"DEBUG: [While] Iteration {iteration}, evaluating {len(conditions)} conditions")

            all_true = True
            for idx, condition in enumerate(conditions):
                variable = condition.get("variable", "")
                operator = condition.get("operator", "equals")
                value = condition.get("value", "")

                result = self._evaluate_single_condition(variable, operator, value, context, results)
                print(f"DEBUG: [While] Condition {idx}: {variable} {operator} {value} = {result}")

                if not result:
                    all_true = False
                    break

            if all_true:
                print(f"DEBUG: [While] All conditions true, continuing loop")
                return {"output": "loop"}
            else:
                print(f"DEBUG: [While] Condition(s) false, exiting loop after {iteration} iterations")
                del context[iteration_key]
                return {"output": "exit"}

        else:
            # No conditions - exit immediately (prevents infinite loop)
            print(f"DEBUG: [While] No conditions configured, exiting loop")
            if iteration_key in context:
                del context[iteration_key]
            return {"output": "exit"}

    async def _execute_http_request_node(self, node_data: dict, context: dict, results: dict):
        url = node_data.get("url", "")
        method = node_data.get("method", "GET").upper()
        headers_str = node_data.get("headers", "{}")
        body_str = node_data.get("body", "{}")

        resolved_url = self._resolve_placeholders(url, context, results)
        resolved_headers_str = self._resolve_placeholders(headers_str, context, results)
        resolved_body_str = self._resolve_placeholders(body_str, context, results)

        try:
            headers = json.loads(resolved_headers_str)
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON in headers: {resolved_headers_str}"}

        try:
            body = json.loads(resolved_body_str) if method in ["POST", "PUT", "PATCH"] else None
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON in body: {resolved_body_str}"}

        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_REQUEST_TIMEOUT) as client:
                response = None
                if method == "GET":
                    response = await client.get(resolved_url, headers=headers)
                elif method == "POST":
                    response = await client.post(resolved_url, headers=headers, json=body)
                elif method == "PUT":
                    response = await client.put(resolved_url, headers=headers, json=body)
                elif method == "PATCH":
                    response = await client.patch(resolved_url, headers=headers, json=body)
                elif method == "DELETE":
                    response = await client.delete(resolved_url, headers=headers)
                else:
                    return {"error": f"Unsupported HTTP method: {method}"}

                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    return {"output": response.json()}
                else:
                    return {"output": response.text}
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except httpx.RequestError as e:
            return {"error": f"HTTP request failed: {e}"}
        except Exception as e:
            return {"error": f"Error executing HTTP request: {e}"}

    async def _execute_llm_node(self, node_data: dict, context: dict, results: dict, company_id: int, workflow, conversation_id: str):
        prompt = node_data.get("prompt", "")
        resolved_prompt = self._resolve_placeholders(prompt, context, results)

        # 1. Get the system prompt - use custom if provided, otherwise fall back to agent's prompt
        custom_system_prompt = node_data.get("system_prompt", "")
        if custom_system_prompt:
            # Resolve any placeholders in the custom system prompt
            system_prompt = self._resolve_placeholders(custom_system_prompt, context, results)
        else:
            # Fall back to agent's system prompt
            system_prompt = self._executing_agent.prompt if self._executing_agent else "You are a helpful assistant."

        # 2. Get the chat history
        chat_history = []
        if conversation_id:
            # Assuming a function exists to get chat messages by conversation_id
            # This might need to be created in chat_service.py
            history_messages = conversation_session_service.get_chat_history(self.db, conversation_id)
            for msg in history_messages:
                role = "assistant" if msg.sender == "agent" else msg.sender
                chat_history.append({"role": role, "content": msg.message})

        # 3. Get the tools associated with the agent
        agent_tools = self._executing_agent.tools if self._executing_agent else []

        # 4. Get attachments from context (for vision model support)
        # Only include attachments if agent has vision_enabled
        attachments = []
        if self._executing_agent and getattr(self._executing_agent, 'vision_enabled', False):
            # First check user_attachments (set during workflow execution)
            attachments = context.get("user_attachments", [])

            # Also check if any context variable contains attachments (from Listen node)
            # This handles cases where Listen node saved {text, attachments} format
            if not attachments:
                for key, value in context.items():
                    if isinstance(value, dict) and "attachments" in value:
                        attachments = value.get("attachments", [])
                        if attachments:
                            print(f"DEBUG: Found attachments in context variable '{key}'")
                            break
        else:
            print(f"DEBUG: Vision not enabled for agent, skipping attachments")

        llm_response = await self.llm_tool_service.execute(
            model=node_data.get("model"),
            system_prompt=system_prompt,
            chat_history=chat_history,
            user_prompt=resolved_prompt,
            tools=agent_tools,
            knowledge_base_id=node_data.get("knowledge_base_id"),
            company_id=company_id,
            attachments=attachments
        )

        # Extract the content from the LLM response
        if isinstance(llm_response, dict):
            response_text = llm_response.get("content", "")
        else:
            response_text = str(llm_response)

        return {"output": response_text}

    # ============================================================
    # NEW CHAT-SPECIFIC NODE EXECUTION METHODS
    # ============================================================

    def _execute_intent_router_node(self, node_data: dict, context: dict, results: dict):
        """
        Routes workflow based on detected intent in context.
        Returns intent_name to determine which edge to follow.
        """
        detected_intent = context.get("detected_intent")
        intent_confidence = context.get("intent_confidence", 0.0)

        routes = node_data.get("routes", [])

        # Check if detected intent matches any configured route
        for route in routes:
            intent_name = route.get("intent_name")
            min_confidence = route.get("min_confidence", 0.7)

            if detected_intent == intent_name and intent_confidence >= min_confidence:
                print(f"✓ Intent router: Routing to '{intent_name}' (confidence: {intent_confidence:.2f})")
                return {
                    "output": intent_name,
                    "route": intent_name,
                    "confidence": intent_confidence
                }

        # No matching route, use default
        print(f"✓ Intent router: Using default route (no intent match)")
        return {
            "output": "default",
            "route": "default",
            "confidence": 0.0
        }

    async def _execute_entity_collector_node(
        self, node_data: dict, context: dict, results: dict, workflow: Workflow, conversation_id: str
    ):
        """
        Collects required entities from context or prompts user for missing ones.
        """
        entities_to_collect = node_data.get("entities_to_collect", [])
        collection_strategy = node_data.get("collection_strategy", "ask_if_missing")
        prompts = node_data.get("prompts", {})
        max_attempts = node_data.get("max_attempts", 3)

        missing_entities = []
        collected_entities = {}

        # Check which entities are already in context
        for entity_name in entities_to_collect:
            if entity_name in context and context[entity_name]:
                collected_entities[entity_name] = context[entity_name]
                print(f"✓ Entity '{entity_name}' already in context: {context[entity_name]}")
            else:
                missing_entities.append(entity_name)
                print(f"✗ Entity '{entity_name}' missing from context")

        if not missing_entities:
            # All entities collected
            return {
                "output": collected_entities,
                "status": "complete",
                "collected": collected_entities
            }

        if collection_strategy == "extract_only":
            # Don't prompt, just return what we have
            return {
                "output": collected_entities,
                "status": "partial",
                "collected": collected_entities,
                "missing": missing_entities
            }

        # Ask for first missing entity
        first_missing = missing_entities[0]
        prompt_text = prompts.get(first_missing, f"Please provide your {first_missing}")

        print(f"ℹ Prompting user for entity '{first_missing}'")

        return {
            "status": "paused_for_prompt",
            "prompt": {
                "text": prompt_text,
                "options": []
            },
            "collecting_entity": first_missing,
            "remaining_entities": missing_entities
        }

    def _execute_check_entity_node(self, node_data: dict, context: dict, results: dict):
        """
        Checks if a specific entity exists in context.
        Returns boolean for routing (true/false edges).
        """
        entity_name = node_data.get("entity_name")
        check_type = node_data.get("check_type", "exists")  # exists, not_empty, valid

        entity_value = context.get(entity_name)

        if check_type == "exists":
            has_entity = entity_name in context
        elif check_type == "not_empty":
            has_entity = entity_name in context and entity_value not in [None, "", []]
        elif check_type == "valid":
            # Could add regex validation here
            validation_regex = node_data.get("validation_regex")
            if validation_regex and entity_value:
                import re
                has_entity = bool(re.match(validation_regex, str(entity_value)))
            else:
                has_entity = entity_name in context and entity_value is not None
        else:
            has_entity = False

        print(f"✓ Check entity '{entity_name}': {has_entity} (value: {entity_value})")

        return {
            "output": has_entity,
            "entity_name": entity_name,
            "entity_value": entity_value,
            "check_result": has_entity
        }

    def _execute_update_context_node(self, node_data: dict, context: dict, results: dict):
        """
        Updates context with new variables or values.
        Supports both single variable (variable_name/variable_value) and multiple variables (variables dict).
        """
        updated_vars = {}

        # Handle single variable format from frontend (variable_name + variable_value)
        var_name = node_data.get("variable_name")
        var_value = node_data.get("variable_value")
        update_mode = node_data.get("update_mode", "set")

        if var_name:
            # Resolve placeholders in value
            resolved_value = self._resolve_placeholders(str(var_value) if var_value else "", context, results)

            if update_mode == "append":
                # Append to existing value (for strings/lists)
                existing = context.get(var_name, "")
                if isinstance(existing, list):
                    if isinstance(resolved_value, list):
                        context[var_name] = existing + resolved_value
                    else:
                        context[var_name] = existing + [resolved_value]
                else:
                    context[var_name] = str(existing) + str(resolved_value)
            elif update_mode == "merge":
                # Merge for dicts
                existing = context.get(var_name, {})
                if isinstance(existing, dict) and isinstance(resolved_value, dict):
                    existing.update(resolved_value)
                    context[var_name] = existing
                else:
                    context[var_name] = resolved_value
            else:
                # Default: set/replace
                context[var_name] = resolved_value

            updated_vars[var_name] = context[var_name]
            print(f"✓ Updated context: {var_name} = {context[var_name]} (mode: {update_mode})")

        # Also support legacy variables dict format
        variables = node_data.get("variables", {})
        for name, value in variables.items():
            resolved_value = self._resolve_placeholders(str(value), context, results)
            context[name] = resolved_value
            updated_vars[name] = resolved_value
            print(f"✓ Updated context: {name} = {resolved_value}")

        return {
            "output": "Context updated",
            "updated_variables": updated_vars
        }

    def _execute_tag_conversation_node(self, node_data: dict, context: dict, results: dict, conversation_id: str):
        """
        Adds tags to the conversation for organization and filtering.
        """
        tags = node_data.get("tags", [])

        # Resolve any placeholders in tags
        resolved_tags = []
        for tag in tags:
            resolved_tag = self._resolve_placeholders(str(tag), context, results)
            resolved_tags.append(resolved_tag)

        # Update conversation session with tags
        try:
            session = conversation_session_service.get_session(self.db, conversation_id)
            if session:
                current_tags = session.context.get("tags", []) if session.context else []
                updated_tags = list(set(current_tags + resolved_tags))  # Remove duplicates

                session_context = session.context or {}
                session_context["tags"] = updated_tags

                conversation_session_service.update_session_context(
                    self.db, conversation_id, session_context
                )

                print(f"✓ Added tags to conversation: {resolved_tags}")

                return {
                    "output": "Tags added",
                    "tags_added": resolved_tags,
                    "all_tags": updated_tags
                }
        except Exception as e:
            print(f"✗ Error adding tags: {e}")
            return {"error": f"Failed to add tags: {e}"}

    def _execute_assign_to_agent_node(
        self, node_data: dict, context: dict, results: dict, conversation_id: str, company_id: int
    ):
        """
        Assigns the conversation to a human agent or agent pool.
        """
        assignment_type = node_data.get("assignment_type", "pool")  # pool, specific, round_robin
        agent_id = node_data.get("agent_id")
        pool_name = node_data.get("pool_name", "support")
        priority = node_data.get("priority", "normal")
        notes = node_data.get("notes", "")

        resolved_notes = self._resolve_placeholders(notes, context, results)

        try:
            session = conversation_session_service.get_session(self.db, conversation_id)
            if session:
                # Update status for agent assignment (AI stays enabled - can be toggled manually)
                session_update = ConversationSessionUpdate(
                    status='pending_agent_assignment'
                )
                conversation_session_service.update_session(self.db, conversation_id, session_update)

                # Store assignment info in context
                assignment_info = {
                    "assigned_at": datetime.now().isoformat(),
                    "assignment_type": assignment_type,
                    "pool": pool_name,
                    "priority": priority,
                    "notes": resolved_notes
                }

                if agent_id:
                    assignment_info["agent_id"] = agent_id

                session_context = session.context or {}
                session_context["assignment"] = assignment_info
                conversation_session_service.update_session_context(
                    self.db, conversation_id, session_context
                )

                print(f"✓ Assigned conversation to {assignment_type}: {pool_name or agent_id}")

                return {
                    "output": "Assigned to agent",
                    "assignment": assignment_info
                }
        except Exception as e:
            print(f"✗ Error assigning to agent: {e}")
            return {"error": f"Failed to assign to agent: {e}"}

    def _execute_set_status_node(self, node_data: dict, context: dict, results: dict, conversation_id: str):
        """
        Sets the conversation status (e.g., resolved, pending, escalated).
        """
        status = node_data.get("status", "active")
        reason = node_data.get("reason", "")

        resolved_reason = self._resolve_placeholders(reason, context, results)

        try:
            session_update = ConversationSessionUpdate(
                status=status
            )
            conversation_session_service.update_session(self.db, conversation_id, session_update)

            # Also store in context
            session = conversation_session_service.get_session(self.db, conversation_id)
            if session:
                session_context = session.context or {}
                session_context["status_history"] = session_context.get("status_history", [])
                session_context["status_history"].append({
                    "status": status,
                    "reason": resolved_reason,
                    "timestamp": datetime.now().isoformat()
                })
                conversation_session_service.update_session_context(
                    self.db, conversation_id, session_context
                )

            print(f"✓ Set conversation status to: {status}")

            return {
                "output": f"Status set to {status}",
                "status": status,
                "reason": resolved_reason
            }
        except Exception as e:
            print(f"✗ Error setting status: {e}")
            return {"error": f"Failed to set status: {e}"}

    async def _execute_channel_redirect_node(
        self,
        node_data: dict,
        context: dict,
        results: dict,
        conversation_id: str,
        company_id: int,
        contact_id: int,
        workflow_id: int = None,
        next_node_id: str = None
    ) -> dict:
        """
        Redirects/continues conversation on another messaging channel.

        Supports two redirect types:
        - invite_link: Send message/link to target channel inviting user to continue
        - full_transfer: Migrate conversation context to target channel

        Supports workflow continuation options:
        - original: Continue workflow on original channel (default)
        - transfer: Transfer workflow execution to target channel

        Args:
            node_data: Node configuration parameters
            context: Current workflow context
            results: Results from previous nodes
            conversation_id: Current session's conversation_id
            company_id: Company ID for multi-tenancy
            contact_id: Contact ID for the current user
            workflow_id: Current workflow ID (for transfer)
            next_node_id: Next node ID after this one (for transfer)

        Returns:
            dict with output/error and redirect details
        """
        from app.services import messaging_service, integration_service

        # Extract node parameters
        target_channel = node_data.get("target_channel", "whatsapp")
        redirect_type = node_data.get("redirect_type", "invite_link")
        contact_info_source = node_data.get("contact_info_source", "auto")
        variable_name = node_data.get("variable_name", "")
        original_session_behavior = node_data.get("original_session_behavior", "keep_active")

        # Workflow continuation option
        workflow_continuation = node_data.get("workflow_continuation", "original")

        # Invite link options
        invite_message = node_data.get("invite_message", "Continue our conversation on {{channel}}!")

        # Full transfer options
        copy_context_variables = node_data.get("copy_context_variables", False)
        context_variables_to_copy = node_data.get("context_variables_to_copy", [])
        transfer_message = node_data.get("transfer_message", "Continuing conversation from another channel.")

        # Error handling
        fallback_on_failure = node_data.get("fallback_on_failure", "continue")
        max_retries = node_data.get("max_retries", 3)

        print(f"✓ Channel Redirect: {redirect_type} to {target_channel} (workflow: {workflow_continuation})")

        try:
            # Step 1: Resolve target contact information
            recipient_id, info_source = self._resolve_redirect_contact_info(
                target_channel, contact_info_source, variable_name, contact_id, company_id, context
            )

            if not recipient_id:
                error_msg = f"No contact information found for {target_channel}. Checked: "
                if contact_info_source in ["auto", "contact"]:
                    error_msg += f"contact record, "
                if contact_info_source in ["auto", "variable"] and variable_name:
                    error_msg += f"context.{variable_name}"
                print(f"✗ Channel Redirect: {error_msg}")

                if fallback_on_failure == "error_edge":
                    return {
                        "output": None,
                        "error": "missing_contact_info",
                        "error_message": error_msg,
                        "redirect_attempted": False
                    }
                return {"output": "redirect_skipped", "error_message": error_msg}

            print(f"  → Target recipient: {recipient_id} (source: {info_source})")

            # Step 2: Get integration for target channel
            integration_type = self._get_integration_type_for_channel(target_channel)
            integration = integration_service.get_integration_by_type_and_company(
                self.db, integration_type, company_id
            )

            if not integration:
                error_msg = f"No {target_channel} integration configured for this company"
                print(f"✗ Channel Redirect: {error_msg}")

                if fallback_on_failure == "error_edge":
                    return {
                        "output": None,
                        "error": "missing_integration",
                        "error_message": error_msg,
                        "redirect_attempted": False
                    }
                return {"output": "redirect_skipped", "error_message": error_msg}

            # Step 3: Find existing session or create new one
            # First, try to find an existing session for this contact on the target channel
            target_session = conversation_session_service.get_session_by_contact_and_channel(
                self.db,
                contact_id=contact_id,
                channel=target_channel,
                company_id=company_id
            )

            if target_session:
                print(f"  → Found existing session: {target_session.conversation_id}")
            else:
                # No existing session found, create a new one using recipient_id
                target_workflow_id = workflow_id if workflow_continuation == "transfer" else None
                target_session = conversation_session_service.get_or_create_session(
                    self.db,
                    conversation_id=recipient_id,
                    workflow_id=target_workflow_id,
                    contact_id=contact_id,
                    channel=target_channel,
                    company_id=company_id
                )
                print(f"  → Created new session: {target_session.conversation_id}")

            # Step 4: Execute redirect based on type
            if redirect_type == "invite_link":
                result = await self._execute_invite_link_redirect(
                    target_channel, recipient_id, invite_message,
                    integration, context, results
                )
            else:  # full_transfer
                result = await self._execute_full_transfer_redirect(
                    target_channel, recipient_id, transfer_message,
                    copy_context_variables, context_variables_to_copy,
                    target_session, integration, context, results
                )

            if result.get("error"):
                if fallback_on_failure == "error_edge":
                    return result
                return {"output": "redirect_failed", "error_message": result.get("error_message")}

            # Step 5: Handle workflow transfer if enabled
            stop_execution = False
            if workflow_continuation == "transfer" and workflow_id and next_node_id:
                # Transfer workflow to target session
                target_session.workflow_id = workflow_id
                target_session.next_step_id = next_node_id
                target_session.status = "waiting_for_input"

                # Copy full context to target session
                target_context = target_session.context or {}
                target_context.update(context)
                target_context["_transferred_from_channel"] = {
                    "original_conversation_id": conversation_id,
                    "transferred_at": datetime.now().isoformat()
                }
                conversation_session_service.update_session_context(
                    self.db, target_session.conversation_id, target_context
                )

                self.db.commit()
                print(f"  → Workflow transferred to target session (next_step: {next_node_id})")
                stop_execution = True

            # Step 6: Handle original session behavior
            await self._handle_original_session_behavior(
                original_session_behavior, conversation_id, target_session.conversation_id
            )

            print(f"✓ Channel Redirect: Successfully initiated to {target_channel}")

            return {
                "output": "workflow_transferred" if stop_execution else "redirect_initiated",
                "redirect_type": redirect_type,
                "target_channel": target_channel,
                "target_session_id": target_session.conversation_id,
                "original_session_behavior": original_session_behavior,
                "workflow_continuation": workflow_continuation,
                "stop_execution": stop_execution  # Signal to stop workflow on original channel
            }

        except Exception as e:
            print(f"✗ Channel Redirect Error: {e}")
            if fallback_on_failure == "error_edge":
                return {
                    "output": None,
                    "error": "redirect_failed",
                    "error_message": str(e),
                    "redirect_attempted": True
                }
            return {"output": "redirect_failed", "error_message": str(e)}

    def _resolve_redirect_contact_info(
        self,
        target_channel: str,
        source: str,
        variable_name: str,
        contact_id: int,
        company_id: int,
        context: dict
    ) -> tuple:
        """
        Resolves contact information for the target channel.

        Returns:
            (recipient_id, source_type) or (None, None) if not found
        """
        from app.services import contact_service

        recipient_id = None
        info_source = None

        # Channel to contact field mapping
        channel_field_map = {
            "whatsapp": "phone_number",
            "telegram": "telegram_id",
            "instagram": "instagram_psid",
            "messenger": "messenger_psid"
        }

        # Try contact record first (if source is auto or contact)
        if source in ["auto", "contact"]:
            contact = contact_service.get_contact(self.db, contact_id, company_id)
            if contact:
                field = channel_field_map.get(target_channel)
                if field == "phone_number":
                    recipient_id = contact.phone_number
                elif contact.custom_attributes and field:
                    recipient_id = contact.custom_attributes.get(field)

                if recipient_id:
                    info_source = "contact"

        # Try workflow variable (if source is auto or variable, and not found yet)
        if not recipient_id and source in ["auto", "variable"] and variable_name:
            # Resolve variable from context
            resolved_var = self._resolve_placeholders(f"{{{{{variable_name}}}}}", context, {})
            if resolved_var and resolved_var != f"{{{{{variable_name}}}}}":
                recipient_id = resolved_var
                info_source = "variable"

        return (recipient_id, info_source)

    def _get_integration_type_for_channel(self, channel: str) -> str:
        """Maps channel name to integration type."""
        channel_integration_map = {
            "whatsapp": "whatsapp",
            "telegram": "telegram",
            "instagram": "instagram",
            "messenger": "messenger"
        }
        return channel_integration_map.get(channel, channel)

    async def _execute_invite_link_redirect(
        self,
        target_channel: str,
        recipient_id: str,
        invite_message: str,
        integration,
        context: dict,
        results: dict
    ) -> dict:
        """
        Sends an invite message to the target channel.
        """
        from app.services import messaging_service

        # Resolve placeholders in message
        resolved_message = self._resolve_placeholders(invite_message, context, results)
        resolved_message = resolved_message.replace("{{channel}}", target_channel.title())

        try:
            if target_channel == "whatsapp":
                result = await messaging_service.send_whatsapp_message(
                    recipient_id, resolved_message, integration, self.db
                )
            elif target_channel == "telegram":
                result = await messaging_service.send_telegram_message(
                    int(recipient_id), resolved_message, integration
                )
            elif target_channel == "instagram":
                result = await messaging_service.send_instagram_message(
                    recipient_id, resolved_message, integration
                )
            elif target_channel == "messenger":
                result = await messaging_service.send_messenger_message(
                    recipient_id, resolved_message, integration
                )
            else:
                return {"error": "unsupported_channel", "error_message": f"Channel {target_channel} not supported"}

            print(f"  → Invite message sent to {target_channel}")
            return {"output": "message_sent", "result": result}

        except Exception as e:
            return {"error": "send_failed", "error_message": str(e)}

    async def _execute_full_transfer_redirect(
        self,
        target_channel: str,
        recipient_id: str,
        transfer_message: str,
        copy_context_variables: bool,
        context_variables_to_copy: list,
        target_session,
        integration,
        context: dict,
        results: dict
    ) -> dict:
        """
        Performs full transfer: copies context and sends welcome message to target channel.
        """
        from app.services import messaging_service

        try:
            # Copy context variables if enabled
            if copy_context_variables:
                target_context = target_session.context or {}

                if context_variables_to_copy:
                    # Copy specific variables
                    for var in context_variables_to_copy:
                        if var in context:
                            target_context[var] = context[var]
                else:
                    # Copy all context variables
                    target_context.update(context)

                # Mark as transferred
                target_context["_transferred_from"] = {
                    "channel": "original",
                    "timestamp": datetime.now().isoformat()
                }

                conversation_session_service.update_session_context(
                    self.db, target_session.conversation_id, target_context
                )
                print(f"  → Context copied to target session")

            # Send transfer welcome message
            resolved_message = self._resolve_placeholders(transfer_message, context, results)

            if target_channel == "whatsapp":
                result = await messaging_service.send_whatsapp_message(
                    recipient_id, resolved_message, integration, self.db
                )
            elif target_channel == "telegram":
                result = await messaging_service.send_telegram_message(
                    int(recipient_id), resolved_message, integration
                )
            elif target_channel == "instagram":
                result = await messaging_service.send_instagram_message(
                    recipient_id, resolved_message, integration
                )
            elif target_channel == "messenger":
                result = await messaging_service.send_messenger_message(
                    recipient_id, resolved_message, integration
                )
            else:
                return {"error": "unsupported_channel", "error_message": f"Channel {target_channel} not supported"}

            print(f"  → Transfer message sent to {target_channel}")
            return {"output": "transfer_complete", "result": result}

        except Exception as e:
            return {"error": "transfer_failed", "error_message": str(e)}

    async def _handle_original_session_behavior(
        self,
        behavior: str,
        original_conversation_id: str,
        target_conversation_id: str
    ):
        """
        Handles the original session based on the configured behavior.
        """
        session = conversation_session_service.get_session(self.db, original_conversation_id)
        if not session:
            return

        if behavior == "pause":
            # Pause session - store resume info
            session_context = session.context or {}
            session_context["_paused_for_redirect"] = {
                "target_session": target_conversation_id,
                "paused_at": datetime.now().isoformat()
            }
            conversation_session_service.update_session_context(
                self.db, original_conversation_id, session_context
            )

            session_update = ConversationSessionUpdate(status="paused")
            conversation_session_service.update_session(self.db, original_conversation_id, session_update)
            print(f"  → Original session paused")

        elif behavior == "close":
            # Close/resolve session
            session_context = session.context or {}
            session_context["_closed_for_redirect"] = {
                "target_session": target_conversation_id,
                "closed_at": datetime.now().isoformat()
            }
            conversation_session_service.update_session_context(
                self.db, original_conversation_id, session_context
            )

            session_update = ConversationSessionUpdate(status="resolved")
            conversation_session_service.update_session(self.db, original_conversation_id, session_update)
            print(f"  → Original session closed/resolved")

        else:  # keep_active
            # Just link sessions in context for reference
            session_context = session.context or {}
            session_context["_linked_redirect_sessions"] = session_context.get("_linked_redirect_sessions", [])
            session_context["_linked_redirect_sessions"].append({
                "target_session": target_conversation_id,
                "redirected_at": datetime.now().isoformat()
            })
            conversation_session_service.update_session_context(
                self.db, original_conversation_id, session_context
            )
            print(f"  → Original session kept active, linked to target")

    async def _execute_question_classifier_node(self, node_data: dict, context: dict, results: dict, company_id: int):
        """
        Classifies user question into predefined classes using LLM.
        Returns the class name to determine which edge to follow.
        """
        model = node_data.get("model", "groq/llama-3.1-8b-instant")
        classes = node_data.get("classes", [])  # [{name: "billing", description: "..."}, ...]
        input_variable = node_data.get("input_variable", "user_message")
        output_variable = node_data.get("output_variable", "classification")

        # Get the question to classify from context
        question = context.get(input_variable, "")

        if not question:
            print(f"✗ Question classifier: No input found in '{input_variable}'")
            return {"output": "default", "classification": None}

        if not classes:
            print(f"✗ Question classifier: No classes configured")
            return {"output": "default", "classification": None}

        # Build classification prompt
        class_names = [cls["name"] for cls in classes]
        class_descriptions = "\n".join([
            f"- {cls['name']}: {cls.get('description', 'No description provided')}"
            for cls in classes
        ])

        prompt = f"""Classify the following question into exactly one of these categories:

{class_descriptions}

Question: "{question}"

Instructions:
- Respond with ONLY the category name, nothing else
- Choose the most relevant category
- If no category fits well, respond with "default"

Category:"""

        print(f"✓ Question classifier: Classifying '{question[:50]}...' into classes: {class_names}")

        try:
            # Call LLM using existing llm_tool_service
            llm_response = await self.llm_tool_service.execute(
                model=model,
                system_prompt="You are a classification assistant. Your only job is to classify questions into predefined categories. Always respond with just the category name, nothing else.",
                user_prompt=prompt,
                company_id=company_id
            )

            # Extract classification from response
            if isinstance(llm_response, dict):
                classification = llm_response.get("content", "").strip()
            else:
                classification = str(llm_response).strip()

            # Normalize and match to configured class
            classification_lower = classification.lower().strip()
            class_names_lower = [cls["name"].lower() for cls in classes]

            matched_class = None
            for cls in classes:
                if cls["name"].lower() == classification_lower:
                    matched_class = cls["name"]
                    break

            if matched_class:
                print(f"✓ Question classifier: Classified as '{matched_class}'")
                context[output_variable] = matched_class
                return {"output": matched_class, "classification": matched_class}
            else:
                print(f"ℹ Question classifier: LLM returned '{classification}' which doesn't match any class, using default")
                context[output_variable] = "default"
                return {"output": "default", "classification": None}

        except Exception as e:
            print(f"✗ Question classifier error: {e}")
            return {"output": "default", "classification": None, "error": str(e)}

    async def _execute_extract_entities_node(self, node_data: dict, context: dict, results: dict, company_id: int, workflow: Workflow, conversation_id: str):
        """
        Extracts entities from text using LLM.
        If extraction fails, pauses workflow to prompt user for missing entities.
        """
        entities_config = node_data.get("entities", [])
        input_source = node_data.get("input_source", "{{context.user_message}}")
        model = node_data.get("model", "groq/llama-3.1-8b-instant")
        retry_prompt_template = node_data.get("retry_prompt_template", "I couldn't find your {entity_description}. Please provide it.")
        max_retries = node_data.get("max_retries", 2)

        if not entities_config:
            print("✗ Extract entities: No entities configured")
            return {"output": {}, "status": "complete"}

        # Check if resuming from pause (user providing missing entity)
        # First, check for stale markers - if variable_to_save doesn't match, we have stale data
        extracting_entity_name = context.get("_extracting_entity_name")
        variable_to_save = context.get("variable_to_save", "")

        is_valid_resume = (
            extracting_entity_name is not None and
            variable_to_save == extracting_entity_name
        )

        if extracting_entity_name and not is_valid_resume:
            print(f"⚠ Extract entities: Stale resume markers detected (variable_to_save='{variable_to_save}' != extracting_entity_name='{extracting_entity_name}'). Starting fresh extraction.")
            # Clear stale markers and entity values
            context.pop("_extracting_entity_name", None)
            context.pop("_missing_entities", None)
            context.pop("_extraction_attempts", None)
            for entity_config in entities_config:
                entity_name = entity_config["name"]
                context.pop(entity_name, None)

        if is_valid_resume:
            missing_entities = context.get("_missing_entities", [])
            extraction_attempts = context.get("_extraction_attempts", {})

            # Deserialize missing_entities if it's a JSON string
            if isinstance(missing_entities, str):
                try:
                    missing_entities = json.loads(missing_entities)
                except (json.JSONDecodeError, TypeError):
                    missing_entities = []

            print(f"✓ Extract entities: Resuming, user provided value for '{extracting_entity_name}'")

            # Get the user's response - try extracting_entity_name first, then fall back to user_message
            user_provided_text = context.get(extracting_entity_name, "") or context.get("user_message", "")

            is_valid = False
            validation_error = None

            if user_provided_text:
                # Find the entity config for this entity
                entity_config = next((e for e in entities_config if e["name"] == extracting_entity_name), None)

                if entity_config:
                    # Use LLM to extract just the value from user's response
                    entity_description = entity_config.get("description", extracting_entity_name)
                    entity_type = entity_config.get("type", "text")

                    extraction_prompt = f"""Extract only the {entity_description} from this message.
Return ONLY the extracted value, nothing else.

Entity to extract: {extracting_entity_name} ({entity_type})
Description: {entity_description}
Message: "{user_provided_text}"

Extracted value:"""

                    try:
                        llm_response = await self.llm_tool_service.execute(
                            model=model,
                            system_prompt="You are an entity extraction assistant. Extract only the requested value from the message. Return ONLY the value itself, nothing else.",
                            chat_history=[],
                            user_prompt=extraction_prompt,
                            tools=[],
                            knowledge_base_id=None,
                            company_id=company_id
                        )

                        if isinstance(llm_response, dict):
                            extracted_value = llm_response.get("content", "").strip()
                        else:
                            extracted_value = str(llm_response).strip() if llm_response else user_provided_text

                        # Validate extracted value based on entity type
                        if extracted_value and extracted_value.lower() not in ['null', 'none', 'n/a']:
                            # Type-based validation
                            if entity_type == "number":
                                # Check if it's a valid number
                                try:
                                    float(extracted_value)
                                    is_valid = True
                                except ValueError:
                                    validation_error = f"'{extracted_value}' is not a valid number"
                            elif entity_type == "email":
                                # Basic email validation
                                import re
                                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                                if re.match(email_pattern, extracted_value):
                                    is_valid = True
                                else:
                                    validation_error = f"'{extracted_value}' is not a valid email"
                            elif entity_type == "phone":
                                # Basic phone validation (digits, spaces, +, -, ())
                                import re
                                phone_pattern = r'^[+]?[\d\s\-()]+$'
                                if re.match(phone_pattern, extracted_value) and len(extracted_value.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')) >= 10:
                                    is_valid = True
                                else:
                                    validation_error = f"'{extracted_value}' is not a valid phone number"
                            else:
                                # For text and other types, any non-empty value is valid
                                is_valid = True

                        # Save if valid, otherwise mark as still missing
                        if is_valid:
                            context[extracting_entity_name] = extracted_value
                            # Save to memory for persistence
                            memory_service.set_memory(
                                self.db,
                                MemoryCreate(key=extracting_entity_name, value=extracted_value),
                                agent_id=self._executing_agent_id,
                                session_id=conversation_id
                            )
                            print(f"✓ Extracted and validated '{extracted_value}' for {extracting_entity_name} (type: {entity_type})")
                        else:
                            # Validation failed - don't save, keep in missing list
                            print(f"✗ Validation failed for {extracting_entity_name}: {validation_error}")

                    except Exception as e:
                        print(f"✗ LLM extraction failed for {extracting_entity_name}: {e}, using raw input")
                        context[extracting_entity_name] = user_provided_text
                        is_valid = True  # Exception path - accept raw input
                else:
                    # No config found, use raw input
                    context[extracting_entity_name] = user_provided_text
                    is_valid = True
            else:
                print(f"⚠ Warning: No user input found for '{extracting_entity_name}', using empty value")
                context[extracting_entity_name] = ""
                is_valid = True

            # Remove from missing list only if validation passed
            if is_valid and extracting_entity_name in missing_entities:
                missing_entities.remove(extracting_entity_name)

            # Clear the resumption markers
            del context["_extracting_entity_name"]

            # Check if there are more missing entities
            if missing_entities:
                # Ask for the next missing entity
                next_entity_name = missing_entities[0]
                entity_config = next((e for e in entities_config if e["name"] == next_entity_name), None)

                if entity_config:
                    entity_description = entity_config.get("description", next_entity_name)
                    entity_type_next = entity_config.get("type", "text")
                    prompt_text = retry_prompt_template.replace("{entity_description}", entity_description).replace("{entity_name}", next_entity_name)

                    # If this is the same entity that just failed validation, add the error message
                    if next_entity_name == extracting_entity_name and not is_valid and validation_error:
                        prompt_text = f"{validation_error}. {prompt_text}"

                    # Update context for next iteration
                    context["_extracting_entity_name"] = next_entity_name
                    context["_missing_entities"] = missing_entities
                    context["_extraction_attempts"] = extraction_attempts
                    context["variable_to_save"] = next_entity_name  # Standard pause/resume mechanism expects this

                    # Save to memory (for debugging and backup)
                    memory_service.set_memory(
                        self.db,
                        MemoryCreate(key="variable_to_save", value=next_entity_name),
                        agent_id=self._executing_agent_id,
                        session_id=conversation_id
                    )
                    memory_service.set_memory(
                        self.db,
                        MemoryCreate(key="_extracting_entity_name", value=next_entity_name),
                        agent_id=self._executing_agent_id,
                        session_id=conversation_id
                    )
                    memory_service.set_memory(
                        self.db,
                        MemoryCreate(key="_missing_entities", value=json.dumps(missing_entities)),
                        agent_id=self._executing_agent_id,
                        session_id=conversation_id
                    )

                    print(f"ℹ Extract entities: Still missing {len(missing_entities)} entities, asking for '{next_entity_name}'")

                    return {
                        "status": "paused_for_prompt",
                        "prompt": {
                            "text": prompt_text,
                            "options": []
                        },
                        "output_variable": next_entity_name,
                        "re_execute_node": True  # Re-execute this node to continue collection
                    }

            # All entities collected, clean up context
            context.pop("_missing_entities", None)
            context.pop("_extraction_attempts", None)

            # Gather all extracted entities
            extracted_entities = {}
            for entity_config in entities_config:
                entity_name = entity_config["name"]
                extracted_entities[entity_name] = context.get(entity_name)

            print(f"✓ Extract entities: All entities collected: {list(extracted_entities.keys())}")
            return {"output": extracted_entities, "status": "complete"}

        # First time execution - attempt LLM extraction
        # Resolve input source placeholder
        input_text = self._resolve_placeholders(input_source, context, results)

        if not input_text:
            print(f"✗ Extract entities: No input text found from source '{input_source}'")
            input_text = ""

        # Build LLM extraction prompt
        entities_list = "\n".join([
            f"- {entity['name']}: {entity.get('description', 'No description')} (type: {entity.get('type', 'text')})"
            for entity in entities_config
        ])

        prompt = f"""Extract the following entities from the message.
Return a JSON object with entity names as keys.
If an entity is not found, use null as the value.

Entities to extract:
{entities_list}

Message: "{input_text}"

Return only valid JSON, nothing else:"""

        print(f"✓ Extract entities: Attempting to extract {len(entities_config)} entities from: '{input_text[:100]}...'")

        try:
            # Call LLM
            llm_response = await self.llm_tool_service.execute(
                model=model,
                system_prompt="You are an entity extraction assistant. Extract the requested entities from the message and return them in valid JSON format. Always return a JSON object with entity names as keys. Use null for entities that cannot be found.",
                chat_history=[],
                user_prompt=prompt,
                tools=[],  # Empty list instead of None
                knowledge_base_id=None,
                company_id=company_id
            )

            # Parse JSON response
            if isinstance(llm_response, dict):
                response_text = llm_response.get("content", "")
            else:
                response_text = str(llm_response) if llm_response else ""

            if not response_text:
                raise ValueError("LLM returned empty response")

            # Extract JSON from response (handle markdown code blocks)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()

            extracted_entities = json.loads(response_text)

            # Validate that extracted_entities is a dict
            if not isinstance(extracted_entities, dict):
                raise ValueError(f"LLM returned non-dict response: {type(extracted_entities)}")

            print(f"✓ Extract entities: LLM returned: {extracted_entities}")

        except Exception as e:
            print(f"✗ Extract entities: LLM extraction failed: {e}")
            import traceback
            traceback.print_exc()
            # Treat all as missing
            extracted_entities = {entity["name"]: None for entity in (entities_config or [])}

        # Save extracted entities to context and check which are missing
        missing_entities = []
        for entity_config in entities_config:
            entity_name = entity_config["name"]
            entity_value = extracted_entities.get(entity_name)
            is_required = entity_config.get("required", True)

            if entity_value is not None and entity_value != "":
                # Validate extracted value based on entity type
                entity_type = entity_config.get("type", "text")
                is_valid = False
                validation_error = None

                if entity_type == "number":
                    try:
                        float(entity_value)
                        is_valid = True
                    except ValueError:
                        validation_error = f"'{entity_value}' is not a valid number"
                elif entity_type == "email":
                    import re
                    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    if re.match(email_pattern, str(entity_value)):
                        is_valid = True
                    else:
                        validation_error = f"'{entity_value}' is not a valid email"
                elif entity_type == "phone":
                    import re
                    phone_pattern = r'^[+]?[\d\s\-()]+$'
                    entity_value_str = str(entity_value)
                    if re.match(phone_pattern, entity_value_str) and len(entity_value_str.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')) >= 10:
                        is_valid = True
                    else:
                        validation_error = f"'{entity_value}' is not a valid phone number"
                else:
                    # For text and other types, any non-empty value is valid
                    is_valid = True

                if is_valid:
                    # Successfully extracted and validated
                    context[entity_name] = entity_value
                    # Also save to memory to ensure persistence across pauses
                    memory_service.set_memory(
                        self.db,
                        MemoryCreate(key=entity_name, value=entity_value),
                        agent_id=self._executing_agent_id,
                        session_id=conversation_id
                    )
                    print(f"✓ Entity '{entity_name}' extracted and validated: {entity_value} (type: {entity_type})")
                else:
                    # Validation failed - treat as missing
                    if is_required:
                        missing_entities.append(entity_name)
                        print(f"✗ Entity '{entity_name}' extracted but validation failed: {validation_error}")
                    else:
                        context[entity_name] = None
                        print(f"ℹ Entity '{entity_name}' validation failed but optional: {validation_error}")
            elif is_required:
                # Missing and required
                missing_entities.append(entity_name)
                print(f"✗ Entity '{entity_name}' missing and required")
            else:
                # Missing but optional
                context[entity_name] = None
                print(f"ℹ Entity '{entity_name}' missing but optional, setting to null")

        # If all required entities extracted, return success
        if not missing_entities:
            print(f"✓ Extract entities: All required entities extracted successfully")
            return {"output": extracted_entities, "status": "complete"}

        # Some entities missing - pause and ask for first one
        first_missing = missing_entities[0]
        entity_config = next((e for e in entities_config if e["name"] == first_missing), None)

        if entity_config:
            entity_description = entity_config.get("description", first_missing)
            prompt_text = retry_prompt_template.replace("{entity_description}", entity_description).replace("{entity_name}", first_missing)
        else:
            prompt_text = f"Please provide {first_missing}"

        # Save state to context for resume
        context["_missing_entities"] = missing_entities
        context["_extracting_entity_name"] = first_missing
        context["_extraction_attempts"] = {entity: 0 for entity in missing_entities}
        context["variable_to_save"] = first_missing  # Standard pause/resume mechanism expects this

        # Save to memory (for debugging and backup)
        memory_service.set_memory(
            self.db,
            MemoryCreate(key="variable_to_save", value=first_missing),
            agent_id=self._executing_agent_id,
            session_id=conversation_id
        )
        memory_service.set_memory(
            self.db,
            MemoryCreate(key="_extracting_entity_name", value=first_missing),
            agent_id=self._executing_agent_id,
            session_id=conversation_id
        )
        memory_service.set_memory(
            self.db,
            MemoryCreate(key="_missing_entities", value=json.dumps(missing_entities)),
            agent_id=self._executing_agent_id,
            session_id=conversation_id
        )

        print(f"ℹ Extract entities: {len(missing_entities)} entities missing, asking for '{first_missing}'")

        return {
            "status": "paused_for_prompt",
            "prompt": {
                "text": prompt_text,
                "options": []
            },
            "output_variable": first_missing,
            "re_execute_node": True  # Re-execute this node to continue collection
        }

    # ============================================================
    # SUBWORKFLOW EXECUTION METHODS
    # ============================================================

    def _get_execution_chain(self, conversation_id: str) -> list:
        """Get list of workflow IDs currently in the execution chain (for circular reference detection)."""
        session = conversation_session_service.get_session(self.db, conversation_id)
        if not session or not session.subworkflow_stack:
            return []
        return [entry["workflow_id"] for entry in session.subworkflow_stack]

    def _detect_circular_reference(self, workflow_id: int, subworkflow_id: int, company_id: int, visited: set = None) -> bool:
        """
        Statically detect if calling subworkflow_id would create a cycle.
        Used for validation at save time and runtime.
        """
        if visited is None:
            visited = set()

        if subworkflow_id == workflow_id:
            return True
        if subworkflow_id in visited:
            return False  # Already checked this path

        visited.add(subworkflow_id)

        # Get the subworkflow and check its subworkflow nodes
        subworkflow = workflow_service.get_workflow(self.db, subworkflow_id, company_id)
        if not subworkflow or not subworkflow.visual_steps:
            return False

        visual_steps = subworkflow.visual_steps
        if isinstance(visual_steps, str):
            try:
                visual_steps = json.loads(visual_steps)
            except json.JSONDecodeError:
                return False

        nodes = visual_steps.get("nodes", [])
        for node in nodes:
            if node.get("type") == "subworkflow":
                nested_subworkflow_id = node.get("data", {}).get("subworkflow_id")
                if nested_subworkflow_id and self._detect_circular_reference(
                    workflow_id, nested_subworkflow_id, company_id, visited
                ):
                    return True

        return False

    async def _execute_subworkflow_node(
        self,
        node_data: dict,
        context: dict,
        results: dict,
        company_id: int,
        workflow: Workflow,
        conversation_id: str,
        current_depth: int = 0
    ):
        """
        Execute a subworkflow node by calling another workflow.

        Node data structure:
        {
            "subworkflow_id": int,      # ID of workflow to call
            "output_variable": str      # Variable name to store subworkflow results
        }
        """
        subworkflow_id = node_data.get("subworkflow_id")
        output_variable = node_data.get("output_variable", "subworkflow_result")

        if not subworkflow_id:
            return {"error": "No subworkflow selected. Please configure the subworkflow node."}

        # Depth check
        if current_depth >= settings.MAX_SUBWORKFLOW_DEPTH:
            return {
                "error": f"Maximum subworkflow depth ({settings.MAX_SUBWORKFLOW_DEPTH}) exceeded. Consider simplifying your workflow structure."
            }

        # Circular reference check at runtime
        execution_chain = self._get_execution_chain(conversation_id)
        if subworkflow_id in execution_chain:
            return {
                "error": f"Circular reference detected: workflow {subworkflow_id} is already in execution chain"
            }

        # Static circular reference check
        if self._detect_circular_reference(workflow.id, subworkflow_id, company_id):
            return {
                "error": f"Circular reference detected: subworkflow {subworkflow_id} would create a cycle"
            }

        # Fetch subworkflow
        subworkflow = workflow_service.get_workflow(self.db, subworkflow_id, company_id)
        if not subworkflow:
            return {"error": f"Subworkflow with ID {subworkflow_id} not found"}

        print(f"✓ Subworkflow node: Executing subworkflow '{subworkflow.name}' (ID: {subworkflow_id}) at depth {current_depth + 1}")

        # Return execution directive - actual execution happens in execute_workflow
        return {
            "status": "execute_subworkflow",
            "subworkflow_id": subworkflow_id,
            "subworkflow_name": subworkflow.name,
            "output_variable": output_variable,
            "depth": current_depth + 1
        }

    async def execute_workflow(self, user_message: str, company_id: int, workflow_id: int = None, workflow: Workflow = None, conversation_id: str = None, attachments: list = None, option_key: str = None, agent_id: int = None):
        if workflow_id:
            workflow_obj = workflow_service.get_workflow(self.db, workflow_id, company_id)
        elif workflow:
            workflow_obj = workflow
        else:
            return {"error": "Either workflow_id or workflow object must be provided."}

        if not workflow_obj:
            return {"error": f"Workflow not found."}

        # Get the executing agent - either from parameter, workflow's agents, or None
        executing_agent = None
        executing_agent_id = agent_id
        if agent_id:
            from app.services import agent_service
            executing_agent = agent_service.get_agent(self.db, agent_id, workflow_obj.company_id)
        elif hasattr(workflow_obj, 'agents') and workflow_obj.agents:
            executing_agent = workflow_obj.agents[0]
            executing_agent_id = executing_agent.id

        # Store executing agent for use in node execution
        self._executing_agent = executing_agent
        self._executing_agent_id = executing_agent_id

        print(f"DEBUG: Fetched workflow: {workflow_obj.name} (ID: {workflow_obj.id})")
        if executing_agent:
            print(f"DEBUG: Executing agent: {executing_agent.name} (ID: {executing_agent.id})")
        else:
            print("DEBUG: No executing agent set for workflow.")
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        session = conversation_session_service.get_or_create_session(
            self.db, conversation_id, workflow_obj.id, contact_id=1, channel="test", company_id=workflow_obj.company_id
        )

        # Load all memories for this session into the context
        context = {memory.key: memory.value for memory in memory_service.get_all_memories(self.db, agent_id=executing_agent_id, session_id=conversation_id)} if executing_agent_id else {}

        # Also merge session.context which contains validation state and other transient data
        # Session context takes precedence for keys like pending_listen_validation_mode, pending_question_text, etc.
        if session.context:
            session_context = session.context if isinstance(session.context, dict) else {}
            context.update(session_context)

        # ============================================================
        # WORKFLOW INTENT DETECTION
        # ============================================================
        # Check if this workflow has intent detection enabled
        if self.workflow_intent_service.workflow_has_intents_enabled(workflow_obj):
            print(f"DEBUG: Intent detection enabled for workflow '{workflow_obj.name}'")

            intent_match = await self.workflow_intent_service.detect_intent_for_workflow(
                message=user_message,
                workflow=workflow_obj,
                conversation_id=conversation_id,
                company_id=company_id
            )

            if intent_match:
                intent_dict, confidence, entities, matched_method = intent_match
                print(f"✓ Workflow intent detected: {intent_dict.get('name')} (confidence: {confidence:.2f}, method: {matched_method})")

                # Add detected intent information to context
                context['detected_intent'] = intent_dict.get('name')
                context['intent_confidence'] = confidence
                context['intent_matched_method'] = matched_method

                # Merge extracted entities into context
                if entities:
                    print(f"✓ Extracted entities: {entities}")
                    context.update(entities)

                    # Save entities to memory for persistence
                    if self._executing_agent_id:
                        for entity_name, entity_value in entities.items():
                            memory_service.set_memory(
                                self.db,
                                MemoryCreate(key=entity_name, value=entity_value),
                                agent_id=self._executing_agent_id,
                                session_id=conversation_id
                            )

                # Check if confidence meets auto-trigger threshold
                if not self.workflow_intent_service.should_auto_trigger(workflow_obj, confidence):
                    min_confidence = workflow_obj.intent_config.get("min_confidence", 0.7)
                    print(f"ℹ Intent confidence {confidence:.2f} below threshold {min_confidence}, workflow may not proceed")
                    # Continue execution anyway since workflow was explicitly called
            else:
                print(f"✗ No intent matched for workflow '{workflow_obj.name}'")

        results = {}

        # Ensure visual_steps is a dictionary
        visual_steps_data = workflow_obj.visual_steps

        # If this is a parent workflow with no visual_steps, try to use the active version instead
        if visual_steps_data is None and hasattr(workflow_obj, 'versions') and workflow_obj.versions:
            active_version = next((v for v in workflow_obj.versions if v.is_active), None)
            if active_version and active_version.visual_steps:
                print(f"DEBUG: Using active version {active_version.id} (v{active_version.version}) instead of parent {workflow_obj.id}")
                visual_steps_data = active_version.visual_steps

        # Handle None or empty visual_steps
        if visual_steps_data is None:
            print(f"WARNING: Workflow {workflow_obj.id} has no visual_steps defined")
            return {"status": "error", "response": "Workflow configuration is incomplete. Please contact support."}

        if isinstance(visual_steps_data, str):
            try:
                visual_steps_data = json.loads(visual_steps_data)
            except json.JSONDecodeError:
                return {"status": "error", "response": "Failed to parse workflow visual steps."}

        # Validate that visual_steps_data has required structure
        if not isinstance(visual_steps_data, dict):
            print(f"WARNING: Workflow {workflow_obj.id} visual_steps is not a dict: {type(visual_steps_data)}")
            return {"status": "error", "response": "Workflow configuration is invalid. Please contact support."}

        graph_engine = GraphExecutionEngine(visual_steps_data)
        
        print(f"DEBUG: Workflow resumed with user_message: '{user_message}'")
        # Check if workflow is paused (indicated by next_step_id being set)
        should_resume = False
        if session.next_step_id:
            current_node_id = session.next_step_id

            # Check if the node exists in the current workflow version
            # This can fail if the workflow version was changed while session was paused
            if current_node_id not in graph_engine.nodes:
                print(f"WARNING: Paused node '{current_node_id}' not found in current workflow version. Restarting workflow.")
                # Reset session state and start fresh
                session_update = ConversationSessionUpdate(
                    next_step_id=None,
                    context={}
                )
                conversation_session_service.update_session(self.db, conversation_id, session_update)
                # Clear memories for clean restart
                memory_service.delete_all_memories(self.db, agent_id=self._executing_agent_id, session_id=conversation_id)
                # Start from beginning
                current_node_id = graph_engine.find_start_node()
                context = {"initial_user_message": user_message, "user_attachments": attachments or []}
                memory_service.set_memory(self.db, MemoryCreate(key="initial_user_message", value=user_message), agent_id=self._executing_agent_id, session_id=conversation_id)
            else:
                should_resume = True
                print(f"DEBUG: Resuming from paused state. Context from memory: {context}")
                # Add attachments to context when resuming
                context["user_attachments"] = attachments or []

                # Validate prompt input using multi-stage validation
                pending_allow_text = context.get("pending_allow_text_input", True)
                pending_options = context.get("pending_prompt_options", [])
                pending_validation_mode = context.get("pending_validation_mode", "exact")
                pending_validation_llm_provider = context.get("pending_validation_llm_provider", "groq")
                pending_validation_llm_model = context.get("pending_validation_llm_model")  # None uses provider default
                pending_prompt_text = context.get("pending_prompt_text", "")
                retry_count = context.get("_validation_retry_count", 0)
                max_retries = context.get("_validation_max_retries", 3)

                # Voice option extraction: Use OpenAI structured output to extract chosen option
                # This runs before multi-stage validation for more accurate voice input handling
                if pending_options and not option_key and user_message:
                    try:
                        from app.services.voice_option_extraction_service import extract_chosen_option
                        
                        extraction_result = await extract_chosen_option(
                            db=self.db,
                            company_id=company_id,
                            user_input=user_message,
                            prompt_text=pending_prompt_text,
                            options=pending_options
                        )
                        
                        if extraction_result and extraction_result.get("chosen_option"):
                            option_key = extraction_result["chosen_option"]
                            print(f"DEBUG: OpenAI voice extraction - option_key set to: {option_key} (confidence: {extraction_result.get('confidence', 0)})")
                    except Exception as e:
                        print(f"DEBUG: Voice option extraction failed (will use fallback validation): {e}")

                if pending_options:
                    input_value = option_key if option_key else user_message
                    validation_mode = ValidationMode(pending_validation_mode) if pending_validation_mode != "none" else ValidationMode.EXACT

                    # Perform validation based on mode
                    # pending_options now contains full option dicts with key and value
                    validation_result = await self.input_validation_service.validate_prompt_response(
                        db=self.db,
                        company_id=company_id,
                        user_input=input_value,
                        options=pending_options,
                        allow_text_input=pending_allow_text,
                        prompt_context=pending_prompt_text,
                        validation_mode=validation_mode,
                        llm_provider=pending_validation_llm_provider,
                        llm_model=pending_validation_llm_model
                    )

                    if validation_result.is_valid:
                        # Valid input - use matched option key if available
                        if validation_result.matched_option_key:
                            # Update the value to save with the matched key
                            if option_key:
                                option_key = validation_result.matched_option_key
                            else:
                                user_message = validation_result.matched_option_key
                        print(f"DEBUG: Validation passed - matched: {validation_result.matched_option_key}, confidence: {validation_result.confidence}")
                        self._clear_validation_state(context)
                    else:
                        # Validation failed - check retry count
                        retry_count += 1
                        print(f"DEBUG: Validation failed ({retry_count}/{max_retries}): {validation_result.reason}")

                        if retry_count >= max_retries:
                            # Max retries exceeded - clear state and continue with original input
                            print(f"DEBUG: Max retries exceeded, continuing with original input")
                            self._clear_validation_state(context)
                        else:
                            # Re-ask with hint
                            context["_validation_retry_count"] = retry_count
                            hint_text = validation_result.hint_message or "Please select one of the options below:"
                            reask_prompt = f"{hint_text}"
                            if pending_prompt_text and hint_text != pending_prompt_text:
                                reask_prompt = f"{hint_text}\n\n{pending_prompt_text}"

                            # Update session context with retry count
                            session_update = ConversationSessionUpdate(
                                context=context,
                                status='active'
                            )
                            conversation_session_service.update_session(self.db, conversation_id, session_update)

                            return {
                                "status": "paused_for_prompt",
                                "prompt": {
                                    "text": reask_prompt,
                                    "options": pending_options,  # Already list of {key, value} dicts
                                    "allow_text_input": pending_allow_text
                                },
                                "validation_retry": retry_count
                            }
                else:
                    # No pending options - clear validation state
                    self._clear_validation_state(context)

                # Validate listen node response if validation was enabled
                # Store extracted value from LLM validation (if any) for use when saving
                listen_extracted_value = None
                pending_listen_mode = context.get("pending_listen_validation_mode")
                if pending_listen_mode and pending_listen_mode != "none":
                    question_text = context.get("pending_question_text", "")
                    expected_input_type = context.get("expected_input_type", "any")
                    listen_retry_count = context.get("_listen_validation_retry_count", 0)
                    listen_max_retries = context.get("_listen_validation_max_retries", 3)
                    listen_llm_provider = context.get("pending_listen_validation_llm_provider", "groq")
                    listen_llm_model = context.get("pending_listen_validation_llm_model")  # None uses provider default

                    # If no explicit question, try to use previous message as context
                    if not question_text:
                        question_text = context.get("_last_agent_message", "")

                    validation_mode = ValidationMode(pending_listen_mode)
                    validation_result = await self.input_validation_service.validate_listen_response(
                        db=self.db,
                        company_id=company_id,
                        user_input=user_message,
                        question_text=question_text,
                        expected_input_type=expected_input_type,
                        validation_mode=validation_mode,
                        llm_provider=listen_llm_provider,
                        llm_model=listen_llm_model
                    )

                    # Store extracted value for later use (LLM mode extracts entities from responses)
                    if validation_result.extracted_value:
                        listen_extracted_value = validation_result.extracted_value
                        print(f"DEBUG: LLM extracted value: '{listen_extracted_value}' from '{user_message}'")

                    if not validation_result.is_valid:
                        listen_retry_count += 1
                        print(f"DEBUG: Listen validation failed ({listen_retry_count}/{listen_max_retries}): {validation_result.reason}")

                        if listen_retry_count >= listen_max_retries:
                            # Max retries - continue anyway
                            print(f"DEBUG: Listen max retries exceeded, continuing")
                            self._clear_listen_validation_state(context)
                        else:
                            # Re-ask with hint message
                            context["_listen_validation_retry_count"] = listen_retry_count
                            hint_text = validation_result.hint_message or f"Please answer: {question_text}"

                            # Broadcast the hint message to the user
                            from app.api.v1.endpoints.websocket_conversations import manager as ws_manager
                            from app.services import chat_service
                            from app.schemas import chat_message as schemas_chat_message

                            hint_message = schemas_chat_message.ChatMessageCreate(message=hint_text, message_type="message")
                            db_hint_message = chat_service.create_chat_message(
                                self.db, hint_message,
                                self._executing_agent_id, conversation_id,
                                workflow_obj.company_id, "agent",
                                assignee_id=None
                            )
                            asyncio.create_task(ws_manager.broadcast_to_session(
                                str(conversation_id),
                                json.dumps({
                                    "message": hint_text,
                                    "message_type": "message",
                                    "sender": "agent",
                                    "message_id": db_hint_message.id,
                                    "timestamp": db_hint_message.timestamp.isoformat()
                                }),
                                "agent"
                            ))

                            # Update session to stay paused
                            session_update = ConversationSessionUpdate(
                                context=context,
                                status='active'
                            )
                            conversation_session_service.update_session(self.db, conversation_id, session_update)

                            return {
                                "status": "paused_for_input",
                                "expected_input_type": expected_input_type,
                                "question_text": question_text,
                                "validation_hint": hint_text,
                                "retry_count": listen_retry_count
                            }
                    else:
                        self._clear_listen_validation_state(context)

                # The variable to save was stored in the context before pausing.
                variable_to_save = context.get("variable_to_save")
                print(f"DEBUG: Retrieved variable_to_save: '{variable_to_save}'")
                if variable_to_save:
                    # Determine what value to save to the workflow variable
                    # Priority: option_key > LLM extracted value > raw user_message
                    if option_key:
                        value_to_save = option_key
                    elif listen_extracted_value:
                        value_to_save = listen_extracted_value
                        print(f"DEBUG: Using LLM extracted value: '{value_to_save}'")
                    else:
                        value_to_save = user_message
                    print(f"DEBUG: Will save to variable '{variable_to_save}': option_key={option_key}, extracted={listen_extracted_value}, user_message={user_message}, value_to_save={value_to_save}")

                    # Check if the incoming message is a JSON string (from a form submission)
                    try:
                        form_data = json.loads(value_to_save)
                        context[variable_to_save] = form_data
                    except (json.JSONDecodeError, TypeError):
                        # It's a plain text response (e.g., from a prompt)
                        # If there are attachments, save them along with the message
                        if attachments:
                            context[variable_to_save] = {
                                "text": value_to_save,
                                "attachments": attachments
                            }
                            print(f"DEBUG: Saved message with {len(attachments)} attachment(s) to '{variable_to_save}'")
                        else:
                            # Check if this is a location input that needs geocoding
                            expected_input_type = context.get("expected_input_type")
                            if expected_input_type == "location" and isinstance(value_to_save, str) and value_to_save.strip():
                                # User typed a text location (Instagram, etc.) - geocode it
                                print(f"DEBUG: Geocoding text location: '{value_to_save}'")
                                geocoded = await geocoding_service.forward_geocode(value_to_save)
                                if geocoded and geocoded.get("latitude") and geocoded.get("longitude"):
                                    # Format to match WhatsApp/WebSocket location format
                                    lat = geocoded["latitude"]
                                    lng = geocoded["longitude"]
                                    context[variable_to_save] = {
                                        "text": f"📍 Location ({lat:.4f}, {lng:.4f})",
                                        "attachments": [{
                                            "file_name": "location",
                                            "file_type": "application/geo+json",
                                            "file_size": 46,
                                            "location": {
                                                "latitude": lat,
                                                "longitude": lng
                                            }
                                        }],
                                        "display_name": geocoded.get("display_name", value_to_save),
                                        "original_input": value_to_save
                                    }
                                    print(f"DEBUG: Geocoded location: lat={lat}, lng={lng}")
                                else:
                                    # If geocoding fails, save as text with empty location
                                    context[variable_to_save] = {
                                        "text": value_to_save,
                                        "attachments": [],
                                        "display_name": value_to_save,
                                        "original_input": value_to_save,
                                        "error": "Could not geocode location"
                                    }
                            else:
                                context[variable_to_save] = value_to_save
                        # Clear expected_input_type after processing
                        context.pop("expected_input_type", None)
                    print(f"DEBUG: Context after updating with user message: {context}")
                    # Save the updated context back to memory
                    memory_service.set_memory(self.db, MemoryCreate(key=variable_to_save, value=context[variable_to_save]), agent_id=self._executing_agent_id, session_id=conversation_id)
        else:
            current_node_id = graph_engine.find_start_node()
            # For the very first message in a workflow
            context["initial_user_message"] = user_message
            context["user_attachments"] = attachments or []
            memory_service.set_memory(self.db, MemoryCreate(key="initial_user_message", value=user_message), agent_id=self._executing_agent_id, session_id=conversation_id)

        last_executed_node_id = None
        response_messages = []  # Collect all response node outputs
        while current_node_id:
            node = graph_engine.nodes[current_node_id]
            node_type = node.get("type")
            node_data = node.get("data", {})

            result = None
            if node_type == "start":
                initial_input_variable = node_data.get("initial_input_variable", "user_message")
                context[initial_input_variable] = user_message
                result = {"output": "Start node processed"} # Indicate success, no real output
            elif node_type == "tool":
                # Support multiple keys for backwards compatibility: tool_name, tool, name
                tool_name = node_data.get("tool_name") or node_data.get("tool") or node_data.get("name")
                if not tool_name:
                    result = {"error": f"Tool node '{current_node_id}' has no tool configured. Please select a tool in the properties panel."}
                else:
                    raw_params = node_data.get("parameters", {}) or node_data.get("params", {})
                    resolved_params = {k: self._resolve_placeholders(v, context, results) for k, v in raw_params.items()}
                    result = await self._execute_tool(tool_name, resolved_params, company_id=workflow_obj.company_id, session_id=conversation_id)

            elif node_type == "http_request":
                result = await self._execute_http_request_node(node_data, context, results)

            elif node_type == "llm":
                result = await self._execute_llm_node(node_data, context, results, company_id=workflow_obj.company_id, workflow=workflow_obj, conversation_id=conversation_id)

            elif node_type == "data_manipulation":
                result = await self._execute_data_manipulation_node(node_data, context, results)

            elif node_type == "code":
                result = await self._execute_code_node(node_data, context, results)

            elif node_type == "knowledge":
                result = await self._execute_knowledge_retrieval_node(node_data, context, results, company_id=workflow_obj.company_id, workflow=workflow_obj)

            elif node_type == "condition":
                result = self._execute_conditional_node(node_data, context, results)

            elif node_type == "listen":
                params = node_data.get("params", {})
                expected_input_type = params.get("expected_input_type", "any")
                question_text = params.get("question_text", "")
                validation_mode = params.get("validation_mode", "none")
                validation_llm_provider = params.get("validation_llm_provider", "groq")
                validation_llm_model = params.get("validation_llm_model")  # None uses provider default
                max_retries = params.get("max_retries", 3)

                # Resolve placeholders in question text
                if question_text:
                    question_text = self._resolve_placeholders(question_text, context, results)

                # Store validation config in context for when user responds
                if validation_mode and validation_mode != "none":
                    context["pending_listen_validation_mode"] = validation_mode
                    context["pending_listen_validation_llm_provider"] = validation_llm_provider
                    context["pending_listen_validation_llm_model"] = validation_llm_model
                    context["pending_question_text"] = question_text
                    context["_listen_validation_max_retries"] = max_retries
                    context["_listen_validation_retry_count"] = 0

                # Store expected input type for resume handling
                context["expected_input_type"] = expected_input_type

                result = {
                    "status": "paused_for_input",
                    "expected_input_type": expected_input_type,
                    "question_text": question_text if question_text else None
                }

            elif node_type == "prompt":
                params = node_data.get("params", {})
                options_mode = params.get("options_mode", "manual")
                options_list = []

                if options_mode == "variable":
                    # Resolve variable reference
                    options_variable = params.get("options_variable", "")
                    if options_variable:
                        resolved_options = self._resolve_placeholders(options_variable, context, results)
                        # If resolved value is a string (JSON), parse it
                        if isinstance(resolved_options, str):
                            try:
                                resolved_options = json.loads(resolved_options)
                            except json.JSONDecodeError:
                                resolved_options = []
                        # Handle dictionary - convert to list of {key, value} pairs
                        if isinstance(resolved_options, dict):
                            options_list = [
                                {"key": str(k), "value": str(v)}
                                for k, v in resolved_options.items()
                            ]
                        # Ensure it's a list of key-value dicts
                        elif isinstance(resolved_options, list):
                            options_list = [
                                opt if isinstance(opt, dict) and 'key' in opt and 'value' in opt
                                else {"key": str(opt), "value": str(opt)}
                                for opt in resolved_options
                            ]
                else:
                    # Manual mode: use options array directly
                    options = params.get("options", [])
                    if isinstance(options, str):
                        # Backward compatibility: comma-separated string
                        options_list = [
                            {"key": opt.strip(), "value": opt.strip()}
                            for opt in options.split(',') if opt.strip()
                        ]
                    elif isinstance(options, list):
                        options_list = [
                            opt if isinstance(opt, dict) and 'key' in opt and 'value' in opt
                            else {"key": str(opt), "value": str(opt)}
                            for opt in options
                        ]

                allow_text_input = params.get("allow_text_input", False)
                validation_mode = params.get("validation_mode", "exact")
                validation_llm_provider = params.get("validation_llm_provider", "groq")
                validation_llm_model = params.get("validation_llm_model")  # None uses provider default
                max_retries = params.get("max_retries", 3)

                prompt_text = params.get("prompt_text", "Please provide input.")
                resolved_prompt_text = self._resolve_placeholders(prompt_text, context, results)

                # Track as last agent message for potential fallback validation context
                context["_last_agent_message"] = resolved_prompt_text

                # Store validation data in context for when user responds
                # Store full options with both key and value so LLM can see the labels
                context["pending_prompt_options"] = options_list
                context["pending_allow_text_input"] = allow_text_input
                context["pending_validation_mode"] = validation_mode
                context["pending_validation_llm_provider"] = validation_llm_provider
                context["pending_validation_llm_model"] = validation_llm_model
                context["pending_prompt_text"] = resolved_prompt_text
                context["_validation_max_retries"] = max_retries
                context["_validation_retry_count"] = 0

                result = {
                    "status": "paused_for_prompt",
                    "prompt": {
                        "text": resolved_prompt_text,
                        "options": options_list,
                        "allow_text_input": allow_text_input
                    }
                }
            
            elif node_type == "form":
                params = node_data.get("params", {})

                form_title = params.get("title", "Please fill out this form.")
                resolved_form_title = self._resolve_placeholders(form_title, context, results)

                result = {
                    "status": "paused_for_form",
                    "form": {
                        "title": resolved_form_title,
                        "fields": params.get("fields", [])
                    }
                }

            elif node_type == "response":
                output_value = node_data.get("output_value", "")
                resolved_output = self._resolve_placeholders(output_value, context, results)
                result = {"output": resolved_output}
                # Track last agent message for validation context (used when listen node has no explicit question)
                if resolved_output:
                    message_text = resolved_output.get("text", str(resolved_output)) if isinstance(resolved_output, dict) else str(resolved_output)
                    context["_last_agent_message"] = message_text
                # Broadcast intermediate response messages immediately
                if resolved_output:
                    response_messages.append(resolved_output)
                    # Check if there's a next node - if so, broadcast this as intermediate message
                    next_check = graph_engine.get_next_node(current_node_id, result)
                    if next_check:  # There's more nodes after this response
                        # Import here to avoid circular import
                        from app.api.v1.endpoints.websocket_conversations import manager as ws_manager
                        from app.services import chat_service, messaging_service, integration_service
                        from app.schemas import chat_message as schemas_chat_message

                        # Save to database and broadcast properly formatted message
                        # Handle dict output (e.g., from Listen node with attachments)
                        # Ensure message_text is always a string
                        if isinstance(resolved_output, dict):
                            message_text = resolved_output.get("text", str(resolved_output))
                        elif isinstance(resolved_output, str):
                            message_text = resolved_output
                        else:
                            message_text = str(resolved_output)
                        agent_message = schemas_chat_message.ChatMessageCreate(message=message_text, message_type="message")
                        db_agent_message = chat_service.create_chat_message(
                            self.db, agent_message,
                            self._executing_agent_id, conversation_id,
                            workflow_obj.company_id, "agent",
                            assignee_id=None
                        )
                        await ws_manager.broadcast_to_session(
                            str(conversation_id),
                            schemas_chat_message.ChatMessage.model_validate(db_agent_message).model_dump_json(),
                            "agent"
                        )

                        # Get the session to check channel type
                        current_session = conversation_session_service.get_session_by_conversation_id(
                            self.db, conversation_id, workflow_obj.company_id
                        )
                        session_channel = current_session.channel if current_session else None

                        # Define text-based channels that don't need TTS
                        text_channels = ['whatsapp', 'telegram', 'instagram', 'messenger']

                        # For text-based channels, send message via appropriate service
                        if session_channel in text_channels:
                            try:
                                if session_channel == 'whatsapp':
                                    # Get WhatsApp integration for this company
                                    whatsapp_integration = integration_service.get_integration_by_type_and_company(
                                        self.db, 'whatsapp', workflow_obj.company_id
                                    )
                                    if whatsapp_integration:
                                        # Use contact phone number if available, otherwise use conversation_id (which is the phone number for WhatsApp)
                                        recipient_phone = conversation_id
                                        if current_session.contact and current_session.contact.phone_number:
                                            recipient_phone = current_session.contact.phone_number
                                        await messaging_service.send_whatsapp_message(
                                            recipient_phone_number=recipient_phone,
                                            message_text=message_text,
                                            integration=whatsapp_integration,
                                            db=self.db
                                        )
                                        print(f"[workflow_execution] Sent intermediate response to WhatsApp: {message_text[:50]}...")
                                elif session_channel == 'telegram':
                                    # Get Telegram integration
                                    telegram_integration = integration_service.get_integration_by_type_and_company(
                                        self.db, 'telegram', workflow_obj.company_id
                                    )
                                    if telegram_integration:
                                        await messaging_service.send_telegram_message(
                                            chat_id=int(conversation_id),
                                            message_text=message_text,
                                            integration=telegram_integration
                                        )
                                        print(f"[workflow_execution] Sent intermediate response to Telegram: {message_text[:50]}...")
                                # Add other channels as needed (instagram, messenger)
                            except Exception as channel_error:
                                print(f"[workflow_execution] Error sending to {session_channel}: {channel_error}")

                        # Generate TTS only for voice-capable channels (web_chat, twilio_voice, freeswitch)
                        elif session_channel not in text_channels:
                            try:
                                from app.services import widget_settings_service, credential_service
                                from app.services.tts_service import TTSService
                                widget_settings = widget_settings_service.get_widget_settings(self.db, self._executing_agent_id)
                                if widget_settings and widget_settings.communication_mode == 'chat_and_voice':
                                    tts_provider = (self._executing_agent.tts_provider if self._executing_agent else None) or 'voice_engine'
                                    voice_id = (self._executing_agent.voice_id if self._executing_agent else None) or 'default'
                                    openai_api_key = None
                                    openai_credential = credential_service.get_credential_by_service_name(self.db, 'openai', workflow_obj.company_id)
                                    if openai_credential:
                                        try:
                                            openai_api_key = credential_service.get_decrypted_credential(self.db, openai_credential.id, workflow_obj.company_id)
                                        except Exception:
                                            pass
                                    tts_service = TTSService(openai_api_key=openai_api_key)
                                    # Use message_text (already extracted from dict if needed)
                                    audio_stream = tts_service.text_to_speech_stream(message_text, voice_id, tts_provider)
                                    async for audio_chunk in audio_stream:
                                        await ws_manager.broadcast_bytes_to_session(str(conversation_id), audio_chunk)
                                    await tts_service.close()
                                    # Send audio_end marker so frontend knows this TTS is complete
                                    await ws_manager.broadcast_to_session(
                                        str(conversation_id),
                                        json.dumps({"type": "audio_end"}),
                                        "agent"
                                    )
                                    print(f"[workflow_execution] TTS audio sent for intermediate response in session: {conversation_id}")
                            except Exception as tts_error:
                                print(f"[workflow_execution] TTS error for intermediate response: {tts_error}")

            # ============================================================
            # NEW CHAT-SPECIFIC NODES
            # ============================================================

            elif node_type == "intent_router":
                # Routes based on detected intent in context
                result = self._execute_intent_router_node(node_data, context, results)

            elif node_type == "entity_collector":
                # Collects required entities from user
                result = await self._execute_entity_collector_node(
                    node_data, context, results, workflow_obj, conversation_id
                )

            elif node_type == "check_entity":
                # Checks if entity exists in context
                result = self._execute_check_entity_node(node_data, context, results)

            elif node_type == "update_context":
                # Updates context variables
                result = self._execute_update_context_node(node_data, context, results)

            elif node_type == "tag_conversation":
                # Adds tags to conversation
                result = self._execute_tag_conversation_node(
                    node_data, context, results, conversation_id
                )

            elif node_type == "assign_to_agent":
                # Transfers conversation to human agent
                result = self._execute_assign_to_agent_node(
                    node_data, context, results, conversation_id, workflow_obj.company_id
                )

            elif node_type == "set_status":
                # Sets conversation status
                result = self._execute_set_status_node(
                    node_data, context, results, conversation_id
                )

            elif node_type == "channel_redirect":
                # Redirects conversation to another channel (WhatsApp, Telegram, etc.)
                # Pre-compute next node ID for workflow transfer capability
                redirect_next_node_id = graph_engine.get_next_node(current_node_id, {"output": "success"})
                result = await self._execute_channel_redirect_node(
                    node_data, context, results, conversation_id,
                    workflow_obj.company_id, session.contact_id,
                    workflow_id=workflow_obj.id,
                    next_node_id=redirect_next_node_id
                )

            elif node_type == "question_classifier":
                # Classifies question using LLM and routes accordingly
                result = await self._execute_question_classifier_node(
                    node_data, context, results, workflow_obj.company_id
                )

            elif node_type == "extract_entities":
                # Extracts entities from message using LLM
                result = await self._execute_extract_entities_node(
                    node_data, context, results, workflow_obj.company_id, workflow_obj, conversation_id
                )

            elif node_type == "subworkflow":
                # Execute another workflow as a subworkflow
                # Get current depth from session's subworkflow_stack
                current_depth = len(session.subworkflow_stack or [])
                result = await self._execute_subworkflow_node(
                    node_data, context, results, company_id, workflow_obj, conversation_id, current_depth
                )

            elif node_type == "foreach_loop":
                # For Each Loop - iterates over an array
                result = self._execute_foreach_loop_node(node_data, context, results, current_node_id)

            elif node_type == "while_loop":
                # While Loop - repeats while condition is true
                result = self._execute_while_loop_node(node_data, context, results, current_node_id)

            results[current_node_id] = result
            last_executed_node_id = current_node_id

            if result and result.get("status") in ["paused_for_input", "paused_for_prompt", "paused_for_form"]:
                # Check if this node wants to re-execute itself (for multi-step collection)
                # If result has 're_execute_node', save current node as next_step_id instead
                if result.get("re_execute_node"):
                    next_node_id = current_node_id  # Re-execute this node
                else:
                    next_node_id = graph_engine.get_next_node(current_node_id, result)

                # Before pausing, save the variable name that should receive the input.
                # First check if the result provides output_variable (for dynamic nodes like extract_entities)
                # Otherwise check the node's configuration data
                variable_to_save = result.get("output_variable")
                if not variable_to_save:
                    variable_to_save = node_data.get("output_variable")
                    if not variable_to_save:
                        params = node_data.get("params", {})
                        # Check both 'output_variable' and 'save_to_variable' for backward compatibility
                        variable_to_save = params.get("output_variable") or params.get("save_to_variable")
                
                print(f"DEBUG: Pausing node data: {node_data}")
                print(f"DEBUG: 'output_variable' from node data is: '{variable_to_save}'")
                if variable_to_save:
                    context["variable_to_save"] = variable_to_save
                    memory_service.set_memory(self.db, MemoryCreate(key="variable_to_save", value=variable_to_save), agent_id=self._executing_agent_id, session_id=conversation_id)

                # Store expected_input_type for use when resuming (e.g., for geocoding text locations)
                if "expected_input_type" in result:
                    context["expected_input_type"] = result["expected_input_type"]
                    memory_service.set_memory(self.db, MemoryCreate(key="expected_input_type", value=result["expected_input_type"]), agent_id=self._executing_agent_id, session_id=conversation_id)

                # Keep session status as 'active' so it remains visible in the UI
                # The presence of next_step_id indicates the workflow is paused waiting for input
                session_update = ConversationSessionUpdate(
                    next_step_id=next_node_id,
                    context=context,
                    status='active'  # Keep as active instead of paused to keep conversation visible
                )
                conversation_session_service.update_session(self.db, conversation_id, session_update)
                
                response_payload = {
                    "status": result.get("status"),
                    "conversation_id": conversation_id,
                    "next_node_id": next_node_id
                }
                if "prompt" in result:
                    response_payload["prompt"] = result["prompt"]
                if "form" in result:
                    response_payload["form"] = result["form"]
                if "expected_input_type" in result:
                    response_payload["expected_input_type"] = result["expected_input_type"]

                # Include last response message for TTS in voice mode
                if response_messages:
                    response_payload["response"] = response_messages[-1]

                return response_payload

            # Handle subworkflow execution
            if result and result.get("status") == "execute_subworkflow":
                subworkflow_id = result["subworkflow_id"]
                output_variable = result["output_variable"]
                depth = result["depth"]

                # Push current state to subworkflow stack
                subworkflow_stack = list(session.subworkflow_stack or [])
                subworkflow_entry = {
                    "workflow_id": subworkflow_id,
                    "parent_node_id": current_node_id,
                    "parent_workflow_id": workflow_obj.id,
                    "parent_next_step_id": graph_engine.get_next_node(current_node_id, {"output": "subworkflow_complete"}),
                    "output_variable": output_variable,
                    "depth": depth
                }
                subworkflow_stack.append(subworkflow_entry)

                # Update session with stack and switch to subworkflow
                session_update = ConversationSessionUpdate(
                    subworkflow_stack=subworkflow_stack,
                    workflow_id=subworkflow_id,
                    next_step_id=None,  # Start from beginning of subworkflow
                    context=context
                )
                conversation_session_service.update_session(self.db, conversation_id, session_update)
                self.db.refresh(session)

                print(f"✓ Subworkflow: Pushed to stack, entering subworkflow {subworkflow_id} at depth {depth}")

                # Recursively execute subworkflow
                return await self.execute_workflow(
                    user_message=user_message,
                    company_id=company_id,
                    workflow_id=subworkflow_id,
                    conversation_id=conversation_id,
                    attachments=attachments
                )

            if result and "error" in result:
                # The get_next_node method will handle routing to the error path if it exists
                pass

            # Handle workflow transfer - stop execution on original channel
            if result and result.get("stop_execution"):
                print(f"✓ Workflow transferred to another channel, stopping execution on original")
                # Clear workflow state on original session since workflow transferred
                session_update = ConversationSessionUpdate(
                    workflow_id=None,
                    next_step_id=None,
                    context=context,
                    status='active'
                )
                conversation_session_service.update_session(self.db, conversation_id, session_update)

                return {
                    "status": "workflow_transferred",
                    "conversation_id": conversation_id,
                    "output": result.get("output", "Workflow transferred to another channel"),
                    "response": response_messages[-1] if response_messages else result.get("output")
                }

            print(f"DEBUG: About to call get_next_node for node '{current_node_id}' with result: {result}")
            current_node_id = graph_engine.get_next_node(current_node_id, result)
            print(f"DEBUG: get_next_node returned: {current_node_id}")

        # Get the final output before checking for subworkflow completion
        if response_messages:
            final_output = response_messages[-1]
        else:
            final_output = results.get(last_executed_node_id, {}).get("output", "Workflow completed.")

        # ============================================================
        # SUBWORKFLOW COMPLETION - Check if we need to return to parent
        # ============================================================
        subworkflow_stack = list(session.subworkflow_stack or [])
        if subworkflow_stack:
            # This workflow was a subworkflow - pop stack and continue parent
            completed_entry = subworkflow_stack.pop()
            output_variable = completed_entry["output_variable"]
            parent_workflow_id = completed_entry["parent_workflow_id"]
            parent_next_step_id = completed_entry["parent_next_step_id"]

            print(f"✓ Subworkflow completed: Returning to parent workflow {parent_workflow_id}, next step: {parent_next_step_id}")

            # Store subworkflow results in context under the configured output variable
            context[output_variable] = {
                "output": final_output,
                "results": {k: v.get("output") for k, v in results.items() if isinstance(v, dict) and "output" in v}
            }

            # Update session to return to parent workflow
            session_update = ConversationSessionUpdate(
                subworkflow_stack=subworkflow_stack if subworkflow_stack else None,
                workflow_id=parent_workflow_id,
                next_step_id=parent_next_step_id,
                context=context
            )
            conversation_session_service.update_session(self.db, conversation_id, session_update)
            self.db.refresh(session)

            # Continue parent workflow from where it left off
            # Pass empty user_message since we're continuing, not responding to new input
            return await self.execute_workflow(
                user_message="",
                company_id=company_id,
                workflow_id=parent_workflow_id,
                conversation_id=conversation_id,
                attachments=None
            )

        # Finalizing the workflow (only if not a subworkflow)
        # Instead of marking the session as 'completed', keep it 'active' so multiple workflows can run
        # and the conversation remains visible. Track workflow completion in context.
        context['last_workflow_completed_at'] = datetime.now().isoformat()
        context['last_workflow_id'] = workflow_obj.id

        # Clean up any extraction-related markers from context and memory so they don't interfere with future runs
        extraction_markers = [
            '_extracting_entity_name', '_missing_entities', '_extraction_attempts',
            'variable_to_save', 'expected_input_type',
            'pending_prompt_options', 'pending_allow_text_input', 'pending_validation_mode',
            'pending_validation_llm_provider', 'pending_validation_llm_model', 'pending_prompt_text',
            '_validation_max_retries', '_validation_retry_count',
            'pending_listen_validation_mode', 'pending_listen_validation_llm_provider',
            'pending_listen_validation_llm_model', 'pending_question_text',
            '_listen_validation_max_retries', '_listen_validation_retry_count',
        ]
        for marker in extraction_markers:
            context.pop(marker, None)
            # Also delete from memory service
            try:
                if self._executing_agent_id:
                    memory_service.delete_memory(self.db, marker, self._executing_agent_id, conversation_id)
            except:
                pass  # Marker might not exist in memory
        print(f"DEBUG: Cleaned up extraction markers from context and memory on workflow completion")

        # Reset context to a clean slate - only retain workflow completion bookkeeping metadata.
        # All user-collected variables (e.g. user_name), internal state keys (_last_agent_message,
        # initial_user_message, user_attachments, etc.) must be discarded so the next workflow
        # starts completely fresh and does not pick up stale data from the finished run.
        clean_context = {
            'last_workflow_completed_at': context.get('last_workflow_completed_at'),
            'last_workflow_id': context.get('last_workflow_id'),
        }
        context = clean_context
        print(f"DEBUG: Reset session context to clean slate after workflow completion")

        # Update session context
        session_update = ConversationSessionUpdate(status='active', context=context, subworkflow_stack=None)
        conversation_session_service.update_session(self.db, conversation_id, session_update)

        # Clear workflow_id and next_step_id directly so next message triggers fresh workflow search
        session.workflow_id = None
        session.next_step_id = None
        self.db.commit()
        self.db.refresh(session)
        print(f"DEBUG: Workflow completed. Cleared workflow_id and next_step_id for session {conversation_id}")

        # Clear all memory for this session so next workflow starts fresh
        if self._executing_agent_id:
            memory_service.delete_all_memories(self.db, agent_id=self._executing_agent_id, session_id=conversation_id)
            print(f"DEBUG: Cleared all memory for session {conversation_id}")

        return {"status": "completed", "response": final_output, "conversation_id": conversation_id}