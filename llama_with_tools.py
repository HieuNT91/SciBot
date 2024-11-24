import requests
import json
import re
import sqlite3

# Define the tools
def calculator_tool(expression):
    """
    A simple calculator tool. Evaluates mathematical expressions.
    """
    try:
        # Evaluate the mathematical expression
        print(expression)
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def database_query_tool(query, db_path="database/full_neurips_metadata.db"):
    """
    A tool for querying the SQLite database with custom SQL.
    
    Parameters:
        query (str): The SQL query to execute.
        db_path (str): Path to the SQLite database.
        
    Returns:
        str: The query results or an error message.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute the query
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Format the results as a string
        if rows:
            column_names = [description[0] for description in cursor.description]  # Get column headers
            result = "\n".join([", ".join(map(str, row)) for row in rows])
            result = f"Columns: {', '.join(column_names)}\nResults:\n{result}"
        else:
            result = "No results found."
        
        conn.close()
        return result
    except Exception as e:
        return f"Error: {e}"
    
def parse_tool_request(response):
    """
    Parse the model's response to determine if it wants to use a tool.
    Extracts the tool name and its input.
    """
    # Regular expression to extract tool name and input from the response
    tool_pattern = r"Tool:\s*(\w+)\s*Input:\s*(.*)"
    match = re.search(tool_pattern, response, re.DOTALL)  # Match across multiple lines if necessary
    if match:
        tool_name = match.group(1).strip()  # Extract the tool name (e.g., "Calculator")
        tool_input = match.group(2).strip()  # Extract the tool input (e.g., "2+2")
        return tool_name, tool_input
    return None, None

# Run the appropriate tool
def run_tool(tool_name, tool_input):
    """
    Execute the appropriate tool based on the tool_name.
    """
    if tool_name == "Calculator":
        return calculator_tool(tool_input)
    elif tool_name == "DatabaseQuery":
        # Pass the tool_input as the SQL query
        return database_query_tool(tool_input)
    else:
        return f"Error: Tool '{tool_name}' is not recognized."

# Communicate with the LLaMA model
def interact_with_llama(prompt, model_name="llama3.2"):
    """
    Send a prompt to the LLaMA model and get a response.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Accumulate the streaming response
            full_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:  # Ignore keep-alive newlines
                    try:
                        # Parse each line as JSON
                        chunk_data = json.loads(line)

                        # Extract the "content" field from "message"
                        if "message" in chunk_data and "content" in chunk_data["message"]:
                            content = chunk_data["message"]["content"]
                            full_response += content  # Append to the accumulated response
                            print(content, end="", flush=True)  # Stream in real-time
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON line: {line}")  # Handle malformed JSON gracefully

            return full_response

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to LLaMA API: {e}")
        return None

def llama_with_tools(prompt, model_name="llama3.1:70b"):
    """
    A loop allowing LLaMA to use tools and continue reasoning.
    """
    while True:
        # Step 1: Get the model's response
        response = interact_with_llama(prompt, model_name=model_name)
        if not response:
            print("No response from LLaMA.")
            break

        # Step 2: Check if the model wants to use a tool
        tool_name, tool_input = parse_tool_request(response)
        if tool_name:
            print(f"\nUsing tool: {tool_name} with input: {tool_input}")

            # Step 3: Run the tool and get the result
            tool_result = run_tool(tool_name, tool_input)
            print(f"Tool Result: {tool_result}")

            # Step 4: Provide the result back to the model
            prompt += f"\nTool Result: {tool_result}\n"
        else:
            # If no tool is requested, assume the conversation is complete
            break

    return response

# Example usage
if __name__ == "__main__":

    initial_prompt = """
    You are a helpful assistant that can use tools to assist the user. 
    If you need to use a tool, respond in the following format:
    
    Tool: <Tool Name>
    Input: <Input for the Tool>
    
    Here are the tools you can use:
    1. Calculator: Use this tool for mathematical calculations.
       Example:
       User: What is 2 + 2?
       Assistant: 
       Tool: Calculator
       Input: 2 + 2
       
    2. DatabaseQuery: Use this tool to query the NeurIPS metadata database.
       Example:
       User: Show me all papers published in 2018.
       Assistant:
       Tool: DatabaseQuery
       Input: SELECT title, authors FROM papers WHERE published LIKE '2018%';

       (After using the tool, the assistant will respond with the query results or an error message.)
    
    If you don't need to use a tool, respond directly with your answer.

    Let's begin.

    User: Which paper did Hoi-to Wai wrote ?
    """
    print("\n=== Starting Tool-Using LLaMA ===\n")
    response = llama_with_tools(initial_prompt)
    print("\n\nFinal Response:")
    print(response)