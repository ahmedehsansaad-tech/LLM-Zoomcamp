import json

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()

search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="gpt-5.4-mini"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )


    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer
    

    def rag_agentic(self, query):
        messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": query}
        ]

        # Call #1 — give the model the tool, let it decide
        response = self.llm_client.responses.create(
            model=self.model,
            input=messages,
            tools=[search_tool],
        )

        # Check if the model asked to call search
        call = response.output[0]

        if call.type == "function_call":
            args = json.loads(call.arguments)
            print(f"Model is searching for: {args['query']}")

            # This is YOUR existing self.search — no new search logic at all
            results = self.search(args["query"])
            result_json = json.dumps(results, indent=2)

            # Feed the model's own decision + the result back into messages
            messages.extend(response.output)
            messages.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": result_json,
            })

            # Call #2 — now the model has real FAQ data to answer from
            response = self.llm_client.responses.create(
                model=self.model,
                input=messages,
                tools=[search_tool],
            )

        return response.output_text

    def make_call(self, call):
        args = json.loads(call.arguments)

        if call.name == "search":
            result = self.search(**args)

        result_json = json.dumps(result, indent=2)

        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": result_json,
        }


    def agent_loop(self, question) -> str:
        messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": question}
        ]

        it = 1

        while True:
            print(f"iteration #{it}...")
            has_function_calls = False

            response = self.llm_client.responses.create(
                model=self.model,
                input=messages,
                tools=[search_tool]
            )

            messages.extend(response.output)

            for item in response.output:
                if item.type == "function_call":
                    print("function_call:", item.name, item.arguments)
                    call_output = self.make_call(item)
                    messages.append(call_output)
                    has_function_calls = True

                elif item.type == "message":
                    print("ASSISTANT:")
                    last_answer = item.content[0].text
                    print(item.content[0].text)

            it = it + 1
            if has_function_calls == False:
                break

        return last_answer