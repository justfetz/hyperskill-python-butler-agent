import certifi
import dotenv
import httpx
import json
import os
import ssl
from openai import OpenAI

dotenv.load_dotenv()


def build_ssl_context():
    """certifi's roots plus the OS roots.

    This network runs a TLS-inspecting proxy (Cisco Umbrella) that re-signs
    HTTPS with a corporate root. Windows trusts it; certifi does not. The OS
    certs are read once here, not on every handshake.
    """
    context = ssl.create_default_context(cafile=certifi.where())
    if hasattr(ssl, "enum_certificates"):  # Windows only
        pem = "".join(
            ssl.DER_cert_to_PEM_cert(der)
            for store in ("ROOT", "CA")
            for der, encoding, _ in ssl.enum_certificates(store)
            if encoding == "x509_asn"
        )
        try:
            context.load_verify_locations(cadata=pem)
        except ssl.SSLError:
            pass
    return context


client = OpenAI(
    api_key=os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("LITELLM_BASE_URL") or os.getenv("BASE_URL"),
    max_retries=2,  # backoff is 0.5s + 1s; must fit hstest's 15s per-test budget
    timeout=8.0,   # a hung attempt + retry must still fit in 15s
    http_client=httpx.Client(verify=build_ssl_context()),
)

SYSTEM_PROMPT = """
You are a helpful assistant, your goal is to help user.
You have an access to the wardrobe and weather.
Don't ask for permission to do the task, just do everything you can to help user.
"""

MODEL_NAME = "gpt-4o-mini"
MAX_ITERATIONS = 5

EXIT_COMMANDS = ("q", "quit", "exit")

WARDROBE = {
    "blue sweater": "dirty",
    "brown jacket": "dirty",
}


def check_weather():
    return "Cold, rainy"


def get_wardrobe_items():
    return "; ".join(f"Item {name} is {status}" for name, status in WARDROBE.items())


def wash_clothing(item_name):
    if item_name not in WARDROBE:
        return f"Item '{item_name}' not found in wardrobe"

    WARDROBE[item_name] = "clean"
    return f"{item_name} is washed"


TOOLS_REGISTRY = [
    {
        "type": "function",
        "name": "check_weather",
        "description": "Check the current weather conditions outside.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_wardrobe_items",
        "description": "List every clothing item in the wardrobe with its "
                       "status, either clean or dirty.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "wash_clothing",
        "description": "Wash a dirty clothing item to make it clean. Use this "
                       "whenever the user needs an item that is currently dirty.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "Name of the clothing item to wash",
                },
            },
            "required": ["item_name"],
        },
    },
]

TOOL_NAME_TO_FUNC = {
    "check_weather": check_weather,
    "get_wardrobe_items": get_wardrobe_items,
    "wash_clothing": wash_clothing,
}


def run_agent_loop(context):
    print("[ENTERING AGENT LOOP]")
    answer = None

    for _ in range(MAX_ITERATIONS):
        response = client.responses.create(
            model=MODEL_NAME,
            instructions=SYSTEM_PROMPT,
            input=context,
            tools=TOOLS_REGISTRY,
        )
        print(f"[THINK]: Model decided to return these items: "
              f"{[type(item).__name__ for item in response.output]}")

        context += response.output
        called_tool = False

        for item in response.output:
            if item.type == "function_call":
                called_tool = True
                arguments = json.loads(item.arguments)
                print(f'[ACT]: Calling "{item.name}" with arguments {arguments}')

                result = TOOL_NAME_TO_FUNC[item.name](**arguments)
                print(f"[OBSERVE]: Result {result}")

                context.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result,
                })
            elif item.type == "message":
                answer = "".join(part.text for part in item.content
                                 if part.type == "output_text")

        # Only final once the model stops asking for tools -- otherwise a tool
        # result would never make it back to the model.
        if answer is not None and not called_tool:
            print("[EXITING AGENT LOOP]")
            return answer

    print("[EXITING AGENT LOOP]")
    return answer or "I wasn't able to finish that within the iteration limit."


def main():
    context = []

    while True:
        try:
            user_message = input("[USER]: ").strip()
        except EOFError:
            break

        if user_message.lower() in EXIT_COMMANDS:
            break

        context.append({"role": "user", "content": user_message})
        answer = run_agent_loop(context)
        print(f"[ASSISTANT]: {answer}")


if __name__ == "__main__":
    main()
