from agent.agent import run_agent


def normalize(text):
    return (
        text.lower()
        .replace(",", "")
        .replace(".", "")
        .replace("*", "")
        .strip()
    )


tests = [

    {
        "name": "Calculator",
        "query": "Calculate 125 * 48",
        "expected": "6000"
    },

    {
        "name": "Weather",
        "query": "What is the current weather in Chennai?",
        "expected": "temperature"
    },

    {
        "name": "Policy",
        "query": "What is considered confidential information?",
        "expected": "confidential"
    },

    {
        "name": "Policy - Information Not Found",
        "query": "What is the company's policy on employees bringing pets to work?",
        "expected": "no information"
    },

    {
        "name": "Multi Tool",
        "query": "Calculate 25 * 20 and tell me the current weather in Chennai.",
        "expected": "500"
    },

    {
        "name": "General Conversation",
        "query": "Hello, how are you?",
        "expected": "help"
    }
]


passed = 0
review = 0
failed = 0
skipped = 0


for test in tests:

    print("\n" + "=" * 60)
    print("TEST:", test["name"])
    print("QUERY:", test["query"])
    print("=" * 60)

    try:

        answer = run_agent(test["query"])

        print("ANSWER:")
        print(answer)

        if normalize(test["expected"]) in normalize(answer):

            print("\nSTATUS: PASS")
            passed += 1

        else:

            print("\nSTATUS: REVIEW")
            review += 1

    except Exception as e:

        error_message = str(e)

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:

            print("\nSTATUS: SKIPPED")
            print("REASON: Gemini API quota reached.")
            skipped += 1

        else:

            print("\nSTATUS: FAIL")
            print("ERROR:", e)
            failed += 1


print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

print("PASS   :", passed)
print("REVIEW :", review)
print("FAIL   :", failed)
print("SKIP   :", skipped)