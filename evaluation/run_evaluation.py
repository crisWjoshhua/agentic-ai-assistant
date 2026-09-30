# ==================================================
# AUTOMATED AGENT EVALUATION
# ==================================================

from evaluation.test_cases import test_cases

from agent.langgraph_manual import agent, extract_text


# ==================================================
# RUN ONE TEST
# ==================================================

def run_test(test, index):

    print("\n========================================")
    print(f"TEST {index}")
    print("========================================")

    print("Name:")
    print(test["name"])

    print("\nInput:")
    print(test["input"])

    print("\nExpected behavior:")
    print(test["expected_behavior"])


    # ----------------------------------------------
    # Create a separate conversation thread
    # ----------------------------------------------

    config = {
        "configurable": {
            "thread_id": f"evaluation_{index}"
        }
    }


    # ----------------------------------------------
    # Send input to agent
    # ----------------------------------------------

    result = agent.invoke(
        {
            "messages": [
                ("user", test["input"])
            ]
        },
        config=config
    )


    # ----------------------------------------------
    # Extract response
    # ----------------------------------------------

    response = extract_text(
        result["messages"][-1].content
    )


    print("\nActual response:")
    print(response)


    return response


# ==================================================
# MAIN EVALUATION
# ==================================================

if __name__ == "__main__":

    print("\n")
    print("========================================")
    print("       AGENT AUTOMATED EVALUATION")
    print("========================================")


    results = []


    for index, test in enumerate(test_cases, start=1):

        response = run_test(
            test,
            index
        )

        results.append({
            "name": test["name"],
            "response": response
        })


    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    print("\n\n")
    print("========================================")
    print("       EVALUATION COMPLETE")
    print("========================================")

    print(f"\nTotal tests: {len(results)}")

    for result in results:

        print("\nTest:", result["name"])
        print("Response generated successfully: YES")