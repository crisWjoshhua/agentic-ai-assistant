# ==================================================
# AGENT EVALUATION TEST CASES
# ==================================================


test_cases = [

    # ----------------------------------------------
    # Calculator
    # ----------------------------------------------

    {
        "name": "Calculator Test",
        "input": "Calculate 25 * 40",
        "expected_behavior": "Use calculator tool"
    },


    # ----------------------------------------------
    # Weather
    # ----------------------------------------------

    {
        "name": "Weather Test",
        "input": "What is the weather in Chennai?",
        "expected_behavior": "Use weather tool"
    },


    # ----------------------------------------------
    # Policy / RAG
    # ----------------------------------------------

    {
        "name": "Policy Test",
        "input": "What is the leave policy?",
        "expected_behavior": "Use policy search tool"
    },


    # ----------------------------------------------
    # Follow-up / Memory
    # ----------------------------------------------

    {
        "name": "Memory Follow-up Test",
        "input": "What about parental leave?",
        "expected_behavior": "Use previous conversation context"
    },


    # ----------------------------------------------
    # Irrelevant question
    # ----------------------------------------------

    {
        "name": "General Question Test",
        "input": "What is artificial intelligence?",
        "expected_behavior": "Answer normally without policy search"
    },


    # ----------------------------------------------
    # Multiple tools
    # ----------------------------------------------

    {
        "name": "Multiple Tool Test",
        "input": "Calculate 20 * 5 and tell me the weather in Chennai",
        "expected_behavior": "Use calculator and weather tools"
    }

]


if __name__ == "__main__":

    print("\n========================================")
    print("       AGENT EVALUATION TEST CASES")
    print("========================================\n")

    for test in test_cases:

        print("Test:", test["name"])
        print("Input:", test["input"])
        print("Expected:", test["expected_behavior"])
        print("----------------------------------------")