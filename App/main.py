from agent.agent import run_agent


while True:

    query = input("\nAsk anything: ")

    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    answer = run_agent(query)

    print("\nAssistant:")
    print(answer)