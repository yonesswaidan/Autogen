from autogen_agent import parse_query, search_papers

def main():
    print("Welcome to the Research Paper Finder!")
    print("Ask something like: 'Find a research paper on artificial intelligence that was published after 2020 and has 100 citations.'\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        topic, year, comparator, citations = parse_query(user_input)
        if not topic:
            print("Sorry, I couldn't understand your query. Please try again.")
            continue

        print("\nSearching...\n")
        results = search_papers(topic, year, comparator, citations)

        if not results:
            print("No papers found matching the criteria.\n")
        else:
            print("Here are some matching papers:\n")
            for paper in results:
                title = paper.get('title', 'No title')
                year = paper.get('year', 'N/A')
                citation_count = paper.get('citationCount', 'N/A')
                print(f"- {title} ({year}) - {citation_count} citations")

if __name__ == "__main__":
    main()
