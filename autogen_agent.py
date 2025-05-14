import re
import requests

def parse_query(query):
    """
    Parse the user query to extract topic, year, comparison type, and citation count.
    """
    pattern = r"find a research paper on (.*?) that was published (in|before|after) (\d{4}) and has (\d+) citations"
    match = re.search(pattern, query, re.IGNORECASE)
    if not match:
        return None, None, None, None

    topic = match.group(1).strip()
    comparator = match.group(2).lower()
    year = int(match.group(3))
    citations = int(match.group(4))
    return topic, year, comparator, citations

def search_papers(topic, year, comparator, citations):
    """
    Use the Semantic Scholar API to search for research papers matching the specified criteria.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': topic,
        'limit': 20,
        'fields': 'title,year,citationCount'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        papers = response.json().get('data', [])

        def year_match(paper_year):
            if not paper_year:
                return False
            if comparator == 'in':
                return paper_year == year
            elif comparator == 'before':
                return paper_year < year
            elif comparator == 'after':
                return paper_year > year
            return False

        filtered = [
            paper for paper in papers
            if year_match(paper.get('year')) and paper.get('citationCount', 0) >= citations
        ]

        return filtered

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return []

def evaluate_agent(predicted_results, true_relevant_papers):
    """
    Evaluates the agent's performance using precision, recall, and F1-score.
    
    :param predicted_results: List of predicted (returned) papers by the agent.
    :param true_relevant_papers: List of manually verified relevant papers.
    :return: Dictionary containing precision, recall, and F1-score.
    """
 
    predicted_set = set(predicted_results)
    true_set = set(true_relevant_papers)


    true_positive = len(predicted_set.intersection(true_set))  
    false_positive = len(predicted_set - true_set) 
    false_negative = len(true_set - predicted_set)  

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def get_sample_true_relevant_papers(query):
    """
    Simulate fetching manually validated relevant papers for evaluation.
    Ideally, this list should be gathered through manual review or from a trusted source.
    
    :param query: The user query that was submitted for evaluation.
    :return: A list of relevant papers for the given query.
    """

    if "computer vision" in query.lower():
        return [
            "Design and Development of Cost-Effective Child Surveillance System using Computer Vision Technology",
            "A Comprehensive Review of YOLO Architectures in Computer Vision",
            "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
        ]
    return []

def main():
    """
    Main function to run the research paper finder agent.
    """
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


            true_relevant_papers = get_sample_true_relevant_papers(user_input)
            evaluation_metrics = evaluate_agent([paper.get('title') for paper in results], true_relevant_papers)
            print("\nEvaluation Metrics:")
            print(f"Precision: {evaluation_metrics['precision']:.2f}")
            print(f"Recall: {evaluation_metrics['recall']:.2f}")
            print(f"F1-score: {evaluation_metrics['f1_score']:.2f}")

if __name__ == "__main__":
    main()
