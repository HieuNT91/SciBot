from scibot.search import SemanticSearch
from scibot.ask import AskPipeline, Colors
from scibot.ask_tool import llama_with_tools


def main():
    # Welcome message
    print(f"{Colors.BOLD}Welcome to Scibot!{Colors.RESET}")

    # Default paths and settings
    search_index_path = "database/processed_full_neurips/faiss_index_flatip"
    search_metadata_path = "database/processed_full_neurips/metadata.json"
    model_name = "all-MiniLM-L6-v2"
    ollama_model = "llama3.1:70b"
    default_top_k = 5

    # Prompt user for paths and settings with defaults
    index_path = input(f"Enter the path to the FAISS index file [default: {search_index_path}]: ").strip() or search_index_path
    metadata_path = input(f"Enter the path to the metadata JSON file [default: {search_metadata_path}]: ").strip() or search_metadata_path
    model_name = input(f"Enter the model name [default: {model_name}]: ").strip() or model_name
    top_k = input(f"Enter the number of top results to retrieve [default: {default_top_k}]: ").strip() or default_top_k

    # Convert `top_k` to an integer
    try:
        top_k = int(top_k)
    except ValueError:
        print(f"{Colors.RED}Invalid input for top_k. Using default value: {default_top_k}{Colors.RESET}")
        top_k = default_top_k

    print(f"\n{Colors.CYAN}Loading resources...{Colors.RESET}")

    try:
        # Load resources only once using SemanticSearch
        with SemanticSearch(index_path, metadata_path, model_name) as search_engine:
            ask_engine = AskPipeline(
                model=search_engine.model,
                index=search_engine.index,
                metadata=search_engine.metadata,
                ollama_model=ollama_model,
            )

            print(f"{Colors.GREEN}Resources loaded. You can now enter your queries.{Colors.RESET}")

            while True:
                # Prompt for user input
                user_input = input(f"\n{Colors.BOLD}Enter your query (use 'ask:', 'ask+tool:', or 'search:', or type 'exit' to quit): {Colors.RESET}").strip()
                if user_input.lower() == "exit":
                    print(f"{Colors.MAGENTA}Exiting Scibot. Goodbye!{Colors.RESET}")
                    break

                # Detect mode based on prefix
                if user_input.lower().startswith("ask+tool:"):
                    # Handle tool-using queries
                    query = user_input[10:].strip()  # Remove the 'ask+tool:' prefix
                    print(f"{Colors.CYAN}Running ask+tool pipeline for query: {query}{Colors.RESET}")
                    try:
                        response = llama_with_tools(query)
                        print(f"\n{Colors.BOLD}Final Response:{Colors.RESET}\n{response}")
                    except Exception as e:
                        print(f"{Colors.RED}Error during ask+tool pipeline: {e}{Colors.RESET}")

                elif user_input.lower().startswith("ask:"):
                    # Handle RAG-based queries
                    query = user_input[4:].strip()  # Remove the 'ask:' prefix
                    print(f"{Colors.CYAN}Running RAG pipeline for query: {query}{Colors.RESET}")
                    try:
                        response = ask_engine.generate_rag_response(query, top_k=top_k)
                        print(f"\n{Colors.BOLD}Generated Response:{Colors.RESET}\n{response}")
                    except Exception as e:
                        print(f"{Colors.RED}Error during RAG pipeline: {e}{Colors.RESET}")

                elif user_input.lower().startswith("search:"):
                    # Handle semantic search queries
                    query = user_input[7:].strip()  # Remove the 'search:' prefix
                    print(f"{Colors.CYAN}Running SemanticSearch for query: {query}{Colors.RESET}")
                    try:
                        results = search_engine.query(query_text=query, top_k=top_k)
                        print(f"\n{Colors.BOLD}Top Results:{Colors.RESET}")
                        for i, result in enumerate(results):
                            rank_color = Colors.GREEN if i == 0 else (Colors.YELLOW if i == 1 else Colors.CYAN)
                            print(f"{rank_color}Rank {i+1}:{Colors.RESET}")
                            print(f"  {Colors.BOLD}Title:{Colors.RESET} {result.get('title', 'N/A')}")
                            print(f"  {Colors.BOLD}Chunk:{Colors.RESET} {result.get('chunk', 'N/A')}")
                            print(f"  {Colors.GRAY}Score:{Colors.RESET} {result['score']}")
                            print()
                    except Exception as e:
                        print(f"{Colors.RED}Error during SemanticSearch: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.RED}Invalid input. Please use 'ask:', 'ask+tool:', or 'search:' prefixes.{Colors.RESET}")

    except Exception as e:
        print(f"{Colors.RED}An error occurred during initialization: {e}{Colors.RESET}")

    finally:
        print(f"\n{Colors.CYAN}Exiting Scibot and cleaning up resources...{Colors.RESET}")


if __name__ == "__main__":
    main()