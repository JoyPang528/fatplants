import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import re
import urllib.parse
import json
import os

# --- Configuration ---
NCBI_API_KEY = "your API key"
ENTREZ_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

class CitationRetrievalService:
    """
    A service to retrieve, rank, and format scientific citations from NCBI PubMed.
    """
    def __init__(self, ncbi_api_key: str = NCBI_API_KEY):
        self.ncbi_api_key = ncbi_api_key
        self.logger = print # Simple logger for demonstration

    def _extract_entities(self, query: str) -> list[str]:
        """
        Extracts potential entities/keywords from the user query,
        with a focus on biomedical terms like gene symbols.
        """
        unique_entities = set()
        
        # 1. Prioritize terms that look like gene/protein symbols (all caps, alphanumeric)
        #    or specific combinations like "MAPT-STX6"
        #    This regex looks for words that are all uppercase, or contain hyphens/numbers/letters
        #    and are at least 2 characters long.
        gene_protein_pattern = r'\b[A-Z0-9]+(?:[-_][A-Z0-9]+)*\b'
        for match in re.findall(gene_protein_pattern, query):
            if len(match) > 1: # Avoid single letters unless they are specific (e.g., 'A' for ATP is too broad)
                unique_entities.add(match.upper()) # Ensure consistency in casing

        # 2. Add specific interaction keywords
        if "interaction" in query.lower():
            unique_entities.add("interaction")
        if "interact" in query.lower():
            unique_entities.add("interact")

        # 3. Fallback to general keyword extraction for other important terms
        #    Filter out common stop words and short words, but keep relevant ones like "disease", "metabolism"
        words = re.findall(r'\b\w+\b', query.lower())
        stop_words = set([
            "a", "an", "the", "is", "are", "and", "or", "in", "on", "with", "for",
            "of", "what", "which", "how", "when", "where", "why", "role", "effect",
            "about", "tell", "show", "me", "genes", "proteins", "pathways" # Added more common chat words
        ])
        
        for word in words:
            if word not in stop_words and len(word) > 2:
                # Avoid adding words already captured by the gene/protein pattern if they are short
                if word.upper() not in unique_entities or len(word) > 3: # Only add if not already captured or if longer
                    unique_entities.add(word)
        
        # Convert to list and log
        extracted_list = list(unique_entities)
        self.logger(f"Extracted entities (internal): {extracted_list}")
        print(extracted_list)
        return extracted_list

    def _search_pubmed(self, search_term: str, max_results: int = 10) -> list[dict]:
        """
        Searches the NCBI PubMed database using the Entrez ESearch and EFetch APIs.
        Takes a pre-formatted search term string.
        """
        if not search_term.strip():
            self.logger("Search term is empty. Cannot search PubMed.")
            return []

        self.logger(f"DEBUG: API Key inside _search_pubmed: '{self.ncbi_api_key}'")
        if not self.ncbi_api_key:
            self.logger("ERROR: NCBI_API_KEY is empty. Cannot proceed with API call.")
            return []

        esearch_url = f"{ENTREZ_BASE_URL}esearch.fcgi"
        esearch_params = {
            "db": "pubmed",
            "term": search_term,
            "retmax": max_results,
            "retmode": "json",
            "api_key": self.ncbi_api_key
        }
        try:
            esearch_response = requests.get(esearch_url, params=esearch_params)
            
            self.logger(f"ESearch URL: {esearch_response.url}")
            self.logger(f"ESearch Status: {esearch_response.status_code}")
            self.logger(f"ESearch Response Headers: {esearch_response.headers}")
            self.logger(f"ESearch Response Text (first 500 chars): {esearch_response.text[:500]}...")

            esearch_response.raise_for_status()
            esearch_data = esearch_response.json()
            pubmed_ids = esearch_data.get("esearchresult", {}).get("idlist", [])
            self.logger(f"Found {len(pubmed_ids)} PubMed IDs: {pubmed_ids}")
        except requests.exceptions.RequestException as e:
            self.logger(f"ESearch failed with RequestException: {e}")
            return []
        except json.JSONDecodeError as e:
            self.logger(f"ESearch JSON decode error: {e}. Response was: {esearch_response.text[:200]}...")
            return []

        if not pubmed_ids:
            self.logger("No PubMed IDs found in ESearch result.")
            return []

        efetch_url = f"{ENTREZ_BASE_URL}efetch.fcgi"
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(pubmed_ids),
            "retmode": "xml",
            "api_key": self.ncbi_api_key
        }
        try:
            efetch_response = requests.get(efetch_url, params=efetch_params)
            
            self.logger(f"EFetch URL: {efetch_response.url}")
            self.logger(f"EFetch Status: {efetch_response.status_code}")
            self.logger(f"EFetch Response Headers: {efetch_response.headers}")
            self.logger(f"EFetch Response Text (first 500 chars): {efetch_response.content[:500]}...")

            efetch_response.raise_for_status()
            root = ET.fromstring(efetch_response.content)
        except requests.exceptions.RequestException as e:
            self.logger(f"EFetch failed with RequestException: {e}")
            return []
        except ET.ParseError as e:
            self.logger(f"EFetch XML parse error: {e}. Response was: {efetch_response.content[:200]}...")
            return []

        citations = []
        for article in root.findall(".//PubmedArticle"):
            try:
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else "N/A"

                authors_list = []
                for author_elem in article.findall(".//AuthorList/Author"):
                    last_name = author_elem.find("LastName")
                    fore_name = author_elem.find("ForeName")
                    initials = author_elem.find("Initials")
                    
                    author_name_parts = []
                    if last_name is not None and last_name.text:
                        author_name_parts.append(last_name.text.strip())
                    if initials is not None and initials.text:
                        author_name_parts.append(initials.text.strip())
                    elif fore_name is not None and fore_name.text:
                         author_name_parts.append(fore_name.text.strip())

                    if author_name_parts:
                        authors_list.append(" ".join(author_name_parts))
                authors = ", ".join(authors_list) if authors_list else "N/A"

                journal_elem = article.find(".//Journal/Title")
                journal = journal_elem.text.strip() if journal_elem is not None and journal_elem.text else "N/A"

                pub_date_elem = article.find(".//PubDate")
                year = pub_date_elem.find("Year")
                month = pub_date_elem.find("Month")
                day = pub_date_elem.find("Day")
                
                publication_date = None
                if year is not None and year.text:
                    try:
                        date_str = f"{year.text.strip()}-{month.text.strip() if month is not None and month.text else '01'}-{day.text.strip() if day is not None and day.text else '01'}"
                        publication_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        try:
                            publication_date = datetime(int(year.text.strip()), 1, 1)
                        except ValueError:
                            publication_date = None

                pub_types = [pt.text for pt in article.findall(".//PublicationTypeList/PublicationType") if pt.text]
                
                citation_data = {
                    "title": title,
                    "authors": authors,
                    "journal": journal,
                    "publication_date": publication_date,
                    "publication_types": pub_types
                }
                citations.append(citation_data)

                self.logger(f"Parsed Article: Title='{title}', Authors='{authors}', Journal='{journal}'")

            except Exception as e:
                self.logger(f"Error parsing article: {e}. Article XML (first 200 chars): {ET.tostring(article, encoding='unicode')[:200]}...")
                continue
        
        self.logger(f"Retrieved {len(citations)} detailed citations.")
        return citations

    def _rank_citations(self, citations: list[dict]) -> list[dict]:
        """
        Ranks citations based on publication type and recency.
        """
        type_priority = {
            "Review": 3, "Journal Article": 2, "Clinical Trial": 2, "Meta-Analysis": 2,
            "Guideline": 2, "Case Reports": 1, "Letter": 0, "Editorial": 0, "Comment": 0,
        }

        current_year = datetime.now().year

        def get_rank_score(citation: dict) -> float:
            score = 0
            max_type_score = 0
            for pub_type in citation.get("publication_types", []):
                max_type_score = max(max_type_score, type_priority.get(pub_type, 0))
            score += max_type_score * 100

            pub_date = citation.get("publication_date")
            if pub_date:
                years_ago = current_year - pub_date.year
                recency_score = max(0, 10 - years_ago)
                score += recency_score
            
            return score

        ranked_citations = sorted(citations, key=get_rank_score, reverse=True)
        self.logger(f"Ranked {len(ranked_citations)} citations.")
        return ranked_citations

    def _format_citation(self, citation: dict) -> str:
        """
        Formats a single citation into the specified Markdown string format.
        """
        title = citation.get("title", "N/A")
        authors = citation.get("authors", "N/A")
        journal = citation.get("journal", "N/A")

        encoded_title = urllib.parse.quote_plus(title)
        google_search_link = f"https://www.google.com/search?q={encoded_title}&btnI=I%27m%20Feeling%20Lucky"
        

        # Corrected format: label and value on the same line
        formatted_string = (
            f"Title of the paper: {title}\n"
            f"Authors: {authors}\n"
            f"Journal: {journal}\n"
            f"[Link]({google_search_link})\n"
        )
        return formatted_string

    async def retrieve_and_rank_citations(self, query: str, num_citations: int = 3) -> list[str]:
        """
        Orchestrates the citation retrieval and ranking process.
        This method now performs its own entity extraction.
        """
        # Perform entity extraction internally
        entities = self._extract_entities(query)
        search_term = " ".join(entities) # Join extracted entities into a search string

        if not search_term.strip():
            self.logger("No entities extracted or search term is empty. Cannot search for citations.")
            return ["Not able to scrape citations for this question."]

        raw_citations = self._search_pubmed(search_term, max_results=num_citations * 2)
        if not raw_citations:
            self.logger("Not able to scrape citations for this question.")
            return ["Not able to scrape citations for this question."]

        ranked_citations = self._rank_citations(raw_citations)
        
        formatted_citations = []
        for i, citation in enumerate(ranked_citations[:num_citations]):
            formatted_citations.append(f"--- Citation {i+1} ---\n{self._format_citation(citation)}")
        
        return formatted_citations
    # async def retrieve_and_rank_citations(self, query: str, num_citations: int = 3) -> list[str]:
    #     """
    #     Orchestrates the citation retrieval and ranking process.

    #     Args:
    #         query: The user's input query.
    #         num_citations: The number of top citations to return.

    #     Returns:
    #         A list of formatted, top-ranked citation strings.
    #     """
    #     entities = self._extract_entities(query)
    #     if not entities:
    #         self.logger("No entities extracted, cannot search for citations.")
    #         return []

    #     raw_citations = self._search_pubmed(entities, max_results=num_citations * 2) # Fetch more to allow for ranking
    #     if not raw_citations:
    #         self.logger("Not able to scrape citations for this question.")
    #         return ["Not able to scrape citations for this question."]

    #     ranked_citations = self._rank_citations(raw_citations)
        
    #     formatted_citations = []
    #     for i, citation in enumerate(ranked_citations[:num_citations]):
    #         formatted_citations.append(f"--- Citation {i+1} ---\n{self._format_citation(citation)}")
        
    #     return formatted_citations
    

# --- Example Usage (How it would integrate into your backend) ---
async def main():
    citation_service = CitationRetrievalService()

    # Example 1: Successful retrieval
    query1 = "What genes are associated with Alzheimer's disease?"
    print(f"\n--- Processing Query: '{query1}' ---")
    citations1 = await citation_service.retrieve_and_rank_citations(query1, num_citations=2)
    print("\n".join(citations1))

    # Example 2: Another query
    query2 = "Role of APOE in neurodegeneration"
    print(f"\n--- Processing Query: '{query2}' ---")
    citations2 = await citation_service.retrieve_and_rank_citations(query2, num_citations=1)
    print("\n".join(citations2))

    # Example 3: Query with no expected results (or very few)
    query3 = "Non-existent gene interaction in a fictional disease"
    print(f"\n--- Processing Query: '{query3}' ---")
    citations3 = await citation_service.retrieve_and_rank_citations(query3, num_citations=1)
    print("\n".join(citations3))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
