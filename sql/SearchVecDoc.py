import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Tuple
from sql.ImportDB2Faiss import VetFAISS
# Function to search vet documents based on user query
class VetDocumentSearch:
    @staticmethod
    def search_documents(query, top_k=2):
        """
        Searches veterinary documents using FAISS.

        :param query: The search query (e.g., symptoms, diagnosis).
        :param top_k: Number of top results to return.
        :return: List of matching documents.
        """
        # Assuming VetFAISS.search_vet_doc() is available
        search_results = VetFAISS.search_vet_doc(query, top_k=top_k)

        # Print the search results
        print("\n🔍 Search Results for:", query)
        for result in search_results:
            print(result)  # Modify based on the actual structure of the results

        return search_results

    def convert_float32_to_float(data):
        if isinstance(data, dict):
            return {k: convert_float32_to_float(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_float32_to_float(i) for i in data]
        elif isinstance(data, np.float32):  # Convert np.float32 to float
            return float(data)
        return data
