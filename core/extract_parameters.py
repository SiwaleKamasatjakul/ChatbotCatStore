from core.request_templates import *
from typing import List
import json
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sql.SearchVecDoc import VetDocumentSearch

class ExtractParameters:
    @staticmethod
    def _process_messages(messages):
        allowed_roles = ["user", "assistant"]
        system_prompt = ""
        request_messages_list = []
        context = []

        for message in messages:
            if message.role in ["userprofile"]:
                context.append(json.dumps(message.detail, indent=4,ensure_ascii=False))
            elif message.role in allowed_roles:
                request_messages_list.append({message.role: message.content})
            elif message.role == "system":
                system_prompt += f"\n{message.content}"

        return system_prompt, request_messages_list, context

    @staticmethod
    def _extract_questions_and_history(request_messages_list: List[dict]):
        if not request_messages_list:
            return "", ""

        questions = request_messages_list[-1].get('user', "")
        history = "\n".join([str(request) for request in request_messages_list[:-1]])

        return questions, history

    @staticmethod
    def extract_parameters(body: RequestBody):
        parameters = body.parameters
        model_name = body.modelid
        request_id = body.requestid
        refdata = body.refdata
        session_id = body.sessionid

        system_prompt, request_messages_list, context = ExtractParameters._process_messages(body.messages)
        questions, history = ExtractParameters._extract_questions_and_history(request_messages_list)
        
        # Adding Vet Doc after use Faiss to search 
        vet_doc = VetDocumentSearch.search_documents(question)
        vet_doc_cleaned = VetDocumentSearch.convert_float32_to_float(vet_doc)  # Convert float32 values

        # Append to context
        context.append(json.dumps(vet_doc_cleaned, indent=4, ensure_ascii=False))
        
        return Chatbot_Messages(
            system_prompt=system_prompt,
            stream=parameters.stream,
            questions=questions,
            parameters=parameters,
            model_name=model_name,
            request_id=request_id,
            session_id=session_id,
            history=history,
            context='\n'.join(context),
            refdata=refdata
        )


