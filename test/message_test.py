from typing import Optional, List
from pydantic import BaseModel
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Tuple
from ImportDB2Faiss import VetFAISS
from core.extract_parameters import ExtractParameters

class MessageContent(BaseModel):
    role: Optional[str] = ""
    type: Optional[str] = ""
    content: Optional[str] = ""
    detail: Optional[dict] = {}
    image_url: Optional[dict] = {}

class Parameters(BaseModel):
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 4096
    top_p: Optional[float] = 0.1
    frequency_penalty: Optional[float] = 1
    presence_penalty: Optional[float] = 0
    other_model_specific_parameters: Optional[str] = ""

class ExtractParametersNew:
    def _extract_questions_and_history(request_messages_list: List[dict]) -> Tuple[str, str]:
        if not request_messages_list:
            return "", ""

        # Extract the last user message (Ensures it is a 'user' message)
        questions = ""
        for message in reversed(request_messages_list):
            if 'user' in message:
                questions = message['user']
                break  # Stop at the latest user message

        # Extract history (Excluding the last user message)
        history = "\n".join([
            str(request) for request in request_messages_list[:-1]
        ])

        return questions, history


class RequestBody(BaseModel):
    requestid: str = ""
    sessionid: str = ""
    modelid: Optional[str] = "gpt-4o"
    messages: Optional[List[MessageContent]] = []
    parameters: Optional[Parameters] = Parameters()
    refdata: Optional[list] = []

# Create messages
messages = [
    MessageContent(
        role="user",
        content="""ตอนนี้รานีอ้วนมาก""",
        detail={},
        image_url={}
    ),
    MessageContent(
        role="assistant",
        content="""สวัสดีค่ะ คุณเปี๊ยก ปากเกร็ด! Nhamaew Pet Ai ยินดีที่ได้ช่วยดูแลน้องรานีค่ะ

จากข้อมูลที่คุณเปี๊ยกให้มา น้องรานีเป็นแมวอายุเพียง 2 เดือนเท่านั้น การที่น้องดูอ้วนในวัยนี้อาจเกิดจากหลายสาเหตุค่ะ มาดูกันว่าเราจะช่วยน้องได้อย่างไรบ้าง:

1. **ตรวจสอบอาหาร**: 
   - ตรวจสอบปริมาณและประเภทของอาหารที่ให้น้องรานีค่ะ ควรให้อาหารที่เหมาะสมกับวัยและน้ำหนักของน้อง โดยปกติแล้วลูกแมวควรได้รับอาหารลูกแมวที่มีสารอาหารครบถ้วน
   - หลีกเลี่ยงการให้อาหารคนหรือขนมที่มีไขมันสูง

2. **การออกกำลังกาย**:
   - แม้จะยังเล็ก แต่การเล่นกับน้องรานีด้วยของเล่นแมว เช่น ลูกบอลหรือของเล่นที่มีเสียง จะช่วยให้น้องได้ออกกำลังกายและเผาผลาญพลังงาน

3. **ตรวจสุขภาพ**:
   - หากน้องรานีดูอ้วนผิดปกติหรือมีอาการอื่น ๆ เช่น หายใจลำบาก ควรพาน้องไปพบสัตวแพทย์เพื่อตรวจสุขภาพเพิ่มเติมค่ะ

4. **การให้น้ำ**:
   - ควรให้น้ำสะอาดที่ไม่ใช่น้ำประปา เพื่อสุขภาพที่ดีของน้อง

Nhamaew Pet Ai หวังว่าคำแนะนำนี้จะช่วยให้น้องรานีมีสุขภาพดีและมีความสุขนะคะ หากมีคำถามเพิ่มเติมหรือข้อสงสัยใด ๆ คุณเปี๊ยกสามารถสอบถามได้เสมอค่ะ!""",
        detail={},
        image_url={}
    ),
    MessageContent(
        role="user",
        content="""แล้วต้องกินอาหารอะไร""",
        detail={},
        image_url={}
    ),
    MessageContent(
        role="assistant",
        content="""สวัสดีค่ะ คุณเปี๊ยก ปากเกร็ด! Nhamaew Pet Ai ยินดีที่ได้ช่วยดูแลน้องรานีค่ะ

สำหรับน้องรานีที่อายุ 2 เดือน การเลือกอาหารที่เหมาะสมเป็นสิ่งสำคัญมากเพื่อให้แน่ใจว่าน้องได้รับสารอาหารที่จำเป็นต่อการเจริญเติบโตและสุขภาพที่ดีค่ะ นี่คือคำแนะนำในการเลือกอาหารสำหรับน้องรานี:

1. **อาหารลูกแมว**:
   - ควรเลือกอาหารที่ออกแบบมาสำหรับลูกแมวโดยเฉพาะ เนื่องจากมีสารอาหารที่จำเป็นต่อการเจริญเติบโต เช่น โปรตีนสูง แคลเซียม และวิตามินต่าง ๆ
   - อาหารลูกแมวมีทั้งแบบแห้งและแบบเปียก คุณสามารถเลือกตามความสะดวกและความชอบของน้องรานีได้ค่ะ

2. **ปริมาณอาหาร**:
   - ปฏิบัติตามคำแนะนำบนบรรจุภัณฑ์ของอาหารที่เลือกใช้ โดยคำนึงถึงน้ำหนักและอายุของน้องรานี
   - แบ่งอาหารออกเป็นมื้อเล็ก ๆ หลายมื้อในแต่ละวัน เพื่อช่วยในการย่อยอาหารและป้องกันการกินมากเกินไป

3. **น้ำสะอาด**:
   - อย่าลืมให้น้ำสะอาดที่ไม่ใช่น้ำประปาเสมอ เพื่อสุขภาพที่ดีของน้อง

4. **หลีกเลี่ยงอาหารคน**:
   - หลีกเลี่ยงการให้อาหารคนหรือขนมที่มีไขมันสูง เพราะอาจทำให้น้องรานีมีน้ำหนักเกินและสุขภาพไม่ดี

Nhamaew Pet Ai หวังว่าคำแนะนำนี้จะช่วยให้น้องรานีมีสุขภาพดีและเติบโตอย่างแข็งแรงนะคะ หากคุณเปี๊ยกมีคำถามเพิ่มเติมหรือข้อสงสัยใด ๆ สามารถสอบถามได้เสมอค่ะ!""",
        detail={},
        image_url={}
    ),
    MessageContent(
        role="system",
        content="""Simple System""",
        detail={},
        image_url={}
    ),
        MessageContent(
        role="user",
        content="""รานีปวดตัว""",
        detail={},
        image_url={}
    )
]

# Create request body
request_body = RequestBody(
    requestid="unique_request_id",
    sessionid="session_12345",
    messages=messages
)

'''


# Function to search vet documents based on user query
def search_vet_documents(request_body):
    # Extract content where role is 'user'
    user_message = next((message.content for message in request_body.messages if message.role == "user"), None)
    print(user_message)
    if user_message:
        query = user_message  # Use content of user message as query
        print(f"query {query}")
        # Assuming vet_faiss.search_vet_doc() is a function you have access to
        search_results = VetFAISS.search_vet_doc(query, top_k=2)

        # Print the search results
        print("\n🔍 Search Results for:", query)
        for result in search_results:
            print(result)  # Modify based on the actual structure of the results
    else:
        print("No user message found to process")

# Call the function
search_vet_documents(request_body)
'''


system_prompt, request_messages_list, context = ExtractParameters._process_messages(messages)
print("=" * 50)
print("🔹 SYSTEM PROMPT:")
print(system_prompt)
print("=" * 50)

print("\n🔹 REQUEST MESSAGES:")
print(json.dumps(request_messages_list, indent=4, ensure_ascii=False))  # Pretty print JSON

print("\n🔹 CONTEXT:")
print(json.dumps(context, indent=4, ensure_ascii=False))  # Pretty print JSON
print("=" * 50)


question,history = ExtractParameters._extract_questions_and_history(request_messages_list)

print(f"Question {question}")
'''
print(f"History")
print(json.dumps(history, indent=4, ensure_ascii=False)) 
'''