import sqlite3
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class VetDB:
    DB_NAME = r"C:\Users\banas\OneDrive\เอกสาร\GitHub\ChatbotCatStore\database\vet_doc.db"
    FAISS_INDEX_FILE = r"C:\Users\banas\OneDrive\เอกสาร\GitHub\ChatbotCatStore\vectordb\vet_doc_faiss_index.pkl"
    TFIDF_VECTORIZER_FILE = r"C:\Users\banas\OneDrive\เอกสาร\GitHub\ChatbotCatStore\vectordb\vet_doc_tfidf_vectorizer.pkl"

    @staticmethod
    def _connect():
        """Establish a database connection."""
        return sqlite3.connect(VetDB.DB_NAME)

    @staticmethod
    def create_tables():
        """Create tables for cat veterinary records."""
        with VetDB._connect() as conn:
            cursor = conn.cursor()

            # Cats Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS vet_doc (
                data_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symptoms TEXT NOT NULL,
                cause TEXT,
                diagnosis TEXT,
                treatment TEXT
            )
            """)

            conn.commit()
        print("✅ Tables created successfully!")

    @staticmethod
    def add_vet_doc(symptoms, cause, diagnosis, treatment):
        """Add a veterinary record to the database."""
        
        with VetDB._connect() as conn: 
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO vet_doc (symptoms, cause, diagnosis, treatment)
            VALUES (?, ?, ?, ?)
            """, (symptoms, cause, diagnosis, treatment))
            conn.commit()
        print(f"🐱 Symptoms: {symptoms}, Diagnosis: {diagnosis}, Treatment: {treatment}")


 

    @staticmethod
    def get_all_vet_doc():
        """Fetch all medical records."""
        with VetDB._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vet_doc")
            return cursor.fetchall()


class VetFAISS:
    def __init__(self):
        """Initialize FAISS and TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer()
        self.index = None

    def build_faiss_index(self):
        """Convert medical records into FAISS searchable index."""
        records = VetDB.get_all_vet_doc()
        if not records:
            print("⚠️ No medical records found!")
            return

        texts = [f"{symptoms} {diagnosis}" for _, _, _, symptoms, diagnosis in records]

        # Convert texts into TF-IDF vectors
        X = self.vectorizer.fit_transform(texts).toarray()
        dim = X.shape[1]

        # Create FAISS index
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(X, dtype=np.float32))

        # Save FAISS index and vectorizer
        with open(VetDB.FAISS_INDEX_FILE, "wb") as f:
            pickle.dump(self.index, f)
        with open(VetDB.TFIDF_VECTORIZER_FILE, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print("✅ FAISS index built and saved!")

    @staticmethod
    def search_vet_doc(query_text, top_k=3):
        """Search FAISS index using symptoms as query."""
        try:
            with open(VetDB.FAISS_INDEX_FILE, "rb") as f:
                index = pickle.load(f)
            with open(VetDB.TFIDF_VECTORIZER_FILE, "rb") as f:
                vectorizer = pickle.load(f)
        except FileNotFoundError:
            print("⚠️ FAISS index not found! Run `build_faiss_index()` first.")
            return []

        # Convert query into vector
        query_vec = vectorizer.transform([query_text]).toarray().astype(np.float32)

        # Perform search
        D, I = index.search(query_vec, top_k)

        # Get matching records
        records = VetDB.get_all_vet_doc()
        results = [records[i] for i in I[0] if i >= 0]

        return results



# 🔹 Example Usage
if __name__ == "__main__":
    # Initialize Database and FAISS
    VetDB.create_tables()
    vet_faiss = VetFAISS()

    # Add Vetdoc 
    VetDB.add_vet_doc("โรคหวัดแมว", 
                    "น้ำมูกไหล ไอ จาม ตาแดง น้ำตาไหล เบื่ออาหาร มีไข้", 
                    "เชื้อไวรัส เช่น ไวรัสเฮอร์ปีส์ (Feline Herpesvirus) หรือ คาลิซิไวรัส (Calicivirus) เชื้อแบคทีเรีย เช่น Chlamydophila felis", 
                    "ให้ยาปฏิชีวนะ (กรณีติดเชื้อแบคทีเรีย) ให้ยาลดน้ำมูก ลดไข้ ดูแลให้แมวอบอุ่นและได้รับสารอาหารเพียงพอ")

    VetDB.add_vet_doc("โรคเอดส์แมว (FIV)", 
                    "น้ำหนักลด ภูมิคุ้มกันต่ำ เป็นโรคติดเชื้อง่าย เช่น ติดเชื้อที่ผิวหนัง หรือลำไส้อักเสบ เหงือกอักเสบ เบื่ออาหาร", 
                    "ไวรัส FIV ติดต่อผ่านน้ำลาย เช่น การกัดกัน", 
                    "ไม่มีวิธีรักษาโดยตรง ต้องดูแลสุขภาพให้แข็งแรง ให้ยาปฏิชีวนะเมื่อมีการติดเชื้อแทรกซ้อน")

    VetDB.add_vet_doc("โรคมะเร็งเม็ดเลือดขาวในแมว (FeLV)", 
                    "อ่อนเพลีย ซีด น้ำหนักลด ภูมิคุ้มกันต่ำ ติดเชื้อง่าย มีก้อนเนื้องอก หรือมะเร็งในอวัยวะต่าง ๆ", 
                    "ไวรัส FeLV ติดต่อผ่านน้ำลาย น้ำมูก ปัสสาวะ และอุจจาระ", 
                    "ไม่มีวิธีรักษาโดยตรง ดูแลสุขภาพแมวให้แข็งแรง ฉีดวัคซีนป้องกันในแมวที่ยังไม่ติดเชื้อ")

    VetDB.add_vet_doc("โรคไตวายเรื้อรัง (CKD)", 
                    "ดื่มน้ำมาก ปัสสาวะบ่อย น้ำหนักลด อาเจียน เบื่ออาหาร ซึม", 
                    "พันธุกรรม หรือภาวะไตเสื่อมตามอายุ การกินอาหารที่มีเกลือสูง", 
                    "ควบคุมอาหารให้เหมาะกับแมวโรคไต ให้น้ำเกลือใต้ผิวหนังช่วยขับของเสีย ให้ยาควบคุมความดันโลหิตและบรรเทาอาการ")

    VetDB.add_vet_doc("โรคพยาธิในทางเดินอาหาร", 
                    "ถ่ายเหลว อาเจียน ท้องอืด น้ำหนักลด พบพยาธิในอุจจาระ", 
                    "พยาธิตัวกลม พยาธิตัวตืด หรือพยาธิปากขอ การกินอาหารหรือน้ำที่มีไข่พยาธิปนเปื้อน", 
                    "ให้ยาถ่ายพยาธิตามชนิดของพยาธิ ควบคุมสุขอนามัยและอาหารของแมว")

    VetDB.add_vet_doc("โรคเชื้อราในแมว (Ringworm)", 
                    "ขนร่วงเป็นวง ๆ ผิวหนังแดง คัน มีสะเก็ดแห้งเป็นขุย อาจลามไปติดคนหรือสัตว์อื่น", 
                    "เชื้อรากลุ่ม Dermatophytes ติดต่อผ่านการสัมผัสโดยตรง", 
                    "ให้ยาทาฆ่าเชื้อรา เช่น ยา Miconazole อาบน้ำด้วยแชมพูฆ่าเชื้อรา ในกรณีรุนแรงอาจต้องกินยาฆ่าเชื้อรา")

    VetDB.add_vet_doc("โรคเบาหวานในแมว (Feline Diabetes Mellitus)", 
                    "ดื่มน้ำเยอะ ปัสสาวะบ่อย น้ำหนักลดแต่กินอาหารมาก อ่อนเพลีย ซึม", 
                    "ภาวะดื้อต่ออินซูลินหรือการสร้างอินซูลินผิดปกติ อ้วน หรือมีพันธุกรรมที่เสี่ยง", 
                    "ควบคุมอาหารที่มีคาร์โบไฮเดรตต่ำ ฉีดอินซูลินตามที่สัตวแพทย์กำหนด ออกกำลังกายควบคุมน้ำหนัก")

    VetDB.add_vet_doc("โรคทองแดงในแมว (Feline Hepatic Lipidosis)", 
                  "เบื่ออาหาร น้ำหนักลดอย่างรวดเร็ว ตัวเหลือง อาเจียน อ่อนเพลีย ซึม", 
                  "เกิดจากภาวะขาดอาหารเป็นเวลานาน ทำให้ไขมันสะสมในตับมากเกินไป จนตับทำงานผิดปกติ", 
                  "กระตุ้นให้แมวกินอาหาร อาจต้องให้อาหารผ่านสายยาง กรณีรุนแรงต้องให้สารน้ำและยาฟื้นฟูตับโดยสัตวแพทย์")

    # Build FAISS Index
    vet_faiss.build_faiss_index()

    query = "อ่อนเพลียและเบื่ออาหาร"
    similar_cases = vet_faiss.search_vet_doc(query, top_k=2)

    print("\n🔍 Search Results for:", query)
    for case in similar_cases:
        if len(case) == 5:  # Ensure the tuple has the expected number of elements
            print(f"📋 Record ID: {case[0]}, Cat ID: {case[1]}, Symptoms: {case[2]}, Diagnosis: {case[3]}, Treatment: {case[4]}")
        else:
            print("⚠️ Unexpected tuple length:", case)
