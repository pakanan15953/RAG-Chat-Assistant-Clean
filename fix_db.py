import sqlite3
conn = sqlite3.connect(r'C:\Users\king\Documents\Project\RAG-Chat-Assistant-Clean\questions.db')
qa_c = conn.cursor()
def get_all_questions():
    qa_c.execute("SELECT id, question, answer, timestamp, correct_answer FROM questions")
    return qa_c.fetchall()
conn.close()